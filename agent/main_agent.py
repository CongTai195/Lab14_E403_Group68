from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from agent.retriever import FacebookPolicyRetriever, ROOT_DIR


GENERATION_MODEL = "gpt-4o-mini"
TOP_K = 5
MAX_RETRIES = 4
RETRY_DELAY_SECONDS = 2.0


class MainAgent:
    def __init__(self, top_k: int = TOP_K) -> None:
        load_dotenv(ROOT_DIR / ".env")
        self.name = "FacebookPolicyRAG-v1"
        self.top_k = top_k
        self.retriever = FacebookPolicyRetriever()
        self.openai_client = OpenAI()

    @staticmethod
    def _format_contexts(contexts: list[dict[str, Any]]) -> str:
        formatted = []
        for idx, context in enumerate(contexts, start=1):
            formatted.append(
                "\n".join(
                    [
                        f"[Context {idx}]",
                        f"chunk_id: {context['chunk_id']}",
                        f"title: {context['title']}",
                        f"url: {context['canonical_url']}",
                        f"heading: {context['heading_path']}",
                        context["text"],
                    ]
                )
            )
        return "\n\n".join(formatted)

    def _generate_answer(self, question: str, contexts: list[dict[str, Any]]) -> tuple[str, dict[str, int]]:
        if not contexts:
            return "Không đủ căn cứ trong kho tài liệu để trả lời câu hỏi này.", {}

        context_text = self._format_contexts(contexts)
        prompt = f"""
Bạn là trợ lý RAG chuyên trả lời về chính sách Facebook/Meta bằng tiếng Việt.
Chỉ được dùng các đoạn context bên dưới. Không suy đoán ngoài tài liệu.
Nếu context không đủ căn cứ, hãy nói rõ: "Không đủ căn cứ trong kho tài liệu để trả lời chính xác."
Khi trả lời, nêu ngắn gọn và có thể trích nguồn bằng chunk_id hoặc tiêu đề nếu hữu ích.

Câu hỏi:
{question}

Context:
{context_text}
""".strip()

        request_kwargs = {
            "model": GENERATION_MODEL,
            "input": prompt,
        }
        if GENERATION_MODEL.startswith("gpt-5"):
            request_kwargs["reasoning"] = {"effort": "xhigh"}

        for attempt in range(MAX_RETRIES):
            try:
                response = self.openai_client.responses.create(**request_kwargs)
                break
            except RateLimitError:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))

        usage = getattr(response, "usage", None)
        usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else {}
        return response.output_text, usage_dict

    async def query(self, question: str) -> dict[str, Any]:
        contexts = await asyncio.to_thread(self.retriever.search, question, self.top_k)
        answer, usage = await asyncio.to_thread(self._generate_answer, question, contexts)
        return {
            "answer": answer,
            "contexts": [context["text"] for context in contexts],
            "metadata": {
                "model": GENERATION_MODEL,
                "embedding_model": self.retriever.embedding_model,
                "tokens_used": usage,
                "sources": [
                    {
                        "chunk_id": context["chunk_id"],
                        "title": context["title"],
                        "canonical_url": context["canonical_url"],
                        "source_doc_id": context["source_doc_id"],
                        "source_file": context["source_file"],
                        "score": context["score"],
                        "rrf_score": context.get("rrf_score"),
                        "dense_rank": context.get("dense_rank"),
                        "dense_score": context.get("dense_score"),
                        "sparse_rank": context.get("sparse_rank"),
                        "sparse_score": context.get("sparse_score"),
                        "retrieval_method": context.get("retrieval_method"),
                    }
                    for context in contexts
                ],
            },
        }


if __name__ == "__main__":
    agent = MainAgent()

    async def test() -> None:
        response = await agent.query("Nội dung bắt nạt và quấy rối bị xử lý như thế nào?")
        print(response["answer"])
        print(response["metadata"]["sources"])

    asyncio.run(test())
