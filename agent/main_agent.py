from __future__ import annotations

import asyncio
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from agent.retriever import FacebookPolicyRetriever, ROOT_DIR


GENERATION_MODEL = "gpt-4o-mini"
MAX_RETRIES = 4
RETRY_DELAY_SECONDS = 2.0


class BasePolicyAgent:
    def __init__(
        self,
        *,
        name: str,
        top_k: int,
        retrieval_mode: str,
        use_rerank: bool,
        candidate_k: int,
    ) -> None:
        load_dotenv(ROOT_DIR / ".env")
        self.name = name
        self.top_k = top_k
        self.retrieval_mode = retrieval_mode
        self.use_rerank = use_rerank
        self.candidate_k = candidate_k
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

    def _build_prompt(self, question: str, context_text: str) -> str:
        raise NotImplementedError

    def _format_final_answer(self, answer: str, contexts: list[dict[str, Any]]) -> str:
        return answer.strip()

    def _generate_answer(self, question: str, contexts: list[dict[str, Any]]) -> tuple[str, dict[str, int]]:
        if not contexts:
            return "Không đủ căn cứ trong kho tài liệu để trả lời câu hỏi này.", {}

        context_text = self._format_contexts(contexts)
        prompt = self._build_prompt(question, context_text)

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
        final_answer = self._format_final_answer(response.output_text, contexts)
        return final_answer, usage_dict

    async def query(self, question: str) -> dict[str, Any]:
        contexts = await asyncio.to_thread(
            self.retriever.search,
            question,
            self.top_k,
            self.candidate_k,
            self.retrieval_mode,
            self.use_rerank,
        )
        answer, usage = await asyncio.to_thread(self._generate_answer, question, contexts)
        return {
            "answer": answer,
            "contexts": [context["text"] for context in contexts],
            "retrieved_ids": [context["chunk_id"] for context in contexts],
            "metadata": {
                "agent_version": self.name,
                "model": GENERATION_MODEL,
                "embedding_model": self.retriever.embedding_model,
                "tokens_used": usage,
                "retrieval_mode": self.retrieval_mode,
                "use_rerank": self.use_rerank,
                "sources": [
                    {
                        "chunk_id": context["chunk_id"],
                        "title": context["title"],
                        "canonical_url": context["canonical_url"],
                        "source_doc_id": context["source_doc_id"],
                        "source_file": context["source_file"],
                        "score": context["score"],
                        "rrf_score": context.get("rrf_score"),
                        "rerank_score": context.get("rerank_score"),
                        "rerank_rank": context.get("rerank_rank"),
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


class AgentV1Base(BasePolicyAgent):
    def __init__(self) -> None:
        super().__init__(
            name="Agent_V1_Base",
            top_k=3,
            retrieval_mode="sparse",
            use_rerank=False,
            candidate_k=12,
        )

    def _build_prompt(self, question: str, context_text: str) -> str:
        return f"""
Bạn là trợ lý trả lời câu hỏi về chính sách Meta/Facebook bằng tiếng Việt.
Dùng context bên dưới để trả lời ngắn gọn. Nếu không chắc thì nói không rõ.

Câu hỏi:
{question}

Context:
{context_text}
""".strip()


class AgentV2Optimized(BasePolicyAgent):
    def __init__(self) -> None:
        super().__init__(
            name="Agent_V2_Optimized",
            top_k=6,
            retrieval_mode="hybrid",
            use_rerank=True,
            candidate_k=40,
        )

    def _build_prompt(self, question: str, context_text: str) -> str:
        return f"""
Bạn là trợ lý RAG chuyên trả lời câu hỏi về chính sách Facebook/Meta bằng tiếng Việt.

Yêu cầu bắt buộc:
1. Chỉ được dùng thông tin có trong context.
2. Nếu context không đủ căn cứ, phải nói rõ: "Không đủ căn cứ trong kho tài liệu để trả lời chính xác."
3. Trả lời trực tiếp câu hỏi trước, sau đó nêu ngắn gọn căn cứ.
4. Nếu có thể, trích 1-2 nguồn bằng chunk_id hoặc tiêu đề.
5. Không suy đoán ngoài tài liệu, không thêm thông tin không có trong context.

Định dạng mong muốn:
- Trả lời: ...
- Căn cứ: ...
- Nguồn: ...

Câu hỏi:
{question}

Context:
{context_text}
""".strip()

    def _format_final_answer(self, answer: str, contexts: list[dict[str, Any]]) -> str:
        clean_answer = answer.strip()
        top_sources = []
        for context in contexts[:2]:
            source = context.get("chunk_id") or context.get("title")
            if source:
                top_sources.append(source)
        if "Nguồn:" in clean_answer or not top_sources:
            return clean_answer
        return f"{clean_answer}\nNguồn: {', '.join(top_sources)}"


MainAgent = AgentV2Optimized


if __name__ == "__main__":
    agent = AgentV2Optimized()

    async def test() -> None:
        response = await agent.query("Nội dung bắt nạt và quấy rối bị xử lý như thế nào?")
        print(response["answer"])
        print(response["metadata"]["sources"])

    asyncio.run(test())
