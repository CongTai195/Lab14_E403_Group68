from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from agent.retriever import ROOT_DIR


OPENAI_JUDGE_MODEL_A = "gpt-4o-mini"
OPENAI_JUDGE_MODEL_B = "gpt-4.1-nano"
MAX_JUDGE_RETRIES = 3
RETRY_DELAY_SECONDS = 2.0


class LLMJudge:
    def __init__(self) -> None:
        load_dotenv(ROOT_DIR / ".env")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_client = OpenAI() if self.openai_api_key else None

    @staticmethod
    def _build_judge_prompt(question: str, answer: str, ground_truth: str) -> str:
        return f"""
Bạn là model judge chấm câu trả lời của một AI assistant.

Hãy so sánh:
- Câu hỏi
- Câu trả lời của agent
- Ground truth

Tiêu chí:
1. Accuracy: đúng sai so với ground truth
2. Completeness: có đủ ý chính hay không
3. Grounding: có bám đúng nội dung câu hỏi không

Chấm điểm nguyên từ 0 đến 5:
- 0: hoàn toàn sai / không liên quan
- 1: rất kém
- 2: thiếu nhiều
- 3: tạm đúng
- 4: tốt
- 5: rất tốt và đầy đủ

Phải trả về đúng JSON với schema:
{{
  "score": <integer 0-5>,
  "reasoning": "<ngắn gọn, 1-3 câu>"
}}

Câu hỏi:
{question}

Câu trả lời của agent:
{answer}

Ground truth:
{ground_truth}
""".strip()

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        text = text.strip()
        text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("No JSON object found in judge response.")

    @staticmethod
    def _extract_score_reasoning_fallback(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        score_match = re.search(r'(?:"score"|score)\s*[:=]\s*"?([0-5])"?', cleaned, re.IGNORECASE)
        if not score_match:
            score_match = re.search(r"\b([0-5])\b", cleaned)
        score = int(score_match.group(1)) if score_match else 0

        reasoning_match = re.search(
            r'(?:"reasoning"|reasoning)\s*[:=]\s*"?(.+?)"?$',
            cleaned,
            re.IGNORECASE | re.DOTALL,
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else cleaned[:500].strip()
        return {"score": score, "reasoning": reasoning or "No reasoning returned."}

    @classmethod
    def _parse_judge_text(cls, text: str) -> dict[str, Any]:
        try:
            return cls._extract_json(text)
        except Exception:
            return cls._extract_score_reasoning_fallback(text)

    @staticmethod
    def _normalize_result(result: dict[str, Any], model_name: str) -> dict[str, Any]:
        raw_score = result.get("score", 0)
        try:
            score = int(raw_score)
        except (TypeError, ValueError):
            score = 0
        score = max(0, min(5, score))
        reasoning = str(result.get("reasoning", "")).strip() or f"No reasoning from {model_name}."
        return {"score": score, "reasoning": reasoning}

    def _judge_openai(self, model_name: str, question: str, answer: str, ground_truth: str) -> dict[str, Any]:
        if not self.openai_client:
            return {"score": 0, "reasoning": "Error OpenAI: Missing OPENAI_API_KEY", "usage": {}}

        prompt = self._build_judge_prompt(question, answer, ground_truth)
        request_kwargs = {"model": model_name, "input": prompt}

        for attempt in range(MAX_JUDGE_RETRIES):
            try:
                response = self.openai_client.responses.create(**request_kwargs)
                parsed = self._parse_judge_text(response.output_text)
                normalized = self._normalize_result(parsed, model_name)
                usage = getattr(response, "usage", None)
                normalized["usage"] = usage.model_dump() if hasattr(usage, "model_dump") else {}
                return normalized
            except RateLimitError as exc:
                if attempt == MAX_JUDGE_RETRIES - 1:
                    return {"score": 0, "reasoning": f"Error OpenAI: {exc}", "usage": {}}
                time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
            except Exception as exc:
                return {"score": 0, "reasoning": f"Error OpenAI: {exc}", "usage": {}}
        return {"score": 0, "reasoning": "Error OpenAI: Unknown failure", "usage": {}}

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        result_a, result_b = await asyncio.gather(
            asyncio.to_thread(self._judge_openai, OPENAI_JUDGE_MODEL_A, question, answer, ground_truth),
            asyncio.to_thread(self._judge_openai, OPENAI_JUDGE_MODEL_B, question, answer, ground_truth),
        )

        score_a = result_a["score"]
        score_b = result_b["score"]
        diff = abs(score_a - score_b)
        final_score = (score_a + score_b) / 2.0
        agreement_rate = 1.0 / (1.0 + diff)
        status = "conflict" if diff > 1 else "consensus"

        return {
            "final_score": round(final_score, 4),
            "agreement_rate": round(agreement_rate, 4),
            "individual_results": {
                OPENAI_JUDGE_MODEL_A: result_a,
                OPENAI_JUDGE_MODEL_B: result_b,
            },
            "status": status,
        }

    async def check_position_bias(self, response_a: str, response_b: str) -> Dict[str, Any]:
        return {"supported": False, "reason": "Position-bias check is not implemented in this lab version."}
