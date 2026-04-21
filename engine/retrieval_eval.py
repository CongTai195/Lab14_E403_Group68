from __future__ import annotations

from typing import Any, Dict, List


class RetrievalEvaluator:
    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Core metrics (per-query)
    # ------------------------------------------------------------------

    def calculate_hit_rate(
        self,
        expected_ids: List[str],
        retrieved_ids: List[str],
        top_k: int | None = None,
    ) -> float:
        """Return 1.0 if at least one expected_id appears in the top-K retrieved list."""
        k = top_k if top_k is not None else self.top_k
        top_retrieved = retrieved_ids[:k]
        return 1.0 if any(eid in top_retrieved for eid in expected_ids) else 0.0

    def calculate_mrr(
        self,
        expected_ids: List[str],
        retrieved_ids: List[str],
    ) -> float:
        """Return 1/rank of the first relevant chunk in the retrieved list (0.0 if none)."""
        for rank, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id in expected_ids:
                return 1.0 / rank
        return 0.0

    # ------------------------------------------------------------------
    # Per-case scoring  (called by BenchmarkRunner per test case)
    # ------------------------------------------------------------------

    async def score(
        self,
        test_case: Dict[str, Any],
        response: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute retrieval metrics for a single (test_case, agent_response) pair.

        Expected keys:
          test_case  → "expected_retrieval_ids": list[str]
          response   → "metadata": {"sources": [{"chunk_id": str, ...}, ...]}
        """
        expected_ids: List[str] = test_case.get("expected_retrieval_ids") or []

        sources: List[Dict[str, Any]] = (
            response.get("metadata", {}).get("sources") or []
        )
        retrieved_ids: List[str] = [s["chunk_id"] for s in sources if s.get("chunk_id")]

        hit_rate = self.calculate_hit_rate(expected_ids, retrieved_ids)
        mrr = self.calculate_mrr(expected_ids, retrieved_ids)

        # Faithfulness: fraction of retrieved chunks that are in the expected set
        # (rough proxy — no LLM call needed)
        if retrieved_ids:
            expected_set = set(expected_ids)
            faithfulness = sum(1 for cid in retrieved_ids if cid in expected_set) / len(retrieved_ids)
        else:
            faithfulness = 0.0

        # Relevancy: whether the answer is non-empty (placeholder without LLM judge)
        answer: str = response.get("answer", "")
        relevancy = 0.0 if not answer.strip() else 1.0

        return {
            "faithfulness": round(faithfulness, 4),
            "relevancy": round(relevancy, 4),
            "retrieval": {
                "hit_rate": round(hit_rate, 4),
                "mrr": round(mrr, 4),
                "retrieved_ids": retrieved_ids,
                "expected_ids": expected_ids,
            },
        }

    # ------------------------------------------------------------------
    # Batch evaluation  (offline — dataset items already have retrieved_ids)
    # ------------------------------------------------------------------

    async def evaluate_batch(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate Hit Rate and MRR over a pre-collected dataset.

        Each item must have:
          - "expected_retrieval_ids": list[str]   (ground truth)
          - "retrieved_ids": list[str]             (what the retriever returned)
        """
        if not dataset:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0, "total": 0}

        hit_rates: List[float] = []
        mrrs: List[float] = []

        for item in dataset:
            expected = item.get("expected_retrieval_ids") or []
            retrieved = item.get("retrieved_ids") or []
            hit_rates.append(self.calculate_hit_rate(expected, retrieved))
            mrrs.append(self.calculate_mrr(expected, retrieved))

        avg_hit_rate = sum(hit_rates) / len(hit_rates)
        avg_mrr = sum(mrrs) / len(mrrs)

        return {
            "avg_hit_rate": round(avg_hit_rate, 4),
            "avg_mrr": round(avg_mrr, 4),
            "total": len(dataset),
            "hits": sum(1 for h in hit_rates if h > 0),
            "per_query": [
                {"hit_rate": h, "mrr": m}
                for h, m in zip(hit_rates, mrrs)
            ],
        }
