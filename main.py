import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any
from engine.runner import BenchmarkRunner
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from agent.main_agent import AgentV1Base, AgentV2Optimized


@dataclass(frozen=True)
class AgentVersionConfig:
    agent_class: type
    top_k: int


VERSION_CONFIGS = {
    "Agent_V1_Base": AgentVersionConfig(
        agent_class=AgentV1Base,
        top_k=3,
    ),
    "Agent_V2_Optimized": AgentVersionConfig(
        agent_class=AgentV2Optimized,
        top_k=6,
    ),
}

BENCHMARK_BATCH_SIZE = 2
GPT_4O_MINI_INPUT_COST_PER_1M = 0.15
GPT_4O_MINI_OUTPUT_COST_PER_1M = 0.60
TEXT_EMBEDDING_3_LARGE_COST_PER_1M = 0.13


class ExpertEvaluator:
    def __init__(self, config: AgentVersionConfig):
        self.config = config
        self.retrieval_evaluator = RetrievalEvaluator(top_k=config.top_k)

    async def score(self, case: dict[str, Any], resp: dict[str, Any]) -> dict[str, Any]:
        return await self.retrieval_evaluator.score(case, resp)

def _format_benchmark_items(results: list[dict]) -> list[dict]:
    formatted_results = []
    for item in results:
        ragas = item.get("ragas", {})
        retrieval = ragas.get("retrieval", {})
        formatted_results.append(
            {
                "test_case": item.get("test_case"),
                "agent_response": item.get("agent_response"),
                "latency": item.get("latency"),
                "ragas": {
                    "hit_rate": retrieval.get("hit_rate", 0.0),
                    "mrr": retrieval.get("mrr", 0.0),
                    "faithfulness": ragas.get("faithfulness", 0.0),
                    "relevancy": ragas.get("relevancy", 0.0),
                },
                "judge": item.get("judge", {}),
                "status": item.get("status"),
            }
        )
    return formatted_results


def _safe_average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _extract_total_tokens(usage: Any) -> int:
    if not usage:
        return 0
    if isinstance(usage, dict):
        direct_total = usage.get("total_tokens")
        if isinstance(direct_total, (int, float)):
            return int(direct_total)

        input_tokens = usage.get("input_tokens") or usage.get("promptTokenCount") or 0
        output_tokens = usage.get("output_tokens") or usage.get("candidatesTokenCount") or 0
        if isinstance(input_tokens, (int, float)) or isinstance(output_tokens, (int, float)):
            return int(input_tokens) + int(output_tokens)

        nested_input = usage.get("input_tokens_details") or {}
        nested_output = usage.get("output_tokens_details") or {}
        nested_total = 0
        if isinstance(nested_input, dict):
            nested_total += sum(int(v) for v in nested_input.values() if isinstance(v, (int, float)))
        if isinstance(nested_output, dict):
            nested_total += sum(int(v) for v in nested_output.values() if isinstance(v, (int, float)))
        return nested_total
    return 0


def _extract_input_tokens(usage: Any) -> int:
    if isinstance(usage, dict):
        value = usage.get("input_tokens") or usage.get("promptTokenCount") or 0
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _extract_output_tokens(usage: Any) -> int:
    if isinstance(usage, dict):
        value = usage.get("output_tokens") or usage.get("candidatesTokenCount") or 0
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _estimate_openai_cost_usd(results: list[dict]) -> float:
    generation_input_tokens = 0
    generation_output_tokens = 0
    embedding_tokens = 0
    judge_input_tokens = 0
    judge_output_tokens = 0

    for item in results:
        agent_usage = (item.get("_agent_metadata") or {}).get("tokens_used") or {}
        generation_input_tokens += _extract_input_tokens(agent_usage)
        generation_output_tokens += _extract_output_tokens(agent_usage)

        sources = (item.get("_agent_metadata") or {}).get("sources") or []
        for source in sources:
            embedding_tokens += int(source.get("embedding_tokens", 0) or 0)

        judge = item.get("judge", {})
        individual_results = judge.get("individual_results", {})
        openai_judge = individual_results.get("gpt-4o-mini", {})
        judge_usage = openai_judge.get("usage", {})
        judge_input_tokens += _extract_input_tokens(judge_usage)
        judge_output_tokens += _extract_output_tokens(judge_usage)

    generation_cost = (
        generation_input_tokens * GPT_4O_MINI_INPUT_COST_PER_1M
        + generation_output_tokens * GPT_4O_MINI_OUTPUT_COST_PER_1M
    ) / 1_000_000
    judge_cost = (
        judge_input_tokens * GPT_4O_MINI_INPUT_COST_PER_1M
        + judge_output_tokens * GPT_4O_MINI_OUTPUT_COST_PER_1M
    ) / 1_000_000
    embedding_cost = (embedding_tokens * TEXT_EMBEDDING_3_LARGE_COST_PER_1M) / 1_000_000
    return round(generation_cost + judge_cost + embedding_cost, 6)


def _build_performance_report(results: list[dict], runtime_seconds: float) -> dict[str, Any]:
    latencies = [float(item.get("latency", 0.0) or 0.0) for item in results]
    agent_total_tokens = 0
    judge_openai_total_tokens = 0
    judge_nano_total_tokens = 0

    for item in results:
        agent_usage = (item.get("_agent_metadata") or {}).get("tokens_used") or {}
        agent_total_tokens += _extract_total_tokens(agent_usage)

        individual_results = (item.get("judge", {}) or {}).get("individual_results", {})
        judge_openai_total_tokens += _extract_total_tokens(
            (individual_results.get("gpt-4o-mini", {}) or {}).get("usage", {})
        )
        judge_nano_total_tokens += _extract_total_tokens(
            (individual_results.get("gpt-4.1-nano", {}) or {}).get("usage", {})
        )

    return {
        "runtime_seconds": round(runtime_seconds, 4),
        "avg_latency_seconds": round(_safe_average(latencies), 4),
        "max_latency_seconds": round(max(latencies) if latencies else 0.0, 4),
        "throughput_cases_per_minute": round((len(results) / runtime_seconds) * 60, 4) if runtime_seconds > 0 else 0.0,
        "token_usage": {
            "agent_total_tokens": agent_total_tokens,
            "judge_openai_total_tokens": judge_openai_total_tokens,
            "judge_nano_total_tokens": judge_nano_total_tokens,
        },
        "estimated_openai_cost_usd": _estimate_openai_cost_usd(results),
    }

async def run_benchmark_with_results(agent_version: str):
    print(f"🚀 Khởi động Benchmark cho {agent_version}...")
    version_config = VERSION_CONFIGS.get(agent_version, VERSION_CONFIGS["Agent_V2_Optimized"])
    benchmark_start = time.perf_counter()

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return None, None

    print(f"📚 {agent_version}: chạy {len(dataset)} test cases với batch_size={BENCHMARK_BATCH_SIZE}", flush=True)

    runner = BenchmarkRunner(
        version_config.agent_class(),
        ExpertEvaluator(version_config),
        LLMJudge(),
    )
    results = await runner.run_all(dataset, batch_size=BENCHMARK_BATCH_SIZE)

    total = len(results)
    summary = {
        "metadata": {
            "version": agent_version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "top_k": version_config.top_k,
        },
        "metrics": {
            "avg_score": _safe_average([r["judge"]["final_score"] for r in results]),
            "hit_rate": _safe_average([r["ragas"]["retrieval"]["hit_rate"] for r in results]),
            "mrr": _safe_average([r["ragas"]["retrieval"]["mrr"] for r in results]),
            "faithfulness": _safe_average([r["ragas"]["faithfulness"] for r in results]),
            "relevancy": _safe_average([r["ragas"]["relevancy"] for r in results]),
            "agreement_rate": _safe_average([r["judge"]["agreement_rate"] for r in results]),
        },
        "performance": _build_performance_report(results, time.perf_counter() - benchmark_start),
    }
    return results, summary

async def run_benchmark(version):
    _, summary = await run_benchmark_with_results(version)
    return summary

async def main():
    v1_task = run_benchmark_with_results("Agent_V1_Base")
    v2_task = run_benchmark_with_results("Agent_V2_Optimized")
    (v1_results, v1_summary), (v2_results, v2_summary) = await asyncio.gather(v1_task, v2_task)
    
    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    delta = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    print(f"V1 Score: {v1_summary['metrics']['avg_score']}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}")

    decision = "APPROVE" if delta > 0 else "BLOCK"
    summary_report = {
        "metadata": {
            "total": v1_summary["metadata"]["total"],
            "version": "BASELINE (V1)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "versions_compared": ["V1", "V2"],
        },
        "metrics": {
            "avg_score": v1_summary["metrics"]["avg_score"],
            "hit_rate": v1_summary["metrics"]["hit_rate"],
            "mrr": v1_summary["metrics"]["mrr"],
            "agreement_rate": v1_summary["metrics"]["agreement_rate"],
        },
        "regression": {
            "v1": {
                "score": v1_summary["metrics"]["avg_score"],
                "hit_rate": v1_summary["metrics"]["hit_rate"],
                "mrr": v1_summary["metrics"]["mrr"],
                "judge_agreement": v1_summary["metrics"]["agreement_rate"],
            },
            "v2": {
                "score": v2_summary["metrics"]["avg_score"],
                "hit_rate": v2_summary["metrics"]["hit_rate"],
                "mrr": v2_summary["metrics"]["mrr"],
                "judge_agreement": v2_summary["metrics"]["agreement_rate"],
            },
            "decision": decision,
        },
        "performance": {
            "v1": v1_summary.get("performance", {}),
            "v2": v2_summary.get("performance", {}),
            "pipeline_parallel": True,
            "benchmark_batch_size": BENCHMARK_BATCH_SIZE,
        },
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "v1": _format_benchmark_items(v1_results),
                "v2": _format_benchmark_items(v2_results),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if delta > 0:
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")

if __name__ == "__main__":
    asyncio.run(main())
