import json
import asyncio
import os
from typing import List, Dict

# Giả lập việc gọi LLM để tạo dữ liệu (Students will implement this)
async def generate_qa_from_text(text: str, num_pairs: int = 5) -> List[Dict]:
    """
    TODO: Sử dụng OpenAI/Anthropic API để tạo các cặp (Question, Expected Answer, Context)
    từ đoạn văn bản cho trước.
    Yêu cầu: Tạo ít nhất 1 câu hỏi 'lừa' (adversarial) hoặc cực khó.
    """
    print(f"Generating {num_pairs} QA pairs from text...")
    # Placeholder implementation
    return [
        {
            "question": "Câu hỏi mẫu từ tài liệu?",
            "expected_answer": "Câu trả lời kỳ vọng mẫu.",
            "context": text[:200],
            "metadata": {"difficulty": "easy", "type": "fact-check"}
        }
    ]

async def main():
    raw_text = "AI Evaluation là một quy trình kỹ thuật nhằm đo lường chất lượng..."
    qa_pairs = await generate_qa_from_text(raw_text)
    
    with open("data/golden_set.jsonl", "w", encoding="utf-8") as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print("Done! Saved to data/golden_set.jsonl")

if __name__ == "__main__":
    asyncio.run(main())
