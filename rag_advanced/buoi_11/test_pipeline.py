import sys
import json

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from graph_rag import GraphRAGPipeline

def main():
    pipeline = GraphRAGPipeline()
    question = "Nghị định 15/2026/NĐ-CP căn cứ vào những luật nào và thay thế cho nghị định nào?"

    print("=" * 80)
    print(f"Câu hỏi: {question}")
    print("=" * 80)

    for hops in [0, 1]:
        print(f"\n>>> CHẠY PIPELINE VỚI MAX_HOPS = {hops} <<<")
        result = pipeline.query(question=question, top_k=2, max_hops=hops)
        
        print("\n[PROMPT ĐƯỢC XÂY DỰNG GỬI ĐẾN LLM]:")
        print("-" * 60)
        print(result["prompt"])
        print("-" * 60)
        
        print("\n[KẾT QUẢ SINH CÂU TRẢ LỜI TỪ GEMINI (Hoặc Trạng thái)]: ")
        print(f"Trạng thái: {result['status']}")
        print(f"Câu trả lời: {result['answer']}")

if __name__ == "__main__":
    main()
