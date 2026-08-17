"""
==============================================================================
EVALUATION FRAMEWORK MODULE (Buổi 08 - Metric Evaluation & Benchmarking)
Mục đích:
  Đánh giá hiệu năng truy xuất nâng cao qua các chỉ số:
  - Recall@K
  - MRR@K (Mean Reciprocal Rank)
  - nDCG@K (Normalized Discounted Cumulative Gain với binary relevance)
  - Latency Mean & P50 (Median Latency ms)
  so sánh độc lập 4 modes: bm25, semantic, hybrid, hybrid_rerank.
==============================================================================
"""
import sys
import os
import json
import time
import math
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import rag
import advanced_rag

ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)


def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Tính Recall@K với binary relevance.
    Recall@K = |Retrieved@K ∩ Relevant| / |Relevant|
    """
    if not relevant_ids:
        return 0.0
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    hits = len(top_k_retrieved & relevant_set)
    return round(hits / len(relevant_set), 4)


def calculate_mrr_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Tính MRR@K (Reciprocal Rank).
    MRR@K = 1 / first_match_rank (1-indexed) nếu tìm thấy trong top K, ngược lại 0.0.
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in relevant_set:
            return round(1.0 / rank, 4)
    return 0.0


def calculate_ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Tính nDCG@K với binary relevance.
    DCG@K = sum_{i=1}^K (rel_i / log2(i + 1))
    IDCG@K = sum_{i=1}^{min(K, |relevant|)} (1 / log2(i + 1))
    nDCG@K = DCG@K / IDCG@K
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    top_k_retrieved = retrieved_ids[:k]

    dcg = 0.0
    for i, cid in enumerate(top_k_retrieved, start=1):
        rel_i = 1.0 if cid in relevant_set else 0.0
        dcg += rel_i / math.log2(i + 1)

    idcg = 0.0
    ideal_hits = min(k, len(relevant_set))
    for i in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(i + 1)

    if idcg == 0.0:
        return 0.0

    return round(dcg / idcg, 4)


def calculate_metrics_summary(
    eval_results: List[Dict[str, Any]],
    k: int
) -> Dict[str, Any]:
    """
    Tính toán trung bình và median latency cho một mode.
    """
    if not eval_results:
        return {
            "recall_mean": 0.0,
            "mrr_mean": 0.0,
            "ndcg_mean": 0.0,
            "latency_mean_ms": 0.0,
            "latency_p50_ms": 0.0,
            "total_queries": 0
        }

    recalls = [r["recall"] for r in eval_results]
    mrrs = [r["mrr"] for r in eval_results]
    ndcgs = [r["ndcg"] for r in eval_results]
    latencies = sorted([r["latency_ms"] for r in eval_results])

    n = len(eval_results)
    rec_mean = round(sum(recalls) / n, 4)
    mrr_mean = round(sum(mrrs) / n, 4)
    ndcg_mean = round(sum(ndcgs) / n, 4)
    lat_mean = round(sum(latencies) / n, 2)

    # Median latency p50
    mid = n // 2
    if n % 2 == 1:
        lat_p50 = latencies[mid]
    else:
        lat_p50 = (latencies[mid - 1] + latencies[mid]) / 2.0
    lat_p50 = round(lat_p50, 2)

    return {
        "recall_mean": rec_mean,
        "mrr_mean": mrr_mean,
        "ndcg_mean": ndcg_mean,
        "latency_mean_ms": lat_mean,
        "latency_p50_ms": lat_p50,
        "total_queries": n
    }


def run_evaluation(
    eval_file: Path = BASE_DIR / "eval" / "questions.json",
    strategy: str = "hierarchical",
    k: int = 5,
    config: Optional[Dict[str, Any]] = None,
    storage_dir: Path = rag.STORAGE_DIR,
    input_dir: Path = rag.DEFAULT_INPUT_DIR,
    genai_client: Optional[Any] = None,
    reranker_fn: Optional[Any] = None,
    output_dir: Path = BASE_DIR / "reports"
) -> Dict[str, Any]:
    """
    Thực hiện benchmark tự động đánh giá 4 modes trên bộ câu hỏi eval.
    """
    if config is None:
        config = advanced_rag.get_advanced_config()

    if not eval_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file câu hỏi đánh giá: {eval_file}")

    with open(eval_file, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    has_human_review = any(q.get("needs_human_review", False) for q in questions_data)

    modes = ["bm25", "semantic", "hybrid", "hybrid_rerank"]
    mode_evaluations: Dict[str, List[Dict[str, Any]]] = {m: [] for m in modes}

    print(f"=== BẮT ĐẦU ĐÁNH GIÁ ADVANCED RAG ({len(questions_data)} CÂU HỎI) ===")
    print(f"Strategy: {strategy} | Eval K: {k}")
    if has_human_review:
        print("⚠️ CẢNH BÁO: Bộ câu hỏi chứa nhãn 'needs_human_review=true'. Báo cáo sẽ hiển thị cảnh báo.")

    # Tải trước corpus cho BM25
    load_res = rag.load_chunks(input_dir, strategy=strategy)
    all_chunks = load_res["chunks"]

    for q_idx, q_item in enumerate(questions_data, start=1):
        q_id = q_item.get("question_id", f"q{q_idx}")
        question = q_item["question"]
        relevant_ids = q_item.get("relevant_chunk_ids", [])

        print(f"\n[{q_idx}/{len(questions_data)}] Evaluated QID: {q_id} | '{question[:40]}...'")

        for m in modes:
            t0 = time.perf_counter()
            retrieved_ids = []
            try:
                if m == "bm25":
                    bm25_res = advanced_rag.run_bm25_search(question, all_chunks, candidate_k=k)
                    retrieved_ids = [c["chunk_id"] for c in bm25_res[:k]]

                elif m == "semantic":
                    sem_res = advanced_rag.run_semantic_search(
                        question, candidate_k=k, strategy=strategy,
                        storage_dir=storage_dir, genai_client=genai_client, config=config
                    )
                    retrieved_ids = [c["chunk_id"] for c in sem_res[:k]]

                elif m == "hybrid":
                    hyb_res = advanced_rag.run_hybrid_search(
                        question, strategy=strategy, storage_dir=storage_dir,
                        input_dir=input_dir, genai_client=genai_client, config=config
                    )
                    retrieved_ids = [c["chunk_id"] for c in hyb_res["candidates"][:k]]

                elif m == "hybrid_rerank":
                    hyb_res = advanced_rag.run_hybrid_search(
                        question, strategy=strategy, storage_dir=storage_dir,
                        input_dir=input_dir, genai_client=genai_client, config=config
                    )
                    try:
                        rr_res, _ = advanced_rag.run_rerank(question, hyb_res["candidates"], config=config, reranker_fn=reranker_fn)
                        retrieved_ids = [c["chunk_id"] for c in rr_res[:k]]
                    except Exception:
                        retrieved_ids = [c["chunk_id"] for c in hyb_res["candidates"][:k]]

            except Exception as ex:
                print(f"  [ERROR] Mode {m} failed for question '{q_id}': {ex}", file=sys.stderr)
                retrieved_ids = []

            t1 = time.perf_counter()
            lat_ms = round((t1 - t0) * 1000, 2)

            rec = calculate_recall_at_k(retrieved_ids, relevant_ids, k)
            mrr = calculate_mrr_at_k(retrieved_ids, relevant_ids, k)
            ndcg = calculate_ndcg_at_k(retrieved_ids, relevant_ids, k)

            mode_evaluations[m].append({
                "question_id": q_id,
                "retrieved_ids": retrieved_ids,
                "relevant_ids": relevant_ids,
                "recall": rec,
                "mrr": mrr,
                "ndcg": ndcg,
                "latency_ms": lat_ms
            })

    # Tổng hợp bảng metrics theo mode
    mode_metrics_summary = []
    for m in modes:
        summary = calculate_metrics_summary(mode_evaluations[m], k=k)
        summary["mode"] = m
        mode_metrics_summary.append(summary)

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": timestamp_str,
        "strategy": strategy,
        "k": k,
        "has_human_review_warning": has_human_review,
        "config": {
            "embedding_model": config["GEMINI_EMBEDDING_MODEL"],
            "reranker_model": config["RERANKER_MODEL"],
            "rrf_k": config["RRF_K"]
        },
        "mode_metrics": mode_metrics_summary,
        "detail_evaluations": mode_evaluations
    }

    # Lưu báo cáo vào thư mục reports/
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / f"eval_report_{timestamp_str}.json"
    latest_file = output_dir / "latest_eval_report.json"

    with open(report_file, "w", encoding="utf-8") as rf:
        json.dump(report, rf, ensure_ascii=False, indent=2)

    with open(latest_file, "w", encoding="utf-8") as lf:
        json.dump(report, lf, ensure_ascii=False, indent=2)

    print(f"\n=== ĐÃ LƯU BÁO CÁO ĐÁNH GIÁ VÀO: {report_file} ===")
    return report


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Buổi 08 - Advanced RAG Evaluation Framework")
    parser.add_argument("--strategy", type=str, default="hierarchical", choices=["fixed-size", "semantic", "hierarchical"])
    parser.add_argument("--k", type=int, default=5, help="Giá trị K cho Recall@K, MRR@K, nDCG@K (default: 5)")
    parser.add_argument("--eval-file", type=str, default=str(BASE_DIR / "eval" / "questions.json"))

    args = parser.parse_args()

    try:
        rep = run_evaluation(
            eval_file=Path(args.eval_file),
            strategy=args.strategy,
            k=args.k
        )

        print("\n=== KẾT QUẢ ĐÁNH GIÁ CHỈ SỐ METRICS THỰC TẾ ===")
        print(f"{'Mode':<15} | {'Recall@K':<10} | {'MRR@K':<10} | {'nDCG@K':<10} | {'Mean Latency':<14} | {'P50 Latency':<12}")
        print("-" * 80)
        for m_info in rep["mode_metrics"]:
            print(f"{m_info['mode']:<15} | {m_info['recall_mean']:<10.4f} | {m_info['mrr_mean']:<10.4f} | {m_info['ndcg_mean']:<10.4f} | {m_info['latency_mean_ms']:<12.2f}ms | {m_info['latency_p50_ms']:<10.2f}ms")

        if rep["has_human_review_warning"]:
            print("\n⚠️ CHÚ Ý: Bộ câu hỏi gold labels có chứa 'needs_human_review=true'. Không tuyên bố mode chiến thắng chính thức cho đến khi nghiệm thu người.")

    except Exception as ex:
        print(f"LỖI THỰC THI EVALUATION: {ex}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
