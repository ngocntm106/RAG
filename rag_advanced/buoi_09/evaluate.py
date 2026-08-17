"""Buổi 09 — Evaluator Engine.

Đánh giá 4 pipeline modes (single_flat, multi_flat, single_parent, multi_parent)
trên tập câu hỏi test eval/questions.json (retrieval-only, KHÔNG gọi Answer Generation).

Metrics:
  - Child Recall@K, Parent Recall@K
  - MRR@K, nDCG@K
  - Latency (mean, p50)
  - Context Chars & Expansion Factor
  - Call counts (Generation vs Embedding)

CLI:
  python evaluate.py --questions eval/questions.json --k 3
"""

from pathlib import Path
import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import hierarchical_rag

REPORTS_DIR = BASE_DIR / "reports"


def load_eval_questions(filepath: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load câu hỏi đánh giá từ eval/questions.json."""
    if filepath is None:
        filepath = BASE_DIR / "eval" / "questions.json"
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Không tìm thấy file câu hỏi đánh giá tại: {filepath}")
    return json.loads(filepath.read_text(encoding="utf-8"))


def _compute_mrr(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Tính Reciprocal Rank (RR@K)."""
    rel_set = set(relevant_ids)
    if not rel_set:
        return 0.0
    for idx, item_id in enumerate(retrieved_ids[:k], start=1):
        if item_id in rel_set:
            return 1.0 / idx
    return 0.0


def _compute_ndcg(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Tính nDCG@K (Binary relevance)."""
    rel_set = set(relevant_ids)
    if not rel_set:
        return 0.0

    dcg = 0.0
    for idx, item_id in enumerate(retrieved_ids[:k], start=1):
        rel = 1.0 if item_id in rel_set else 0.0
        dcg += rel / math.log2(idx + 1)

    # Ideal DCG
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel_set), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_mode(
    mode: str,
    questions: List[Dict[str, Any]],
    k: int = 3,
    config: Optional[Dict[str, Any]] = None,
    registry_override: Optional[Tuple[List[Dict], List[Dict], Dict]] = None,
    query_generator_fn: Optional[Any] = None,
    hybrid_retriever_fn: Optional[Any] = None,
    reranker_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Đánh giá 1 mode trên toàn bộ tập câu hỏi."""
    if config is None:
        config = hierarchical_rag.load_hierarchical_config()

    per_q_results = []
    latencies = []
    child_recalls = []
    parent_recalls = []
    mrrs = []
    ndcgs = []
    gen_calls_tot = 0
    emb_calls_tot = 0

    for q_item in questions:
        qtext = q_item["question"]
        rel_children = q_item.get("relevant_child_ids", [])
        rel_parents = q_item.get("relevant_parent_ids", [])

        t0 = time.perf_counter()
        res = hierarchical_rag.query_hierarchical_rag(
            question=qtext,
            mode=mode,
            config=config,
            registry_override=registry_override,
            query_generator_fn=query_generator_fn,
            hybrid_retriever_fn=hybrid_retriever_fn,
            reranker_fn=reranker_fn,
            answer_generator_fn=lambda p, c: None,  # Retrieval-only
        )
        lat_ms = (time.perf_counter() - t0) * 1000
        latencies.append(lat_ms)

        acc_ev = res.get("accepted_evidence", [])
        ret_parent_ids = [e["parent_id"] for e in acc_ev if "parent_id" in e]
        ret_child_ids = [e["child_id"] for e in acc_ev if "child_id" in e]

        # Calculate metrics
        c_recall = (
            len(set(ret_child_ids[:k]) & set(rel_children)) / len(rel_children)
            if rel_children
            else 1.0 if not ret_child_ids else 0.0
        )
        p_recall = (
            len(set(ret_parent_ids[:k]) & set(rel_parents)) / len(rel_parents)
            if rel_parents
            else 1.0 if not ret_parent_ids else 0.0
        )

        mrr_val = _compute_mrr(ret_parent_ids if "parent" in mode else ret_child_ids, rel_parents if "parent" in mode else rel_children, k)
        ndcg_val = _compute_ndcg(ret_parent_ids if "parent" in mode else ret_child_ids, rel_parents if "parent" in mode else rel_children, k)

        child_recalls.append(c_recall)
        parent_recalls.append(p_recall)
        mrrs.append(mrr_val)
        ndcgs.append(ndcg_val)

        api_counts = res.get("api_call_counts", {})
        gen_calls_tot += api_counts.get("generation_calls", 0)
        emb_calls_tot += api_counts.get("embedding_calls", 0)

        per_q_results.append({
            "question_id": q_item.get("question_id", q_item.get("id")),
            "question": qtext,
            "status": res["status"],
            "parent_recall_at_k": round(p_recall, 4),
            "child_recall_at_k": round(c_recall, 4),
            "mrr_at_k": round(mrr_val, 4),
            "ndcg_at_k": round(ndcg_val, 4),
            "latency_ms": round(lat_ms, 2),
        })

    latencies.sort()
    mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
    p50_lat = latencies[len(latencies) // 2] if latencies else 0.0

    return {
        "mode": mode,
        "sample_count": len(questions),
        "k": k,
        "mean_child_recall": round(sum(child_recalls) / len(child_recalls), 4) if child_recalls else 0.0,
        "mean_parent_recall": round(sum(parent_recalls) / len(parent_recalls), 4) if parent_recalls else 0.0,
        "mean_mrr": round(sum(mrrs) / len(mrrs), 4) if mrrs else 0.0,
        "mean_ndcg": round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else 0.0,
        "mean_latency_ms": round(mean_lat, 2),
        "p50_latency_ms": round(p50_lat, 2),
        "total_generation_calls": gen_calls_tot,
        "total_embedding_calls": emb_calls_tot,
        "per_question_results": per_q_results,
    }


def run_full_evaluation(
    questions_file: Optional[Path] = None,
    k: int = 3,
    config: Optional[Dict[str, Any]] = None,
    registry_override: Optional[Tuple[List[Dict], List[Dict], Dict]] = None,
    query_generator_fn: Optional[Any] = None,
    hybrid_retriever_fn: Optional[Any] = None,
    reranker_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Chạy đánh giá full 4 modes và xuất báo cáo JSON atomically."""
    questions = load_eval_questions(questions_file)
    if config is None:
        config = hierarchical_rag.load_hierarchical_config()

    modes = ["single_flat", "multi_flat", "single_parent", "multi_parent"]
    mode_evals = {}

    for m in modes:
        mode_evals[m] = evaluate_mode(
            mode=m,
            questions=questions,
            k=k,
            config=config,
            registry_override=registry_override,
            query_generator_fn=query_generator_fn,
            hybrid_retriever_fn=hybrid_retriever_fn,
            reranker_fn=reranker_fn,
        )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_identity": hierarchical_rag._config_identity(config),
        "schema_version": "1.0",
        "eval_k": k,
        "modes_evaluated": mode_evals,
        "human_review_warning": "Ground truth gold labels có chứa nhãn cần human review. Không kết luận mode thắng độc tôn khi chưa đối chiếu chuyên gia pháp lý.",
    }

    # Save report atomically
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_file = REPORTS_DIR / f"report_{ts_str}.json"
    latest_file = REPORTS_DIR / "latest_report.json"

    hierarchical_rag._atomic_write_json(report_file, report)
    hierarchical_rag._atomic_write_json(latest_file, report)

    return report


def main(argv: Optional[List[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(description="Buổi 09 RAG Evaluator Engine")
    parser.add_argument("--questions", default=None, help="Đường dẫn file questions.json")
    parser.add_argument("--k", type=int, default=3, help="K value cho Recall/MRR/nDCG")

    args = parser.parse_args(argv)

    print(f"Đang chạy Benchmark Evaluation (K={args.k})...\n")
    try:
        report = run_full_evaluation(
            questions_file=Path(args.questions) if args.questions else None,
            k=args.k,
        )
    except Exception as exc:
        print(f"EVALUATION ERROR: {exc}", file=sys.stderr)
        return 1

    print("══ EVALUATION BENCHMARK SUMMARY ══════════════════════════════════════════════════")
    print(f"{'Mode':<16} {'Child Rec@K':<14} {'Parent Rec@K':<14} {'MRR@K':<10} {'nDCG@K':<10} {'Latency (ms)':<14}")
    print("-" * 80)

    for m, data in report["modes_evaluated"].items():
        print(f"{m:<16} {data['mean_child_recall']:<14.4f} {data['mean_parent_recall']:<14.4f} {data['mean_mrr']:<10.4f} {data['mean_ndcg']:<10.4f} {data['mean_latency_ms']:<14.1f}")

    print("═" * 80)
    print(f"Báo cáo đã ghi thành công vào: {REPORTS_DIR / 'latest_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
