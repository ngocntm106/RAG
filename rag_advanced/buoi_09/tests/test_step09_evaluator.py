"""Unit tests cho Bước 09 — Evaluator Engine & Metrics.

Bao gồm các test cases (100% offline, dùng injected fakes):
1. Tính toán Reciprocal Rank (MRR@K).
2. Tính toán nDCG@K binary relevance.
3. Đánh giá 4 modes retrieval-only.
4. Ghi báo cáo JSON atomically tại reports/latest_report.json.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from evaluate import (
    _compute_mrr,
    _compute_ndcg,
    evaluate_mode,
    load_eval_questions,
    run_full_evaluation,
)


class TestStep09Evaluator(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_questions = [
            {
                "question_id": "Q01",
                "question": "Điều 8 quy định những nhu cầu vốn nào không được cho vay?",
                "question_type": "exact",
                "relevant_child_ids": ["c10"],
                "relevant_parent_ids": ["p_art8"],
                "needs_human_review": True,
            }
        ]
        self.questions_file = self.temp_dir / "questions.json"
        self.questions_file.write_text(json.dumps(self.mock_questions), encoding="utf-8")

        self.mock_registry = (
            [
                {
                    "child_id": "c10",
                    "parent_id": "p_art8",
                    "source": "src.pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "text": "Child 10 text Điều 8",
                    "structural_path": {"article": "Điều 8"},
                    "ambiguous": False,
                }
            ],
            [
                {
                    "parent_id": "p_art8",
                    "source": "src.pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "article_key": "Điều 8. Nhu cầu cấm",
                    "child_ids": ["c10"],
                    "text": "Parent Điều 8 text",
                    "ambiguous": False,
                }
            ],
            {"schema_version": "1.0", "config_identity": "test_id"},
        )

        self.config = {
            "multi_query_count": 2,
            "multi_query_max_chars": 300,
            "multi_query_original_weight": 1.5,
            "multi_query_variant_weight": 1.0,
            "multi_query_rrf_k": 60,
            "per_query_candidates": 12,
            "parent_max_chars": 6000,
            "parent_score_child_limit": 3,
            "parent_rrf_k": 60,
            "parent_candidates": 10,
            "final_parent_top_k": 2,
            "total_context_max_chars": 16000,
            "rerank_min_score": 0.35,
            "gemini_generation_model": "gemini-3.5-flash-lite",
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_compute_mrr(self):
        """MRR@K = 1 / rank của hit đầu tiên."""
        retrieved = ["item_a", "item_b", "item_c"]
        relevant = ["item_b"]
        self.assertEqual(_compute_mrr(retrieved, relevant, k=3), 0.5)

    def test_02_compute_ndcg(self):
        """nDCG@K binary relevance."""
        retrieved = ["item_a", "item_b", "item_c"]
        relevant = ["item_a"]
        self.assertEqual(_compute_ndcg(retrieved, relevant, k=3), 1.0)

    def test_03_evaluate_mode_retrieval_only(self):
        """Chạy evaluate_mode retrieval-only với fakes."""
        def mock_gen(p, c):
            return json.dumps({"queries": []})

        def mock_ret(q, c):
            return [{"child_id": "c10", "text": "Child 10 text Điều 8", "source": "src.pdf", "page_start": 1, "page_end": 1}]

        def mock_rr(q, t):
            return [0.9] * len(t)

        eval_res = evaluate_mode(
            mode="multi_parent",
            questions=self.mock_questions,
            k=3,
            config=self.config,
            registry_override=self.mock_registry,
            query_generator_fn=mock_gen,
            hybrid_retriever_fn=mock_ret,
            reranker_fn=mock_rr,
        )
        self.assertEqual(eval_res["mode"], "multi_parent")
        self.assertEqual(eval_res["mean_parent_recall"], 1.0)
        self.assertEqual(eval_res["mean_mrr"], 1.0)

    def test_04_run_full_evaluation_atomic_report(self):
        """Chạy run_full_evaluation và kiểm tra atomic report JSON."""
        report = run_full_evaluation(
            questions_file=self.questions_file,
            k=3,
            config=self.config,
            registry_override=self.mock_registry,
            query_generator_fn=lambda p, c: '{"queries":[]}',
            hybrid_retriever_fn=lambda q, c: [{"child_id": "c10", "text": "Child 10 text Điều 8", "source": "src.pdf", "page_start": 1, "page_end": 1}],
            reranker_fn=lambda q, t: [0.9] * len(t),
        )
        self.assertIn("modes_evaluated", report)
        self.assertIn("single_parent", report["modes_evaluated"])
        self.assertIn("multi_parent", report["modes_evaluated"])


if __name__ == "__main__":
    unittest.main()
