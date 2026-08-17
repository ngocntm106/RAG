import sys
import math
import unittest
from pathlib import Path

# Đảm bảo import được evaluate.py từ buoi_08
TEST_DIR = Path(__file__).resolve().parent
BUOI_08_DIR = TEST_DIR.parent
if str(BUOI_08_DIR) not in sys.path:
    sys.path.insert(0, str(BUOI_08_DIR))

import evaluate


class TestEvaluatorMetrics(unittest.TestCase):
    # Test 1: Recall@K ví dụ tính tay
    def test_01_recall_at_k_hand_calculated(self):
        retrieved = ["c1", "c2", "c3", "c4"]
        relevant = ["c2", "c5"]
        # In top 3: ["c1", "c2", "c3"], hit c2 -> 1/2 = 0.5
        rec_3 = evaluate.calculate_recall_at_k(retrieved, relevant, k=3)
        self.assertEqual(rec_3, 0.5)

    # Test 2: MRR@K ví dụ tính tay
    def test_02_mrr_at_k_hand_calculated(self):
        retrieved = ["c1", "c2", "c3"]
        relevant = ["c2"]
        # Match ở rank 2 -> MRR = 1/2 = 0.5
        mrr_3 = evaluate.calculate_mrr_at_k(retrieved, relevant, k=3)
        self.assertEqual(mrr_3, 0.5)

        # Match ở rank 1 -> MRR = 1.0
        mrr_1 = evaluate.calculate_mrr_at_k(["c2", "c1"], relevant, k=3)
        self.assertEqual(mrr_1, 1.0)

        # Không match -> MRR = 0.0
        mrr_none = evaluate.calculate_mrr_at_k(["c1", "c3"], relevant, k=3)
        self.assertEqual(mrr_none, 0.0)

    # Test 3: nDCG@K ví dụ tính tay
    def test_03_ndcg_at_k_hand_calculated(self):
        retrieved = ["c1", "c2", "c3"]
        relevant = ["c2"]
        # rel = [0, 1, 0]
        # DCG@3 = 0/log2(2) + 1/log2(3) + 0/log2(4) = 1 / 1.5849625 = 0.6309297
        # IDCG@3 = 1/log2(2) = 1.0
        # nDCG@3 = 0.6309297 / 1.0 = 0.6309
        ndcg_3 = evaluate.calculate_ndcg_at_k(retrieved, relevant, k=3)
        expected_ndcg = round(1.0 / math.log2(3), 4)
        self.assertEqual(ndcg_3, expected_ndcg)

    # Test 4: Calculation summary (mean & p50 median latency)
    def test_04_metrics_summary_mean_and_p50(self):
        eval_items = [
            {"recall": 1.0, "mrr": 1.0, "ndcg": 1.0, "latency_ms": 10.0},
            {"recall": 0.5, "mrr": 0.5, "ndcg": 0.5, "latency_ms": 20.0},
            {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0, "latency_ms": 100.0},
        ]
        summary = evaluate.calculate_metrics_summary(eval_items, k=3)
        self.assertEqual(summary["recall_mean"], round((1.0 + 0.5 + 0.0)/3, 4))
        self.assertEqual(summary["mrr_mean"], round((1.0 + 0.5 + 0.0)/3, 4))
        self.assertEqual(summary["latency_mean_ms"], round((10.0 + 20.0 + 100.0)/3, 2))
        self.assertEqual(summary["latency_p50_ms"], 20.0)


if __name__ == "__main__":
    unittest.main()
