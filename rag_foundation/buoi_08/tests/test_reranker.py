import os
import sys
import math
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Đảm bảo import được advanced_rag.py từ buoi_08
TEST_DIR = Path(__file__).resolve().parent
BUOI_08_DIR = TEST_DIR.parent
if str(BUOI_08_DIR) not in sys.path:
    sys.path.insert(0, str(BUOI_08_DIR))

import advanced_rag


class TestCrossEncoderReranker(unittest.TestCase):
    def setUp(self):
        self.mock_config = {
            "RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",
            "RERANK_CANDIDATES": 5,
            "FINAL_TOP_K": 3,
            "RERANKER_MAX_LENGTH": 512,
            "RERANK_BATCH_SIZE": 2,
            "RERANK_MIN_SCORE": 0.50,
            "RERANK_DEVICE": "auto"
        }
        self.mock_fused_candidates = [
            {"chunk_id": "c1", "text": "Đoạn 1 về nợ", "source": "s.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1, "rrf_score": 0.03},
            {"chunk_id": "c2", "text": "Đoạn 2 về lãi", "source": "s.pdf", "page_start": 2, "page_end": 2, "fused_rank": 2, "rrf_score": 0.025},
            {"chunk_id": "c3", "text": "Đoạn 3 về quy trình", "source": "s.pdf", "page_start": 3, "page_end": 3, "fused_rank": 3, "rrf_score": 0.02},
            {"chunk_id": "c4", "text": "Đoạn 4 về phạt", "source": "s.pdf", "page_start": 4, "page_end": 4, "fused_rank": 4, "rrf_score": 0.015},
            {"chunk_id": "c5", "text": "Đoạn 5 về khác", "source": "s.pdf", "page_start": 5, "page_end": 5, "fused_rank": 5, "rrf_score": 0.01},
        ]

    # Test 1: Lazy loading (chưa nạp model khi import hoặc lệnh không dùng reranker)
    def test_01_lazy_loading(self):
        self.assertIsNone(advanced_rag._RERANKER_CACHE["model"])
        self.assertIsNone(advanced_rag._RERANKER_CACHE["tokenizer"])

    # Test 2: Tạo đúng 1 pair (question, candidate_text) cho mỗi candidate
    def test_02_one_pair_per_candidate(self):
        pairs_received = []

        def mock_scorer(question, texts):
            for t in texts:
                pairs_received.append((question, t))
            return [1.0] * len(texts)

        res, _ = advanced_rag.run_rerank(
            question="Cơ cấu nợ",
            candidates=self.mock_fused_candidates,
            config=self.mock_config,
            reranker_fn=mock_scorer
        )
        self.assertEqual(len(pairs_received), 5)
        self.assertEqual(pairs_received[0], ("Cơ cấu nợ", "Đoạn 1 về nợ"))

    # Test 3: Scoring giữ nguyên số lượng candidate trong batch
    def test_03_batch_scoring_preserves_count(self):
        def mock_scorer(q, texts):
            return [float(i) for i in range(len(texts))]

        res, _ = advanced_rag.run_rerank(
            question="Test",
            candidates=self.mock_fused_candidates,
            config=self.mock_config,
            reranker_fn=mock_scorer
        )
        # Config FINAL_TOP_K = 3
        self.assertEqual(len(res), 3)

    # Test 4: Công thức Sigmoid chính xác
    def test_04_sigmoid_score_accuracy(self):
        def mock_scorer(q, texts):
            # Trả về 0.0 -> sigmoid(0.0) = 0.5
            return [0.0] * len(texts)

        res, _ = advanced_rag.run_rerank(
            question="Test",
            candidates=self.mock_fused_candidates[:1],
            config=self.mock_config,
            reranker_fn=mock_scorer
        )
        self.assertEqual(res[0]["rerank_raw_score"], 0.0)
        self.assertEqual(res[0]["rerank_score"], 0.5)

    # Test 5: Sắp xếp và tie-break đúng
    def test_05_sort_and_tie_break(self):
        # Đảo ngược thứ tự score: c3 có score cao nhất (2.0)
        def mock_scorer(q, texts):
            # texts: c1, c2, c3, c4, c5
            return [-1.0, 0.0, 2.0, 1.0, 0.5]

        res, _ = advanced_rag.run_rerank(
            question="Test",
            candidates=self.mock_fused_candidates,
            config=self.mock_config,
            reranker_fn=mock_scorer
        )
        # Rerank rank 1 phải là c3 (score raw 2.0)
        self.assertEqual(res[0]["chunk_id"], "c3")
        self.assertEqual(res[0]["rerank_rank"], 1)

    # Test 6: rank_change tính đúng (fused_rank - rerank_rank)
    def test_06_rank_change_arithmetic(self):
        # c3 từ fused_rank=3 nhô lên rerank_rank=1 -> rank_change = 3 - 1 = +2
        def mock_scorer(q, texts):
            return [0.0, 0.0, 5.0, 0.0, 0.0]

        res, _ = advanced_rag.run_rerank(
            question="Test",
            candidates=self.mock_fused_candidates,
            config=self.mock_config,
            reranker_fn=mock_scorer
        )
        c3_res = res[0]
        self.assertEqual(c3_res["chunk_id"], "c3")
        self.assertEqual(c3_res["fused_rank"], 3)
        self.assertEqual(c3_res["rerank_rank"], 1)
        self.assertEqual(c3_res["rank_change"], 2)

    # Test 7: Chỉ rerank tối đa RERANK_CANDIDATES
    def test_07_rerank_candidates_limit(self):
        passed_texts = []
        def mock_scorer(q, texts):
            passed_texts.extend(texts)
            return [1.0] * len(texts)

        custom_cfg = dict(self.mock_config)
        custom_cfg["RERANK_CANDIDATES"] = 2  # Chỉ rerank 2 candidates đầu
        advanced_rag.run_rerank(
            question="Test",
            candidates=self.mock_fused_candidates,
            config=custom_cfg,
            reranker_fn=mock_scorer
        )
        self.assertEqual(len(passed_texts), 2)

    # Test 8: Chỉ trả về đúng FINAL_TOP_K candidates
    def test_08_returns_exactly_final_top_k(self):
        def mock_scorer(q, texts):
            return [float(i) for i in range(len(texts))]

        res, _ = advanced_rag.run_rerank(
            question="Test",
            candidates=self.mock_fused_candidates,
            config=self.mock_config,
            reranker_fn=mock_scorer
        )
        self.assertEqual(len(res), self.mock_config["FINAL_TOP_K"])

    # Test 9: Khi nạp model bị lỗi thì ném ngoại lệ reranker_unavailable
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_09_model_download_failure_raises_error(self, mock_tok):
        mock_tok.side_effect = RuntimeError("Connection timeout")
        with self.assertRaises(RuntimeError) as ctx:
            advanced_rag.get_reranker_model_and_tokenizer(self.mock_config)
        self.assertIn("reranker_unavailable", str(ctx.exception))

    # Test 10: Quá trình test 100% offline, không gọi HuggingFace/Internet
    @patch("transformers.AutoModelForSequenceClassification.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_10_offline_execution(self, mock_tok, mock_model):
        def mock_scorer(q, texts):
            return [1.0] * len(texts)

        res, latency = advanced_rag.run_rerank(
            question="Test offline",
            candidates=self.mock_fused_candidates,
            config=self.mock_config,
            reranker_fn=mock_scorer
        )
        self.assertTrue(len(res) > 0)
        self.assertGreaterEqual(latency, 0.0)
        # Mock transformers không bao giờ được gọi khi dùng injected reranker_fn
        mock_tok.assert_not_called()
        mock_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
