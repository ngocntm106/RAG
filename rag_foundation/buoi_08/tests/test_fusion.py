import os
import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Đảm bảo import được advanced_rag.py từ buoi_08
TEST_DIR = Path(__file__).resolve().parent
BUOI_08_DIR = TEST_DIR.parent
if str(BUOI_08_DIR) not in sys.path:
    sys.path.insert(0, str(BUOI_08_DIR))

import advanced_rag
import rag


class TestRRFFusionAndHybridSearch(unittest.TestCase):
    def setUp(self):
        self.mock_bm25_ranks = [
            {"chunk_id": "c1", "text": "Nội dung 1", "source": "s1.pdf", "page_start": 1, "page_end": 1, "bm25_rank": 1, "bm25_score": 5.0},
            {"chunk_id": "c2", "text": "Nội dung 2", "source": "s1.pdf", "page_start": 2, "page_end": 2, "bm25_rank": 2, "bm25_score": 3.0},
        ]
        self.mock_semantic_ranks = [
            {"chunk_id": "c2", "text": "Nội dung 2", "source": "s1.pdf", "page_start": 2, "page_end": 2, "semantic_rank": 1, "semantic_distance": 0.1},
            {"chunk_id": "c3", "text": "Nội dung 3", "source": "s2.pdf", "page_start": 1, "page_end": 1, "semantic_rank": 2, "semantic_distance": 0.2},
        ]

    # Test 1: RRF formula đúng số học
    def test_01_rrf_formula_arithmetic(self):
        # k=60, w_bm25=1.0, w_sem=1.0
        # c2 có bm25_rank=2 và semantic_rank=1
        # rrf_score(c2) = 1.0/(60+2) + 1.0/(60+1) = 1/62 + 1/61 = 0.016129032... + 0.016393442... = 0.032522
        results, counts = advanced_rag.rrf_fuse(
            bm25_ranks=self.mock_bm25_ranks,
            semantic_ranks=self.mock_semantic_ranks,
            k=60, w_bm25=1.0, w_sem=1.0, rerank_candidates_k=10
        )
        c2_item = next(r for r in results if r["chunk_id"] == "c2")
        expected_score = round(1.0/62.0 + 1.0/61.0, 6)
        self.assertAlmostEqual(c2_item["rrf_score"], expected_score, places=5)
        self.assertEqual(c2_item["fused_rank"], 1)

    # Test 2: Overlap không duplicate
    def test_02_overlap_no_duplicates(self):
        results, counts = advanced_rag.rrf_fuse(
            bm25_ranks=self.mock_bm25_ranks,
            semantic_ranks=self.mock_semantic_ranks,
            k=60, rerank_candidates_k=10
        )
        cids = [r["chunk_id"] for r in results]
        self.assertEqual(len(cids), len(set(cids)))
        self.assertEqual(counts["overlap_count"], 1)
        self.assertEqual(counts["union_count"], 3)

    # Test 3: Candidate chỉ có BM25 vẫn được giữ
    def test_03_bm25_only_candidate_retained(self):
        results, _ = advanced_rag.rrf_fuse(
            bm25_ranks=self.mock_bm25_ranks,
            semantic_ranks=self.mock_semantic_ranks,
            k=60, rerank_candidates_k=10
        )
        c1_item = next(r for r in results if r["chunk_id"] == "c1")
        self.assertIsNone(c1_item["semantic_rank"])
        self.assertIsNone(c1_item["semantic_distance"])
        self.assertEqual(c1_item["matched_by"], ["bm25"])

    # Test 4: Candidate chỉ có Semantic vẫn được giữ
    def test_04_semantic_only_candidate_retained(self):
        results, _ = advanced_rag.rrf_fuse(
            bm25_ranks=self.mock_bm25_ranks,
            semantic_ranks=self.mock_semantic_ranks,
            k=60, rerank_candidates_k=10
        )
        c3_item = next(r for r in results if r["chunk_id"] == "c3")
        self.assertIsNone(c3_item["bm25_rank"])
        self.assertIsNone(c3_item["bm25_score"])
        self.assertEqual(c3_item["matched_by"], ["semantic"])

    # Test 5: Weight 0 loại đóng góp đúng nhánh
    def test_05_weight_zero_disables_branch(self):
        # w_bm25 = 0.0 -> chỉ tính semantic score
        results, _ = advanced_rag.rrf_fuse(
            bm25_ranks=self.mock_bm25_ranks,
            semantic_ranks=self.mock_semantic_ranks,
            k=60, w_bm25=0.0, w_sem=1.0, rerank_candidates_k=10
        )
        c1_item = next(r for r in results if r["chunk_id"] == "c1")
        self.assertEqual(c1_item["rrf_score"], 0.0)

    # Test 6: Tie-break deterministic
    def test_06_deterministic_tie_break(self):
        # 2 items có cùng rrf_score và ranks
        bm25_tied = [
            {"chunk_id": "z_chunk", "text": "txt", "source": "s", "page_start": 1, "page_end": 1, "bm25_rank": 1, "bm25_score": 5.0},
            {"chunk_id": "a_chunk", "text": "txt", "source": "s", "page_start": 1, "page_end": 1, "bm25_rank": 1, "bm25_score": 5.0},
        ]
        results, _ = advanced_rag.rrf_fuse(
            bm25_ranks=bm25_tied, semantic_ranks=[], k=60, rerank_candidates_k=10
        )
        # alpha tie-break -> a_chunk trước z_chunk
        self.assertEqual(results[0]["chunk_id"], "a_chunk")
        self.assertEqual(results[1]["chunk_id"], "z_chunk")

    # Test 7: Metadata mismatch giữa 2 nhánh bị ném lỗi
    def test_07_metadata_mismatch_raises_error(self):
        bm25_bad = [
            {"chunk_id": "c1", "text": "Text A", "source": "s.pdf", "page_start": 1, "page_end": 1, "bm25_rank": 1, "bm25_score": 5.0}
        ]
        sem_bad = [
            {"chunk_id": "c1", "text": "Text B KHÁC TEXT A", "source": "s.pdf", "page_start": 1, "page_end": 1, "semantic_rank": 1, "semantic_distance": 0.1}
        ]
        with self.assertRaises(ValueError) as ctx:
            advanced_rag.rrf_fuse(bm25_bad, sem_bad)
        self.assertIn("bất nhất", str(ctx.exception).lower())

    # Test 8: Trace counts khớp đúng với dữ liệu
    @patch("advanced_rag.run_bm25_search")
    @patch("advanced_rag.run_semantic_search")
    def test_08_pipeline_trace_counts(self, mock_sem_search, mock_bm25_search):
        mock_bm25_search.return_value = self.mock_bm25_ranks
        mock_sem_search.return_value = self.mock_semantic_ranks

        with patch("rag.load_chunks") as mock_load:
            mock_load.return_value = {"chunks": [{"chunk_id": "c1"}, {"chunk_id": "c2"}, {"chunk_id": "c3"}]}
            hyb_res = advanced_rag.run_hybrid_search("Test question")
            trace = hyb_res["pipeline_trace"]

            self.assertEqual(trace["bm25_candidate_count"], 2)
            self.assertEqual(trace["semantic_candidate_count"], 2)
            self.assertEqual(trace["union_count"], 3)
            self.assertEqual(trace["overlap_count"], 1)
            self.assertEqual(trace["fused_count"], 3)
            self.assertIn("latency_ms", trace)

    # Test 9: Hybrid gọi mỗi retriever đúng 1 lần
    @patch("advanced_rag.run_bm25_search")
    @patch("advanced_rag.run_semantic_search")
    def test_09_retrievers_called_once(self, mock_sem_search, mock_bm25_search):
        mock_bm25_search.return_value = self.mock_bm25_ranks
        mock_sem_search.return_value = self.mock_semantic_ranks

        with patch("rag.load_chunks") as mock_load:
            mock_load.return_value = {"chunks": []}
            advanced_rag.run_hybrid_search("Test single call")
            mock_bm25_search.assert_called_once()
            mock_sem_search.assert_called_once()

    # Test 10: Không load reranker model và không gọi generation
    @patch("transformers.AutoModelForSequenceClassification")
    @patch("google.genai.Client")
    def test_10_no_reranker_or_generation_called(self, mock_genai, mock_reranker):
        with patch("advanced_rag.run_bm25_search") as m_bm25, patch("advanced_rag.run_semantic_search") as m_sem, patch("rag.load_chunks"):
            m_bm25.return_value = self.mock_bm25_ranks
            m_sem.return_value = self.mock_semantic_ranks

            res = advanced_rag.run_hybrid_search("Test no rerank")
            self.assertTrue(len(res["candidates"]) > 0)
            mock_reranker.from_pretrained.assert_not_called()
            mock_genai.return_value.models.generate_content.assert_not_called()


if __name__ == "__main__":
    unittest.main()
