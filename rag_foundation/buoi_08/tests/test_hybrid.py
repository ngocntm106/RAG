"""Tests cho Bước 06: rrf_fusion, hybrid_retrieval.

Kiểm tra thuật toán RRF, tính nhất quán của metadata, tie-break deterministic,
và pipeline trace mà không tải reranker hay gọi generation.
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

BUOI08_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BUOI08_DIR))

import advanced_rag
from advanced_rag import rrf_fusion, hybrid_retrieval

# Mock candidates for testing RRF
BM25_DUMMY_RESULTS = [
    {
        "chunk_id": "chunk_1",
        "text": "Nội dung văn bản số một.",
        "source": "QD_123.pdf",
        "page_start": 1,
        "page_end": 1,
        "bm25_rank": 1,
        "bm25_score": 10.5,
    },
    {
        "chunk_id": "chunk_2",
        "text": "Nội dung văn bản số hai.",
        "source": "QD_123.pdf",
        "page_start": 2,
        "page_end": 2,
        "bm25_rank": 2,
        "bm25_score": 8.2,
    },
    {
        "chunk_id": "chunk_only_bm25",
        "text": "Chỉ có ở BM25.",
        "source": "QD_123.pdf",
        "page_start": 3,
        "page_end": 3,
        "bm25_rank": 3,
        "bm25_score": 5.0,
    }
]

SEMANTIC_DUMMY_RESULTS = [
    {
        "chunk_id": "chunk_2",
        "text": "Nội dung văn bản số hai.",
        "source": "QD_123.pdf",
        "page_start": 2,
        "page_end": 2,
        "semantic_rank": 1,
        "semantic_distance": 0.15,
    },
    {
        "chunk_id": "chunk_1",
        "text": "Nội dung văn bản số một.",
        "source": "QD_123.pdf",
        "page_start": 1,
        "page_end": 1,
        "semantic_rank": 2,
        "semantic_distance": 0.20,
    },
    {
        "chunk_id": "chunk_only_semantic",
        "text": "Chỉ có ở Semantic.",
        "source": "QD_123.pdf",
        "page_start": 4,
        "page_end": 4,
        "semantic_rank": 3,
        "semantic_distance": 0.35,
    }
]


class TestRRFFusion(unittest.TestCase):

    def test_rrf_formula_arithmetic(self):
        """1. Kiểm tra tính đúng đắn số học của công thức RRF."""
        # chunk_1: bm25_rank=1, semantic_rank=2
        # rrf_score = 1.0 / (60 + 1) + 1.0 / (60 + 2) = 1/61 + 1/62 = 0.0163934 + 0.016129 = 0.032522
        k = 60
        bm_w = 1.0
        sem_w = 1.0
        results = rrf_fusion(
            bm25_results=[BM25_DUMMY_RESULTS[0]], # chunk_1 (bm25_rank=1)
            semantic_results=[SEMANTIC_DUMMY_RESULTS[1]], # chunk_1 (semantic_rank=2)
            k=k,
            bm25_weight=bm_w,
            semantic_weight=sem_w,
        )
        self.assertEqual(len(results), 1)
        expected_score = bm_w / (k + 1) + sem_w / (k + 2)
        self.assertAlmostEqual(results[0]["rrf_score"], expected_score)

    def test_overlap_deduplication(self):
        """2. Kết quả trùng lặp (overlap) phải được gộp và không duplicate."""
        results = rrf_fusion(
            bm25_results=BM25_DUMMY_RESULTS,
            semantic_results=SEMANTIC_DUMMY_RESULTS,
            k=60,
        )
        # BM25 có 3, Semantic có 3. Tổng cộng có 4 unique chunks: chunk_1, chunk_2, chunk_only_bm25, chunk_only_semantic
        self.assertEqual(len(results), 4)
        chunk_ids = [r["chunk_id"] for r in results]
        self.assertEqual(len(set(chunk_ids)), 4)
        
        # Kiểm tra thứ tự và xếp hạng
        for i, r in enumerate(results):
            self.assertEqual(r["fused_rank"], i + 1)

    def test_keep_only_bm25(self):
        """3. Candidate chỉ có BM25 vẫn được giữ lại trong danh sách fusion."""
        results = rrf_fusion(
            bm25_results=BM25_DUMMY_RESULTS,
            semantic_results=SEMANTIC_DUMMY_RESULTS,
            k=60,
        )
        only_bm25 = [r for r in results if r["chunk_id"] == "chunk_only_bm25"]
        self.assertEqual(len(only_bm25), 1)
        self.assertIsNotNone(only_bm25[0]["bm25_rank"])
        self.assertIsNone(only_bm25[0]["semantic_rank"])
        self.assertEqual(only_bm25[0]["matched_by"], ["bm25"])

    def test_keep_only_semantic(self):
        """4. Candidate chỉ có semantic vẫn được giữ lại trong danh sách fusion."""
        results = rrf_fusion(
            bm25_results=BM25_DUMMY_RESULTS,
            semantic_results=SEMANTIC_DUMMY_RESULTS,
            k=60,
        )
        only_sem = [r for r in results if r["chunk_id"] == "chunk_only_semantic"]
        self.assertEqual(len(only_sem), 1)
        self.assertIsNotNone(only_sem[0]["semantic_rank"])
        self.assertIsNone(only_sem[0]["bm25_rank"])
        self.assertEqual(only_sem[0]["matched_by"], ["semantic"])

    def test_weight_zero_excludes_contribution(self):
        """5. Trọng số bằng 0 loại bỏ sự đóng góp của nhánh tương ứng."""
        # Nếu bm25_weight = 0, điểm rrf của chunk_only_bm25 phải bằng 0.0
        results = rrf_fusion(
            bm25_results=BM25_DUMMY_RESULTS,
            semantic_results=SEMANTIC_DUMMY_RESULTS,
            k=60,
            bm25_weight=0.0,
            semantic_weight=1.0,
        )
        only_bm25 = [r for r in results if r["chunk_id"] == "chunk_only_bm25"][0]
        self.assertEqual(only_bm25["rrf_score"], 0.0)
        
        # Nếu semantic_weight = 0, điểm rrf của chunk_only_semantic phải bằng 0.0
        results_sem_zero = rrf_fusion(
            bm25_results=BM25_DUMMY_RESULTS,
            semantic_results=SEMANTIC_DUMMY_RESULTS,
            k=60,
            bm25_weight=1.0,
            semantic_weight=0.0,
        )
        only_sem = [r for r in results_sem_zero if r["chunk_id"] == "chunk_only_semantic"][0]
        self.assertEqual(only_sem["rrf_score"], 0.0)

    def test_tie_break_deterministic(self):
        """6. Tie-break khi bằng điểm rrf_score phải nhất quán và theo thứ tự ưu tiên."""
        # Tạo hai chunk có cùng điểm RRF (cả hai chỉ có ở một nhánh có hạng giống nhau)
        # chunk_a: bm25_rank=1, semantic_rank=None
        # chunk_b: bm25_rank=1, semantic_rank=None
        # Do bằng score, tie-break:
        # 1. best_rank (cả hai đều = 1)
        # 2. semantic_rank (cả hai đều = inf)
        # 3. bm25_rank (cả hai đều = 1)
        # 4. chunk_id (chọn "chunk_a" trước "chunk_b" theo bảng chữ cái)
        bm25_a = [
            {
                "chunk_id": "chunk_b",
                "text": "B",
                "source": "doc.pdf",
                "page_start": 1,
                "page_end": 1,
                "bm25_rank": 1,
                "bm25_score": 10.0,
            },
            {
                "chunk_id": "chunk_a",
                "text": "A",
                "source": "doc.pdf",
                "page_start": 1,
                "page_end": 1,
                "bm25_rank": 1,
                "bm25_score": 10.0,
            }
        ]
        results = rrf_fusion(bm25_results=bm25_a, semantic_results=[], k=60)
        self.assertEqual(results[0]["chunk_id"], "chunk_a")
        self.assertEqual(results[1]["chunk_id"], "chunk_b")

        # So sánh tie-break theo nhánh:
        # chunk_c: bm25_rank=1 (best=1, sem=inf, bm=1)
        # chunk_d: semantic_rank=1 (best=1, sem=1, bm=inf)
        # Bằng score. Tie-break:
        # 1. best_rank: cả hai đều 1
        # 2. semantic_rank: chunk_d (1) tốt hơn chunk_c (inf) -> chunk_d phải đứng trước chunk_c!
        mixed = rrf_fusion(
            bm25_results=[{
                "chunk_id": "chunk_c",
                "text": "C",
                "source": "doc.pdf",
                "page_start": 1,
                "page_end": 1,
                "bm25_rank": 1,
                "bm25_score": 10.0,
            }],
            semantic_results=[{
                "chunk_id": "chunk_d",
                "text": "D",
                "source": "doc.pdf",
                "page_start": 1,
                "page_end": 1,
                "semantic_rank": 1,
                "semantic_distance": 0.1,
            }],
            k=60
        )
        self.assertEqual(mixed[0]["chunk_id"], "chunk_d")
        self.assertEqual(mixed[1]["chunk_id"], "chunk_c")

    def test_metadata_mismatch_raises_value_error(self):
        """7. Metadata sai lệch (text, source, page) giữa 2 nhánh cùng chunk_id phải báo lỗi."""
        # Sai lệch source
        mismatch_source = [
            {
                "chunk_id": "chunk_1",
                "text": "Nội dung văn bản số một.",
                "source": "QD_KHACO.pdf", # Khác source
                "page_start": 1,
                "page_end": 1,
                "semantic_rank": 2,
                "semantic_distance": 0.20,
            }
        ]
        with self.assertRaises(ValueError) as ctx:
            rrf_fusion(bm25_results=BM25_DUMMY_RESULTS, semantic_results=mismatch_source)
        self.assertIn("mismatch", str(ctx.exception).lower())

        # Sai lệch text
        mismatch_text = [
            {
                "chunk_id": "chunk_1",
                "text": "Nội dung văn bản bị thay đổi.", # Khác text
                "source": "QD_123.pdf",
                "page_start": 1,
                "page_end": 1,
                "semantic_rank": 2,
                "semantic_distance": 0.20,
            }
        ]
        with self.assertRaises(ValueError) as ctx:
            rrf_fusion(bm25_results=BM25_DUMMY_RESULTS, semantic_results=mismatch_text)
        self.assertIn("mismatch", str(ctx.exception).lower())


class TestHybridRetrievalPipeline(unittest.TestCase):

    @patch("advanced_rag.bm25_search")
    @patch("advanced_rag.get_semantic_candidates")
    def test_pipeline_trace_and_exactly_once(self, mock_get_semantic, mock_bm25_search):
        """8. Kiểm tra trace counts & 9. Hybrid gọi mỗi retriever đúng một lần."""
        # Thiết lập mock
        mock_bm25_search.return_value = BM25_DUMMY_RESULTS
        mock_get_semantic.return_value = SEMANTIC_DUMMY_RESULTS

        # Mock rag.load_chunks và load_config
        with patch.object(__import__("rag"), "load_config", return_value={
            "gemini_api_key": "FAKE",
            "gemini_embedding_model": "test-model",
            "gemini_embedding_dim": 128,
            "gemini_generation_model": "test-gen",
            "default_top_k": 5,
            "rag_max_distance": 0.45,
        }), patch.object(__import__("rag"), "load_chunks", return_value=([], {"valid_chunks": 10})):
            results, trace = hybrid_retrieval(
                question="Điều 7",
                strategy="fixed-size",
            )

        # 9. Kiểm tra gọi đúng một lần
        mock_bm25_search.assert_called_once()
        mock_get_semantic.assert_called_once()

        # 8. Kiểm tra trace counts
        self.assertEqual(trace["bm25_candidate_count"], 3)
        self.assertEqual(trace["semantic_candidate_count"], 3)
        self.assertEqual(trace["union_count"], 4)
        self.assertEqual(trace["overlap_count"], 2)
        self.assertEqual(trace["fused_count"], 4) # do top_n trong config mặc định là 5 (> 4)
        self.assertIn("bm25", trace["latency_ms"])
        self.assertIn("semantic", trace["latency_ms"])
        self.assertIn("fusion", trace["latency_ms"])

    def test_no_reranker_or_generation_loaded(self):
        """10. Đảm bảo hybrid retrieval không load reranker hoặc thực hiện sinh văn bản."""
        import inspect
        source = inspect.getsource(advanced_rag.hybrid_retrieval)
        
        # Không được chứa các từ khóa liên quan đến sinh văn bản hay reranking model
        self.assertNotIn("generate_content", source)
        self.assertNotIn("query_knowledge", source)
        self.assertNotIn("rerank_candidates", source)
        self.assertNotIn("CrossEncoder", source)


if __name__ == "__main__":
    unittest.main()
