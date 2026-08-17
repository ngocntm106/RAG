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


class TestAdvancedRAGPipeline(unittest.TestCase):
    def setUp(self):
        self.mock_config = {
            "GEMINI_API_KEY": "fake_test_key",
            "API_KEY_PRESENT": True,
            "GEMINI_EMBEDDING_MODEL": "gemini-embedding-test",
            "GEMINI_EMBEDDING_DIM": 128,
            "GEMINI_GENERATION_MODEL": "gemini-3.5-flash-lite",
            "RAG_MAX_DISTANCE": 0.45,
            "BM25_CANDIDATES": 10,
            "SEMANTIC_CANDIDATES": 10,
            "RRF_K": 60,
            "RRF_BM25_WEIGHT": 1.0,
            "RRF_SEMANTIC_WEIGHT": 1.0,
            "RERANK_CANDIDATES": 10,
            "FINAL_TOP_K": 3,
            "RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",
            "RERANKER_MAX_LENGTH": 512,
            "RERANK_BATCH_SIZE": 2,
            "RERANK_MIN_SCORE": 0.50,
            "RERANK_DEVICE": "auto"
        }
        self.mock_chunks = [
            {"chunk_id": "c1", "strategy": "hierarchical", "source": "s1.pdf", "page_start": 1, "page_end": 1, "text": "Điều 7. Quy định cơ cấu nợ và hoàn trả."},
            {"chunk_id": "c2", "strategy": "hierarchical", "source": "s1.pdf", "page_start": 2, "page_end": 2, "text": "Khoản 2 Điều 7. Miễn giảm lãi suất ngân hàng."},
            {"chunk_id": "c3", "strategy": "hierarchical", "source": "s2.pdf", "page_start": 5, "page_end": 5, "text": "Phân loại rủi ro tín dụng theo quy định."}
        ]

    # Test 1: Gating theo đúng mode rules (accepted đúng tiêu chuẩn)
    def test_01_gating_by_mode(self):
        cand = {
            "chunk_id": "c1", "text": "txt", "source": "s.pdf", "page_start": 1, "page_end": 1,
            "semantic_distance": 0.20, "rerank_score": 0.85
        }
        ev_sem = advanced_rag.format_evidence_schema(cand, accepted=True, mode="semantic")
        self.assertTrue(ev_sem["accepted"])

        ev_rr = advanced_rag.format_evidence_schema(cand, accepted=True, mode="hybrid_rerank")
        self.assertTrue(ev_rr["accepted"])

    # Test 2: Trace counts & timing có đầy đủ keys
    @patch("rag.load_chunks")
    @patch("advanced_rag.run_semantic_search")
    @patch("advanced_rag.generate_answer")
    def test_02_pipeline_trace_schema(self, mock_gen_ans, mock_sem_search, mock_load):
        mock_load.return_value = {"chunks": self.mock_chunks}
        mock_sem_search.return_value = [
            {"chunk_id": "c1", "text": "Điều 7. Quy định cơ cấu nợ và hoàn trả.", "source": "s1.pdf", "page_start": 1, "page_end": 1, "semantic_rank": 1, "semantic_distance": 0.1}
        ]
        mock_gen_ans.return_value = {
            "answer": "Trả lời mẫu [E1]",
            "citations": [{"citation_label": "[E1]", "chunk_id": "c1", "source": "s1.pdf", "page_start": 1, "page_end": 1}],
            "warnings": [], "status": "answered"
        }

        def mock_reranker_fn(q, texts):
            return [2.0] * len(texts)

        res = advanced_rag.run_advanced_query(
            question="Điều 7", mode="hybrid_rerank", config=self.mock_config, reranker_fn=mock_reranker_fn
        )
        trace = res["trace"]
        self.assertIn("bm25_candidates", trace)
        self.assertIn("semantic_candidates", trace)
        self.assertIn("overlap", trace)
        self.assertIn("union", trace)
        self.assertIn("reranked", trace)
        self.assertIn("accepted", trace)
        self.assertIn("generation_called", trace)
        self.assertIn("latency_ms", trace)
        self.assertIn("total", trace["latency_ms"])

    # Test 3: Mapping trích dẫn chuyển E1 sang metadata thật
    @patch("rag.load_chunks")
    @patch("advanced_rag.run_semantic_search")
    @patch("advanced_rag.generate_answer")
    def test_03_citation_mapping_to_real_metadata(self, mock_gen_ans, mock_sem_search, mock_load):
        mock_load.return_value = {"chunks": self.mock_chunks}
        mock_sem_search.return_value = [
            {"chunk_id": "c1", "text": "Điều 7. Quy định cơ cấu nợ và hoàn trả.", "source": "s1.pdf", "page_start": 1, "page_end": 1, "semantic_rank": 1, "semantic_distance": 0.1}
        ]
        mock_gen_ans.return_value = {
            "answer": "Căn cứ theo [E1], cơ cấu nợ áp dụng.",
            "citations": [{"citation_label": "[E1]", "chunk_id": "c1", "source": "s1.pdf", "page_start": 1, "page_end": 1}],
            "warnings": [], "status": "answered"
        }

        def mock_rerank_fn(q, texts):
            return [3.0] * len(texts)

        res = advanced_rag.run_advanced_query(
            question="Điều 7", mode="hybrid_rerank", config=self.mock_config, reranker_fn=mock_rerank_fn
        )
        self.assertEqual(len(res["citations"]), 1)
        self.assertEqual(res["citations"][0]["chunk_id"], "c1")
        self.assertEqual(res["citations"][0]["source"], "s1.pdf")

    # Test 4: Generation chỉ gọi tối đa 1 lần
    @patch("rag.load_chunks")
    @patch("advanced_rag.run_semantic_search")
    @patch("advanced_rag.generate_answer")
    def test_04_generation_called_at_most_once(self, mock_gen_ans, mock_sem_search, mock_load):
        mock_load.return_value = {"chunks": self.mock_chunks}
        mock_sem_search.return_value = [
            {"chunk_id": "c1", "text": "Điều 7. Quy định cơ cấu nợ và hoàn trả.", "source": "s1.pdf", "page_start": 1, "page_end": 1, "semantic_rank": 1, "semantic_distance": 0.1}
        ]
        mock_gen_ans.return_value = {"answer": "Ans", "citations": [], "warnings": [], "status": "answered"}

        res = advanced_rag.run_advanced_query(
            question="Điều 7", mode="hybrid_rerank", config=self.mock_config,
            reranker_fn=lambda q, txts: [2.0]*len(txts)
        )
        mock_gen_ans.assert_called_once()

    # Test 5: Compare command TUYỆT ĐỐI không gọi LLM generation
    @patch("rag.load_chunks")
    @patch("advanced_rag.run_semantic_search")
    @patch("advanced_rag.generate_answer")
    def test_05_compare_never_calls_generation(self, mock_gen_ans, mock_sem_search, mock_load):
        mock_load.return_value = {"chunks": self.mock_chunks}
        mock_sem_search.return_value = [
            {"chunk_id": "c1", "text": "Điều 7. Quy định cơ cấu nợ và hoàn trả.", "source": "s1.pdf", "page_start": 1, "page_end": 1, "semantic_rank": 1, "semantic_distance": 0.1}
        ]

        cmp_out = advanced_rag.run_mode_comparison(
            question="Điều 7", config=self.mock_config, reranker_fn=lambda q, txts: [1.0]*len(txts)
        )
        mock_gen_ans.assert_not_called()
        self.assertIn("comparison_table", cmp_out)
        self.assertEqual(len(cmp_out["mode_results"]), 4)

    # Test 6: Status reranker_unavailable được kích hoạt đúng khi reranker lỗi
    @patch("rag.load_chunks")
    @patch("advanced_rag.run_semantic_search")
    def test_06_reranker_unavailable_status(self, mock_sem_search, mock_load):
        mock_load.return_value = {"chunks": self.mock_chunks}
        mock_sem_search.return_value = [
            {"chunk_id": "c1", "text": "Điều 7. Quy định cơ cấu nợ và hoàn trả.", "source": "s1.pdf", "page_start": 1, "page_end": 1, "semantic_rank": 1, "semantic_distance": 0.1}
        ]

        def bad_reranker(q, txts):
            raise RuntimeError("reranker_unavailable: Connection failed")

        res = advanced_rag.run_advanced_query(
            question="Điều 7", mode="hybrid_rerank", config=self.mock_config, reranker_fn=bad_reranker
        )
        self.assertEqual(res["status"], "reranker_unavailable")
        self.assertTrue(any("reranker_unavailable" in w.lower() for w in res["warnings"]))

    # Test 7: Mọi status trả về đúng schema đầy đủ
    @patch("rag.load_chunks")
    @patch("advanced_rag.run_semantic_search")
    def test_07_full_response_schema(self, mock_sem_search, mock_load):
        mock_load.return_value = {"chunks": self.mock_chunks}
        mock_sem_search.return_value = []

        res = advanced_rag.run_advanced_query(
            question="Câu hỏi không khớp", mode="hybrid", config=self.mock_config
        )
        expected_keys = {"status", "mode", "question", "answer", "evidence", "citations", "warnings", "trace"}
        self.assertEqual(set(res.keys()), expected_keys)


if __name__ == "__main__":
    unittest.main()
