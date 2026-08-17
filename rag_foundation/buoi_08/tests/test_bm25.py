import os
import sys
import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Đảm bảo import được advanced_rag.py từ buoi_08
TEST_DIR = Path(__file__).resolve().parent
BUOI_08_DIR = TEST_DIR.parent
if str(BUOI_08_DIR) not in sys.path:
    sys.path.insert(0, str(BUOI_08_DIR))

import advanced_rag


class TestBM25Retrieval(unittest.TestCase):
    def setUp(self):
        self.sample_chunks = [
            {
                "chunk_id": "c1",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Điều 7. Cơ cấu lại thời hạn trả nợ cho khách hàng gặp khó khăn tài chính."
            },
            {
                "chunk_id": "c2",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 2,
                "page_end": 2,
                "text": "Khoản 2 Điều 7. Chi tiết điều kiện hoãn nợ và miễn lãi vay tín dụng."
            },
            {
                "chunk_id": "c3",
                "strategy": "hierarchical",
                "source": "doc2.pdf",
                "page_start": 5,
                "page_end": 5,
                "text": "Phân loại nhóm nợ và trích lập dự phòng rủi ro ngân hàng."
            }
        ]

    # Test 1: Tokenizer giữ dấu tiếng Việt
    def test_01_tokenizer_preserves_vietnamese_diacritics(self):
        text = "cơ cấu lại thời hạn trả nợ"
        tokens = advanced_rag.tokenize_vi_legal(text)
        expected = ["cơ", "cấu", "lại", "thời", "hạn", "trả", "nợ"]
        self.assertEqual(tokens, expected)

    # Test 2: Tokenizer giữ số Điều/Khoản
    def test_02_tokenizer_preserves_article_and_clause_numbers(self):
        text = "Điều 7, Khoản 2"
        tokens = advanced_rag.tokenize_vi_legal(text)
        expected = ["điều", "7", "khoản", "2"]
        self.assertEqual(tokens, expected)

    # Test 3: Corpus và query dùng cùng preprocessing
    def test_03_corpus_and_query_use_same_preprocessing(self):
        raw_text = "ĐIỀU 7, KHOẢN 2!"
        query_text = "Điều 7 khoản 2"
        tokens_corpus = advanced_rag.tokenize_vi_legal(raw_text)
        tokens_query = advanced_rag.tokenize_vi_legal(query_text)
        self.assertEqual(tokens_corpus, tokens_query)

    # Test 4: Exact legal term được xếp trên đoạn không chứa từ khóa
    def test_04_exact_legal_term_ranked_higher(self):
        res = advanced_rag.run_bm25_search(
            question="Cơ cấu lại thời hạn trả nợ",
            chunks=self.sample_chunks,
            candidate_k=3
        )
        self.assertEqual(res[0]["chunk_id"], "c1")
        self.assertGreater(res[0]["bm25_score"], res[2]["bm25_score"])

    # Test 5: candidate_k lớn hơn corpus vẫn chạy đúng
    def test_05_candidate_k_greater_than_corpus_size(self):
        res = advanced_rag.run_bm25_search(
            question="Điều 7",
            chunks=self.sample_chunks,
            candidate_k=100
        )
        self.assertEqual(len(res), len(self.sample_chunks))

    # Test 6: Empty question hoặc không token fail rõ
    def test_06_empty_or_tokenless_question_fails(self):
        with self.assertRaises(ValueError):
            advanced_rag.run_bm25_search("   ", self.sample_chunks)

        with self.assertRaises(ValueError):
            advanced_rag.run_bm25_search("!!! ??? ", self.sample_chunks)

        with self.assertRaises(TypeError):
            advanced_rag.run_bm25_search(123, self.sample_chunks)

    # Test 7: Tie-break deterministic bằng chunk_id
    def test_07_deterministic_tie_break(self):
        tied_chunks = [
            {"chunk_id": "chunk_z", "strategy": "h", "source": "s", "page_start": 1, "page_end": 1, "text": "Không khớp từ khóa"},
            {"chunk_id": "chunk_a", "strategy": "h", "source": "s", "page_start": 1, "page_end": 1, "text": "Không khớp từ khóa"},
            {"chunk_id": "chunk_m", "strategy": "h", "source": "s", "page_start": 1, "page_end": 1, "text": "Không khớp từ khóa"}
        ]
        res = advanced_rag.run_bm25_search("Ngân hàng", tied_chunks, candidate_k=3)
        # Các score đều bằng 0, tie-break xếp theo chunk_id tăng dần: chunk_a, chunk_m, chunk_z
        chunk_ids = [r["chunk_id"] for r in res]
        self.assertEqual(chunk_ids, ["chunk_a", "chunk_m", "chunk_z"])

    # Test 8: Đảm bảo không gọi Gemini/Chroma/reranker
    @patch("google.genai.Client")
    @patch("chromadb.PersistentClient")
    @patch("transformers.AutoModel")
    def test_08_offline_no_external_service_calls(self, mock_tf, mock_chroma, mock_genai):
        res = advanced_rag.run_bm25_search(
            question="Cơ cấu lại nợ",
            chunks=self.sample_chunks,
            candidate_k=2
        )
        self.assertEqual(len(res), 2)
        mock_genai.assert_not_called()
        mock_chroma.assert_not_called()
        mock_tf.assert_not_called()


if __name__ == "__main__":
    unittest.main()
