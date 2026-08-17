import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Đảm bảo import được advanced_rag.py từ buoi_08
TEST_DIR = Path(__file__).resolve().parent
BUOI_08_DIR = TEST_DIR.parent
if str(BUOI_08_DIR) not in sys.path:
    sys.path.insert(0, str(BUOI_08_DIR))

import advanced_rag
import rag


class TestSemanticCandidateSearch(unittest.TestCase):
    def setUp(self):
        self.mock_config = {
            "GEMINI_API_KEY": "fake_test_api_key_12345",
            "API_KEY_PRESENT": True,
            "GEMINI_EMBEDDING_MODEL": "gemini-embedding-test",
            "GEMINI_EMBEDDING_DIM": 128,
            "GEMINI_GENERATION_MODEL": "gemini-3.5-flash-lite",
            "RAG_MAX_DISTANCE": 0.45,
            "BM25_CANDIDATES": 20,
            "SEMANTIC_CANDIDATES": 20,
            "RRF_K": 60,
            "RRF_BM25_WEIGHT": 1.0,
            "RRF_SEMANTIC_WEIGHT": 1.0,
            "RERANK_CANDIDATES": 20,
            "FINAL_TOP_K": 5,
            "RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",
            "RERANKER_MAX_LENGTH": 512,
            "RERANK_BATCH_SIZE": 4,
            "RERANK_MIN_SCORE": 0.50,
            "RERANK_DEVICE": "auto"
        }

    def _get_mock_genai_client(self):
        client = MagicMock()
        def mock_embed(model, contents, config):
            resp = MagicMock()
            resp.embeddings = [MagicMock(values=[0.1] * 128)]
            return resp
        client.models.embed_content.side_effect = mock_embed
        return client

    # Test 1: Semantic top-k / count / rank ordering đúng (distance thấp hơn xếp trước)
    def test_01_semantic_topk_and_rank_ordering(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_storage:
            fixture_dir = BUOI_08_DIR / "tests" / "fixtures"
            client_mock = self._get_mock_genai_client()

            # Index 8 chunks fixture
            rag.run_index(
                input_dir=fixture_dir, strategy="hierarchical", reset=True,
                storage_dir=Path(tmp_storage), genai_client=client_mock, config=self.mock_config
            )

            res = advanced_rag.run_semantic_search(
                question="Cơ cấu nợ", candidate_k=3, strategy="hierarchical",
                storage_dir=Path(tmp_storage), genai_client=client_mock, config=self.mock_config
            )

            self.assertEqual(len(res), 3)
            self.assertEqual(res[0]["semantic_rank"], 1)
            self.assertEqual(res[1]["semantic_rank"], 2)
            self.assertEqual(res[2]["semantic_rank"], 3)
            # Distance xếp tăng dần
            self.assertLessEqual(res[0]["semantic_distance"], res[1]["semantic_distance"])
            self.assertLessEqual(res[1]["semantic_distance"], res[2]["semantic_distance"])

    # Test 2: Metadata đầy đủ
    def test_02_metadata_completeness(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_storage:
            fixture_dir = BUOI_08_DIR / "tests" / "fixtures"
            client_mock = self._get_mock_genai_client()

            rag.run_index(
                input_dir=fixture_dir, strategy="hierarchical", reset=True,
                storage_dir=Path(tmp_storage), genai_client=client_mock, config=self.mock_config
            )

            res = advanced_rag.run_semantic_search(
                question="Cơ cấu nợ", candidate_k=1, strategy="hierarchical",
                storage_dir=Path(tmp_storage), genai_client=client_mock, config=self.mock_config
            )

            item = res[0]
            self.assertIn("chunk_id", item)
            self.assertIn("text", item)
            self.assertIn("source", item)
            self.assertIn("page_start", item)
            self.assertIn("page_end", item)
            self.assertIn("semantic_rank", item)
            self.assertIn("semantic_distance", item)

    # Test 3: Collection identity / metadata mismatch bị chặn
    def test_03_collection_mismatch_blocked(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_storage:
            chroma_cli = rag.get_chroma_client(Path(tmp_storage))
            coll_name = rag.get_collection_name("hierarchical", self.mock_config)
            bad_meta = {
                "strategy": "hierarchical",
                "embedding_model": "old-model",
                "embedding_dim": 999,
                "distance_metric": "cosine"
            }
            coll = chroma_cli.create_collection(name=coll_name, metadata=bad_meta, embedding_function=None)
            coll.add(
                ids=["c1"], documents=["doc"], embeddings=[[0.1]*128],
                metadatas=[{"source": "s", "strategy": "hierarchical", "page_start": 1, "page_end": 1, "chunk_id": "c1", "embedding_model": "old-model", "embedding_dim": 999}]
            )

            with self.assertRaises(ValueError) as ctx:
                advanced_rag.run_semantic_search(
                    question="test", strategy="hierarchical", storage_dir=Path(tmp_storage),
                    genai_client=self._get_mock_genai_client(), config=self.mock_config
                )
            self.assertIn("mismatch", str(ctx.exception).lower())

    # Test 4: Status command không tạo collection khi chưa tồn tại
    def test_04_status_does_not_create_collection(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_storage:
            st_res = advanced_rag.run_advanced_status(
                strategy="hierarchical", config=self.mock_config, storage_dir=Path(tmp_storage)
            )
            self.assertFalse(st_res["collection_exists"])
            self.assertEqual(st_res["collection_count"], 0)

            # Đảm bảo danh sách collection vẫn rỗng
            chroma_cli = rag.get_chroma_client(Path(tmp_storage))
            self.assertEqual(len(chroma_cli.list_collections()), 0)

    # Test 5: Thiếu API key ngắt an toàn và không dùng vector giả
    def test_05_missing_api_key_fails(self):
        no_key_config = dict(self.mock_config)
        no_key_config["API_KEY_PRESENT"] = False
        no_key_config["GEMINI_API_KEY"] = ""

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_storage:
            with self.assertRaises(ValueError) as ctx:
                advanced_rag.run_prepare_semantic(
                    strategy="hierarchical", storage_dir=Path(tmp_storage), config=no_key_config
                )
            self.assertIn("thiếu gemini_api_key", str(ctx.exception).lower())

    # Test 6: Không gọi LLM generation trong semantic candidate stage
    def test_06_no_generation_called(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_storage:
            fixture_dir = BUOI_08_DIR / "tests" / "fixtures"
            client_mock = self._get_mock_genai_client()

            rag.run_index(
                input_dir=fixture_dir, strategy="hierarchical", reset=True,
                storage_dir=Path(tmp_storage), genai_client=client_mock, config=self.mock_config
            )

            res = advanced_rag.run_semantic_search(
                question="Cơ cấu lại nợ", candidate_k=5, strategy="hierarchical",
                storage_dir=Path(tmp_storage), genai_client=client_mock, config=self.mock_config
            )

            self.assertTrue(len(res) > 0)
            # generate_content của Gemini LLM tuyệt đối không được gọi
            client_mock.models.generate_content.assert_not_called()


if __name__ == "__main__":
    unittest.main()
