import os
import sys
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Đảm bảo import được rag.py dù CWD đứng ở đâu
TEST_DIR = Path(__file__).resolve().parent
BUOI_07_DIR = TEST_DIR.parent
if str(BUOI_07_DIR) not in sys.path:
    sys.path.insert(0, str(BUOI_07_DIR))

import rag


class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        self.fixture_dir = (BUOI_07_DIR / "tests" / "fixtures").resolve()
        # Fake config với vector dim 128 (nhỏ nhất hợp lệ)
        self.mock_config = {
            "GEMINI_API_KEY": "fake_test_api_key_12345",
            "API_KEY_PRESENT": True,
            "GEMINI_EMBEDDING_MODEL": "gemini-embedding-test",
            "GEMINI_EMBEDDING_DIM": 128,
            "GEMINI_GENERATION_MODEL": "gemini-3.5-flash-lite",
            "DEFAULT_TOP_K": 5,
            "RAG_MAX_DISTANCE": 0.45
        }

    def _get_mock_genai_client(self, gen_answer: str = "Trả lời [E1]."):
        client = MagicMock()

        def mock_embed(model, contents, config):
            resp = MagicMock()
            # Trả vector deterministic dim 128
            resp.embedding.values = [0.1] * 128
            return resp

        client.models.embed_content.side_effect = mock_embed

        gen_resp = MagicMock()
        gen_resp.text = gen_answer
        client.models.generate_content.return_value = gen_resp
        return client

    # ----------------------------------------------------------------------
    # Group 1: Loader & Validation (Cases 1-9, 38)
    # ----------------------------------------------------------------------
    def test_01_loader_reads_json_list(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            fpath = Path(tmp) / "data.json"
            fpath.write_text(json.dumps([{
                "chunk_id": "c1", "strategy": "hierarchical", "source": "s.pdf",
                "page_start": 1, "page_end": 1, "text": "text1"
            }]), encoding="utf-8")
            res = rag.load_chunks(Path(tmp), strategy="hierarchical")
            self.assertEqual(len(res["chunks"]), 1)
            self.assertEqual(res["chunks"][0]["chunk_id"], "c1")

    def test_02_loader_reads_object_with_chunks_field(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            fpath = Path(tmp) / "data.json"
            fpath.write_text(json.dumps({"chunks": [{
                "chunk_id": "c1", "strategy": "hierarchical", "source": "s.pdf",
                "page_start": 1, "page_end": 1, "text": "text1"
            }]}), encoding="utf-8")
            res = rag.load_chunks(Path(tmp), strategy="hierarchical")
            self.assertEqual(len(res["chunks"]), 1)

    def test_03_loader_filters_strategy_only(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            fpath = Path(tmp) / "data.json"
            fpath.write_text(json.dumps([
                {"chunk_id": "c1", "strategy": "hierarchical", "source": "s.pdf", "page_start": 1, "page_end": 1, "text": "t1"},
                {"chunk_id": "c2", "strategy": "semantic", "source": "s.pdf", "page_start": 1, "page_end": 1, "text": "t2"}
            ]), encoding="utf-8")
            res = rag.load_chunks(Path(tmp), strategy="hierarchical")
            self.assertEqual(len(res["chunks"]), 1)
            self.assertEqual(res["chunks"][0]["strategy"], "hierarchical")

    def test_04_missing_required_field_fails(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            fpath = Path(tmp) / "data.json"
            # thiếu text
            fpath.write_text(json.dumps([{
                "chunk_id": "c1", "strategy": "hierarchical", "source": "s.pdf", "page_start": 1, "page_end": 1
            }]), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                rag.load_chunks(Path(tmp), strategy="hierarchical")
            self.assertIn("bắt buộc", str(ctx.exception).lower())

    def test_05_field_wrong_type_fails(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            fpath = Path(tmp) / "data.json"
            # chunk_id là số nguyên
            fpath.write_text(json.dumps([{
                "chunk_id": 123, "strategy": "hierarchical", "source": "s.pdf", "page_start": 1, "page_end": 1, "text": "t"
            }]), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                rag.load_chunks(Path(tmp), strategy="hierarchical")
            self.assertIn("string", str(ctx.exception).lower())

    def test_06_boolean_not_accepted_as_page_number(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            fpath = Path(tmp) / "data.json"
            # page_start là True (bool)
            fpath.write_text(json.dumps([{
                "chunk_id": "c1", "strategy": "hierarchical", "source": "s.pdf", "page_start": True, "page_end": 1, "text": "t"
            }]), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                rag.load_chunks(Path(tmp), strategy="hierarchical")
            self.assertIn("integer", str(ctx.exception).lower())

    def test_07_page_start_greater_than_page_end_fails(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            fpath = Path(tmp) / "data.json"
            fpath.write_text(json.dumps([{
                "chunk_id": "c1", "strategy": "hierarchical", "source": "s.pdf", "page_start": 5, "page_end": 2, "text": "t"
            }]), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                rag.load_chunks(Path(tmp), strategy="hierarchical")
            self.assertIn("khoảng trang", str(ctx.exception).lower())

    def test_08_empty_text_skipped_and_counted_in_stats(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            fpath = Path(tmp) / "data.json"
            fpath.write_text(json.dumps([
                {"chunk_id": "c1", "strategy": "hierarchical", "source": "s.pdf", "page_start": 1, "page_end": 1, "text": "   "},
                {"chunk_id": "c2", "strategy": "hierarchical", "source": "s.pdf", "page_start": 1, "page_end": 1, "text": "valid text"}
            ]), encoding="utf-8")
            res = rag.load_chunks(Path(tmp), strategy="hierarchical")
            self.assertEqual(len(res["chunks"]), 1)
            self.assertEqual(res["stats"]["empty_text_skipped"], 1)

    def test_09_duplicate_chunk_id_fails(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            fpath = Path(tmp) / "data.json"
            fpath.write_text(json.dumps([
                {"chunk_id": "c1", "strategy": "hierarchical", "source": "s.pdf", "page_start": 1, "page_end": 1, "text": "t1"},
                {"chunk_id": "c1", "strategy": "hierarchical", "source": "s.pdf", "page_start": 2, "page_end": 2, "text": "t2"}
            ]), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                rag.load_chunks(Path(tmp), strategy="hierarchical")
            self.assertIn("trùng lặp", str(ctx.exception).lower())

    def test_38_loader_blocks_non_object_record(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            fpath = Path(tmp) / "data.json"
            fpath.write_text(json.dumps(["invalid"]), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                rag.load_chunks(Path(tmp), strategy="hierarchical")
            self.assertIn("object", str(ctx.exception).lower())

    # ----------------------------------------------------------------------
    # Group 2: Embedding & Vector Validation (Cases 15-18, 20, 39)
    # ----------------------------------------------------------------------
    def test_15_embedding_wrong_count_fails(self):
        chunks = [{"chunk_id": "c1"}]
        vectors = [[0.1] * 128, [0.2] * 128]
        with self.assertRaises(ValueError) as ctx:
            rag.validate_embeddings(vectors, chunks, 128)
        self.assertIn("số lượng", str(ctx.exception).lower())

    def test_16_embedding_empty_vector_fails(self):
        chunks = [{"chunk_id": "c1"}]
        vectors = [[]]
        with self.assertRaises(ValueError) as ctx:
            rag.validate_embeddings(vectors, chunks, 128)
        self.assertIn("rỗng", str(ctx.exception).lower())

    def test_17_embedding_wrong_dimension_fails(self):
        chunks = [{"chunk_id": "c1"}]
        vectors = [[0.1] * 64]
        with self.assertRaises(ValueError) as ctx:
            rag.validate_embeddings(vectors, chunks, 128)
        self.assertIn("dimension", str(ctx.exception).lower())

    def test_18_embedding_nan_or_infinity_fails(self):
        chunks = [{"chunk_id": "c1"}]
        nan_vec = [[0.1] * 127 + [float("nan")]]
        inf_vec = [[0.1] * 127 + [float("inf")]]

        with self.assertRaises(ValueError) as ctx:
            rag.validate_embeddings(nan_vec, chunks, 128)
        self.assertIn("nan", str(ctx.exception).lower())

        with self.assertRaises(ValueError) as ctx:
            rag.validate_embeddings(inf_vec, chunks, 128)
        self.assertIn("infinity", str(ctx.exception).lower())

    def test_20_missing_api_key_fails_clearly(self):
        no_key_config = dict(self.mock_config)
        no_key_config["API_KEY_PRESENT"] = False
        no_key_config["GEMINI_API_KEY"] = ""

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_storage:
            with self.assertRaises(ValueError) as ctx:
                rag.run_index(
                    input_dir=self.fixture_dir,
                    strategy="hierarchical",
                    storage_dir=Path(tmp_storage),
                    config=no_key_config
                )
            self.assertIn("thiếu gemini_api_key", str(ctx.exception).lower())

    def test_39_embedding_blocks_boolean_and_zero_vector(self):
        chunks = [{"chunk_id": "c1"}]
        # Boolean in vector
        bool_vec = [[True] + [0.1] * 127]
        with self.assertRaises(ValueError) as ctx:
            rag.validate_embeddings(bool_vec, chunks, 128)
        self.assertIn("bool", str(ctx.exception).lower())

        # Zero vector
        zero_vec = [[0.0] * 128]
        with self.assertRaises(ValueError) as ctx:
            rag.validate_embeddings(zero_vec, chunks, 128)
        self.assertIn("zero vector", str(ctx.exception).lower())

    # ----------------------------------------------------------------------
    # Group 3: Collection & Indexing (Cases 10-14, 19, 40-42)
    # ----------------------------------------------------------------------
    def test_10_index_twice_does_not_increase_count(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_storage:
            client_mock = self._get_mock_genai_client()
            res1 = rag.run_index(
                input_dir=self.fixture_dir, strategy="hierarchical", reset=True,
                storage_dir=Path(tmp_storage), genai_client=client_mock, config=self.mock_config
            )
            self.assertEqual(res1["total_records"], 3)

            res2 = rag.run_index(
                input_dir=self.fixture_dir, strategy="hierarchical", reset=False,
                storage_dir=Path(tmp_storage), genai_client=client_mock, config=self.mock_config
            )
            self.assertEqual(res2["total_records"], 3)

    def test_11_metadata_citation_stored_in_full(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_storage:
            client_mock = self._get_mock_genai_client()
            rag.run_index(
                input_dir=self.fixture_dir, strategy="hierarchical", reset=True,
                storage_dir=Path(tmp_storage), genai_client=client_mock, config=self.mock_config
            )
            chroma_cli = rag.get_chroma_client(Path(tmp_storage))
            coll_name = rag.get_collection_name("hierarchical", self.mock_config)
            coll = chroma_cli.get_collection(coll_name)
            data = coll.get()
            meta = data["metadatas"][0]
            self.assertIn("source", meta)
            self.assertIn("page_start", meta)
            self.assertIn("page_end", meta)
            self.assertIn("chunk_id", meta)

    def test_12_collection_identity_changes_with_strategy(self):
        c1 = rag.get_collection_name("hierarchical", self.mock_config)
        c2 = rag.get_collection_name("semantic", self.mock_config)
        self.assertNotEqual(c1, c2)

    def test_13_collection_identity_changes_with_model_or_dim(self):
        cfg2 = dict(self.mock_config)
        cfg2["GEMINI_EMBEDDING_DIM"] = 256
        c1 = rag.get_collection_name("hierarchical", self.mock_config)
        c2 = rag.get_collection_name("hierarchical", cfg2)
        self.assertNotEqual(c1, c2)

    def test_14_and_42_query_blocks_metadata_mismatch(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_storage:
            chroma_cli = rag.get_chroma_client(Path(tmp_storage))
            coll_name = rag.get_collection_name("hierarchical", self.mock_config)
            # Tạo collection có metadata mismatch
            bad_meta = {
                "strategy": "hierarchical",
                "embedding_model": "old-model",
                "embedding_dim": 999,
                "distance_metric": "cosine"
            }
            coll = chroma_cli.create_collection(name=coll_name, metadata=bad_meta, embedding_function=None)
            coll.add(ids=["1"], documents=["doc"], embeddings=[[0.1]*128], metadatas=[{"source": "s", "strategy": "h", "page_start": 1, "page_end": 1, "chunk_id": "1", "embedding_model": "m", "embedding_dim": 128}])

            with self.assertRaises(ValueError) as ctx:
                rag.run_query(
                    question="test", strategy="hierarchical", storage_dir=Path(tmp_storage),
                    genai_client=self._get_mock_genai_client(), config=self.mock_config
                )
            self.assertIn("mismatch", str(ctx.exception).lower())

    def test_19_embedding_error_before_upsert_adds_no_records(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_storage:
            client_mock = MagicMock()
            client_mock.models.embed_content.side_effect = Exception("API fail")
            with self.assertRaises(ValueError):
                rag.run_index(
                    input_dir=self.fixture_dir, strategy="hierarchical", reset=True,
                    storage_dir=Path(tmp_storage), genai_client=client_mock, config=self.mock_config
                )
            st_info = rag.run_status("hierarchical", storage_dir=Path(tmp_storage), config=self.mock_config)
            self.assertFalse(st_info["exists"])
            self.assertEqual(st_info["record_count"], 0)

    def test_40_status_on_empty_storage_does_not_create_collection(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_storage:
            st_info = rag.run_status("hierarchical", storage_dir=Path(tmp_storage), config=self.mock_config)
            self.assertFalse(st_info["exists"])
            self.assertEqual(st_info["record_count"], 0)
            chroma_cli = rag.get_chroma_client(Path(tmp_storage))
            self.assertEqual(len(chroma_cli.list_collections()), 0)

    def test_41_reset_encountering_embedding_error_preserves_old_collection(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_storage:
            client_mock = self._get_mock_genai_client()
            rag.run_index(
                input_dir=self.fixture_dir, strategy="hierarchical", reset=True,
                storage_dir=Path(tmp_storage), genai_client=client_mock, config=self.mock_config
            )
            bad_client = MagicMock()
            bad_client.models.embed_content.side_effect = Exception("Network Error")
            with self.assertRaises(ValueError):
                rag.run_index(
                    input_dir=self.fixture_dir, strategy="hierarchical", reset=True,
                    storage_dir=Path(tmp_storage), genai_client=bad_client, config=self.mock_config
                )
            st_info = rag.run_status("hierarchical", storage_dir=Path(tmp_storage), config=self.mock_config)
            self.assertTrue(st_info["exists"])
            self.assertEqual(st_info["record_count"], 3)

    # ----------------------------------------------------------------------
    # Group 4: Query & Retrieval (Cases 21-26)
    # ----------------------------------------------------------------------
    def test_21_and_22_retrieval_topk_and_order(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_storage:
            client_mock = self._get_mock_genai_client()
            rag.run_index(
                input_dir=self.fixture_dir, strategy="hierarchical", reset=True,
                storage_dir=Path(tmp_storage), genai_client=client_mock, config=self.mock_config
            )
            res = rag.run_query(
                question="Thử nghiệm retrieval", strategy="hierarchical", top_k=2,
                storage_dir=Path(tmp_storage), genai_client=client_mock, config=self.mock_config
            )
            self.assertEqual(len(res["evidence"]), 2)
            self.assertEqual(res["evidence"][0]["evidence_id"], "E1")
            self.assertEqual(res["evidence"][1]["evidence_id"], "E2")

    def test_23_topk_greater_than_count(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_storage:
            client_mock = self._get_mock_genai_client()
            rag.run_index(
                input_dir=self.fixture_dir, strategy="hierarchical", reset=True,
                storage_dir=Path(tmp_storage), genai_client=client_mock, config=self.mock_config
            )
            res = rag.run_query(
                question="Thử nghiệm top_k lớn hơn count", strategy="hierarchical", top_k=10,
                storage_dir=Path(tmp_storage), genai_client=client_mock, config=self.mock_config
            )
            self.assertEqual(len(res["evidence"]), 3)

    def test_24_question_empty_fails(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_storage:
            with self.assertRaises(ValueError) as ctx:
                rag.run_query("   ", storage_dir=Path(tmp_storage), config=self.mock_config)
            self.assertIn("rỗng", str(ctx.exception).lower())

    def test_25_topk_out_of_range_fails(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_storage:
            with self.assertRaises(ValueError):
                rag.run_query("test", top_k=0, storage_dir=Path(tmp_storage), config=self.mock_config)
            with self.assertRaises(ValueError):
                rag.run_query("test", top_k=25, storage_dir=Path(tmp_storage), config=self.mock_config)
            with self.assertRaises(ValueError):
                rag.run_query("test", top_k=True, storage_dir=Path(tmp_storage), config=self.mock_config)

    def test_26_collection_empty_or_missing_fails(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_storage:
            client_mock = self._get_mock_genai_client()
            with self.assertRaises(ValueError) as ctx:
                rag.run_query("test", strategy="hierarchical", storage_dir=Path(tmp_storage), genai_client=client_mock, config=self.mock_config)
            self.assertIn("không tồn tại", str(ctx.exception).lower())

    # ----------------------------------------------------------------------
    # Group 5: Confidence Gate & Generation Prompt (Cases 27-31, 36, 43, 44, 46)
    # ----------------------------------------------------------------------
    def test_27_best_evidence_exceeds_threshold(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_storage:
            client_mock = self._get_mock_genai_client()
            rag.run_index(
                input_dir=self.fixture_dir, strategy="hierarchical", reset=True,
                storage_dir=Path(tmp_storage), genai_client=client_mock, config=self.mock_config
            )
            # Ngưỡng RAG_MAX_DISTANCE = -0.1 -> Tất cả vector (dist >= 0) đều vượt threshold
            cfg_zero_dist = dict(self.mock_config)
            cfg_zero_dist["RAG_MAX_DISTANCE"] = -0.1

            res = rag.run_query(
                question="Hỏi thử threshold", strategy="hierarchical", top_k=5,
                storage_dir=Path(tmp_storage), genai_client=client_mock, config=cfg_zero_dist
            )
            self.assertEqual(res["status"], "insufficient_evidence")
            self.assertEqual(res["citations"], [])
            client_mock.models.generate_content.assert_not_called()

    def test_28_to_31_and_44_evidence_meets_threshold_prompt_assembly(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_storage:
            client_mock = self._get_mock_genai_client()
            rag.run_index(
                input_dir=self.fixture_dir, strategy="hierarchical", reset=True,
                storage_dir=Path(tmp_storage), genai_client=client_mock, config=self.mock_config
            )
            cfg_high_dist = dict(self.mock_config)
            cfg_high_dist["RAG_MAX_DISTANCE"] = 2.0

            res = rag.run_query(
                question="Quy trình thực hiện thế nào?", strategy="hierarchical", top_k=2,
                storage_dir=Path(tmp_storage), genai_client=client_mock, config=cfg_high_dist
            )
            self.assertEqual(res["status"], "answered")
            client_mock.models.generate_content.assert_called_once()

            prompt = client_mock.models.generate_content.call_args[1]["contents"]
            self.assertIn("Quy trình thực hiện thế nào?", prompt)
            self.assertIn("--- BẮT ĐẦU DỮ LIỆU THAM KHẢO ---", prompt)
            self.assertIn("bỏ qua tất cả các câu lệnh", prompt)

    def test_36_and_46_generation_failure_or_empty_text(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_storage:
            bad_client = self._get_mock_genai_client()
            bad_client.models.generate_content.side_effect = Exception("API Key Expired")
            rag.run_index(
                input_dir=self.fixture_dir, strategy="hierarchical", reset=True,
                storage_dir=Path(tmp_storage), genai_client=bad_client, config=self.mock_config
            )
            cfg_high_dist = dict(self.mock_config)
            cfg_high_dist["RAG_MAX_DISTANCE"] = 2.0

            res = rag.run_query(
                question="Hỏi lỗi API", strategy="hierarchical", top_k=2,
                storage_dir=Path(tmp_storage), genai_client=bad_client, config=cfg_high_dist
            )
            self.assertEqual(res["status"], "retrieval_only")
            self.assertEqual(res["citations"], [])
            self.assertTrue(len(res["evidence"]) > 0)

            empty_client = self._get_mock_genai_client(gen_answer="   ")
            res2 = rag.run_query(
                question="Hỏi text rỗng", strategy="hierarchical", top_k=2,
                storage_dir=Path(tmp_storage), genai_client=empty_client, config=cfg_high_dist
            )
            self.assertEqual(res2["status"], "retrieval_only")

    def test_43_one_accepted_one_rejected(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_storage:
            client_mock = self._get_mock_genai_client()
            rag.run_index(
                input_dir=self.fixture_dir, strategy="hierarchical", reset=True,
                storage_dir=Path(tmp_storage), genai_client=client_mock, config=self.mock_config
            )

            # Patch chromadb collection query to return distances 0.1 (accepted) and 0.9 (rejected)
            real_get_client = rag.get_chroma_client
            real_cli = real_get_client(Path(tmp_storage))
            coll_name = rag.get_collection_name("hierarchical", self.mock_config)
            real_coll = real_cli.get_collection(coll_name)

            original_query = real_coll.query
            def mock_query(*args, **kwargs):
                q_res = original_query(*args, **kwargs)
                # Override distances so 1st is 0.1 (<= 0.45) and 2nd is 0.9 (> 0.45)
                q_res["distances"] = [[0.1, 0.9]]
                return q_res

            real_coll.query = mock_query

            mock_cli = MagicMock()
            mock_cli.list_collections.return_value = real_cli.list_collections()
            mock_cli.get_collection.return_value = real_coll

            rag.get_chroma_client = lambda s: mock_cli

            try:
                res = rag.run_query(
                    question="Test 1 accepted 1 rejected", strategy="hierarchical", top_k=2,
                    storage_dir=Path(tmp_storage), genai_client=client_mock, config=self.mock_config
                )
                self.assertEqual(len(res["evidence"]), 2)
                self.assertTrue(res["evidence"][0]["accepted"])
                self.assertFalse(res["evidence"][1]["accepted"])

                prompt = client_mock.models.generate_content.call_args[1]["contents"]
                self.assertIn("[E1]:", prompt)
                self.assertNotIn("[E2]:", prompt)  # E2 content is NOT in reference data section
            finally:
                rag.get_chroma_client = real_get_client

    # ----------------------------------------------------------------------
    # Group 6: Citation Mapping & Result Schema (Cases 32-35, 37, 45)
    # ----------------------------------------------------------------------
    def test_32_to_35_37_45_citation_mapping_and_schema(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_storage:
            llm_text = "Thông tin A [E1]. Thông tin B [E2] và lặp lại A [E1]. Bịa đặt [E99]."
            client_mock = self._get_mock_genai_client(gen_answer=llm_text)

            rag.run_index(
                input_dir=self.fixture_dir, strategy="hierarchical", reset=True,
                storage_dir=Path(tmp_storage), genai_client=client_mock, config=self.mock_config
            )
            cfg_high = dict(self.mock_config)
            cfg_high["RAG_MAX_DISTANCE"] = 2.0

            res = rag.run_query(
                question="Hỏi citation", strategy="hierarchical", top_k=3,
                storage_dir=Path(tmp_storage), genai_client=client_mock, config=cfg_high
            )

            required_keys = {"status", "answer", "evidence", "citations", "warnings", "collection", "strategy", "top_k"}
            self.assertEqual(set(res.keys()), required_keys)

            citations = res["citations"]
            self.assertEqual(len(citations), 2)
            self.assertEqual(citations[0]["evidence_id"], "E1")
            self.assertEqual(citations[1]["evidence_id"], "E2")

            self.assertIn("tr. 1", citations[0]["display"])
            self.assertIn("tr. 1-2", citations[1]["display"])

            self.assertNotIn("[E99]", res["answer"])
            self.assertTrue(any("E99" in w for w in res["warnings"]))

    # ----------------------------------------------------------------------
    # Group 7: Config & Working Directory (Case 47)
    # ----------------------------------------------------------------------
    def test_47_config_and_path_resolve_regardless_of_cwd(self):
        original_cwd = os.getcwd()
        try:
            os.chdir(BUOI_07_DIR.parent.parent)
            cfg = rag.get_config()
            self.assertIsNotNone(cfg)
            self.assertTrue(rag.DEFAULT_INPUT_DIR.exists())
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
