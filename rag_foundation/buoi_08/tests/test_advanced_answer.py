"""Tests cho Bước 08: query_advanced_rag và evaluate_dataset.

Nhóm test 20–30 theo spec:
- Mode validation, gate logic, citation, retrieval-only status
- Generation chỉ một lần (compare không generation)
- Trace schema
- Isolation/UI helpers: config, status read-only, không tải model khi import

Tất cả offline — mock LLM, reranker, Chroma.
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import tempfile
import os

BUOI08_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BUOI08_DIR))

import advanced_rag
from advanced_rag import query_advanced_rag, advanced_status, load_advanced_config

# ─────────────────────────────────────────────────────────────────────────────
# Shared fake candidates
# ─────────────────────────────────────────────────────────────────────────────

BM25_FAKE = [
    {
        "chunk_id": "C01", "text": "Điều 7 quy định cơ cấu lại thời hạn trả nợ.",
        "source": "TT02.pdf", "page_start": 1, "page_end": 1,
        "bm25_rank": 1, "bm25_score": 9.5,
    },
    {
        "chunk_id": "C02", "text": "Khoản 2 điều chỉnh kỳ hạn.",
        "source": "TT02.pdf", "page_start": 2, "page_end": 2,
        "bm25_rank": 2, "bm25_score": 5.0,
    },
]

SEM_FAKE = [
    {
        "chunk_id": "C01", "text": "Điều 7 quy định cơ cấu lại thời hạn trả nợ.",
        "source": "TT02.pdf", "page_start": 1, "page_end": 1,
        "semantic_rank": 1, "semantic_distance": 0.12,
    },
]

FUSED_FAKE = [
    {
        "chunk_id": "C01", "text": "Điều 7 quy định cơ cấu lại thời hạn trả nợ.",
        "source": "TT02.pdf", "page_start": 1, "page_end": 1,
        "bm25_rank": 1, "bm25_score": 9.5,
        "semantic_rank": 1, "semantic_distance": 0.12,
        "rrf_score": 0.032, "fused_rank": 1, "matched_by": ["bm25", "semantic"],
    },
    {
        "chunk_id": "C02", "text": "Khoản 2 điều chỉnh kỳ hạn.",
        "source": "TT02.pdf", "page_start": 2, "page_end": 2,
        "bm25_rank": 2, "bm25_score": 5.0,
        "semantic_rank": None, "semantic_distance": None,
        "rrf_score": 0.016, "fused_rank": 2, "matched_by": ["bm25"],
    },
]

RERANKED_FAKE = [
    {**FUSED_FAKE[0], "rerank_score": 0.88, "rerank_rank": 1, "accepted": True},
    {**FUSED_FAKE[1], "rerank_score": 0.20, "rerank_rank": 2, "accepted": False},
]

HYBRID_TRACE_FAKE = {
    "bm25_candidate_count": 2,
    "semantic_candidate_count": 1,
    "union_count": 2,
    "overlap_count": 1,
    "fused_count": 2,
    "rrf_k": 60,
    "rrf_bm25_weight": 1.0,
    "rrf_semantic_weight": 1.0,
    "latency_ms": {"bm25": 5.0, "semantic": 10.0, "fusion": 1.0},
}


def _fake_genai():
    """Tạo fake google.generativeai module."""
    fake_response = MagicMock()
    fake_response.text = "Đây là câu trả lời tổng hợp."
    fake_model = MagicMock()
    fake_model.generate_content.return_value = fake_response
    fake_genai = MagicMock()
    fake_genai.GenerativeModel.return_value = fake_model
    return fake_genai, fake_model


# ─────────────────────────────────────────────────────────────────────────────
# 20. Mode validation
# ─────────────────────────────────────────────────────────────────────────────
class TestModeValidation(unittest.TestCase):

    def test_invalid_mode_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            query_advanced_rag("câu hỏi", mode="invalid_mode")
        self.assertIn("mode", str(ctx.exception).lower())

    def test_empty_question_raises_value_error(self):
        with self.assertRaises(ValueError):
            query_advanced_rag("", mode="bm25")

    def test_whitespace_question_raises_value_error(self):
        with self.assertRaises(ValueError):
            query_advanced_rag("   ", mode="bm25")

    def test_all_valid_modes_accepted(self):
        """Các mode hợp lệ không raise ValueError khi validate."""
        for mode in ("bm25", "semantic", "hybrid", "hybrid_rerank"):
            with patch.object(advanced_rag, "bm25_search", return_value=BM25_FAKE):
                with patch.object(advanced_rag, "_import_rag") as mock_rag:
                    mock_rag.return_value = MagicMock(
                        load_chunks=MagicMock(return_value=([], {}))
                    )
                    try:
                        # Sẽ lỗi ở bước sau (API key thiếu) nhưng không raise ValueError cho mode
                        result = query_advanced_rag("câu hỏi test", mode=mode)
                    except ValueError as e:
                        if "mode" in str(e).lower():
                            self.fail(f"Mode '{mode}' bị từ chối sai: {e}")
                    except Exception:
                        pass  # Các lỗi khác (không tải được) là OK trong test


# ─────────────────────────────────────────────────────────────────────────────
# 21–22. Retrieval-only khi không có API key / missing genai
# ─────────────────────────────────────────────────────────────────────────────
class TestRetrievalOnlyFallback(unittest.TestCase):

    def _run_bm25_no_key(self, mode="bm25"):
        """Chạy query với BM25 mock, thiếu API key → trả retrieval_only."""
        with patch.object(advanced_rag, "bm25_search", return_value=BM25_FAKE):
            with patch.object(advanced_rag, "_import_rag") as mock_rag:
                mock_rag.return_value = MagicMock(
                    load_chunks=MagicMock(return_value=([], {}))
                )
                with patch.object(advanced_rag, "load_advanced_config", return_value={
                    "gemini_api_key": "",  # Thiếu key
                    "gemini_generation_model": "gemini-test",
                    "gemini_embedding_model": "emb-test",
                    "bm25_candidates": 20,
                    "semantic_candidates": 20,
                    "rerank_candidates": 10,
                    "final_top_k": 5,
                    "rerank_min_score": 0.5,
                }):
                    result = query_advanced_rag("Điều 7 cơ cấu nợ", mode=mode)
        return result

    def test_no_api_key_returns_retrieval_only(self):
        """Thiếu API key → status='retrieval_only', không raise Exception."""
        result = self._run_bm25_no_key()
        self.assertIn(result["status"], ("retrieval_only", "no_candidates"))

    def test_retrieval_only_has_evidence(self):
        """status='retrieval_only' vẫn có evidence từ retrieval."""
        result = self._run_bm25_no_key()
        if result["status"] == "retrieval_only":
            self.assertIsInstance(result["evidence"], list)

    def test_answer_is_none_when_no_key(self):
        """Khi không có API key, answer phải là None."""
        result = self._run_bm25_no_key()
        self.assertIsNone(result.get("answer"))


# ─────────────────────────────────────────────────────────────────────────────
# 23. Citation thật (từ metadata), không phải label giả
# ─────────────────────────────────────────────────────────────────────────────
class TestCitationFormat(unittest.TestCase):

    def test_citations_contain_real_source(self):
        """Citations phải chứa source và page từ metadata thật của chunk."""
        fake_genai, fake_model = _fake_genai()

        with patch.object(advanced_rag, "bm25_search", return_value=BM25_FAKE[:1]):
            with patch.object(advanced_rag, "_import_rag") as mock_rag:
                mock_rag.return_value = MagicMock(
                    load_chunks=MagicMock(return_value=([], {}))
                )
                with patch.object(advanced_rag, "load_advanced_config", return_value={
                    "gemini_api_key": "FAKE_KEY_FOR_TEST",
                    "gemini_generation_model": "gemini-test",
                    "gemini_embedding_model": "emb-test",
                    "bm25_candidates": 20,
                    "semantic_candidates": 20,
                    "rerank_candidates": 10,
                    "final_top_k": 5,
                    "rerank_min_score": 0.5,
                }):
                    with patch.dict("sys.modules", {"google.generativeai": fake_genai}):
                        result = query_advanced_rag("Điều 7", mode="bm25")

        if result["status"] == "ok":
            # Citation phải có source thật "TT02.pdf" từ metadata
            citations_text = " ".join(result.get("citations", []))
            self.assertIn("TT02.pdf", citations_text)


# ─────────────────────────────────────────────────────────────────────────────
# 24. no_candidates status
# ─────────────────────────────────────────────────────────────────────────────
class TestNoCandidatesStatus(unittest.TestCase):

    def test_empty_retrieval_returns_no_candidates(self):
        """Khi BM25 trả rỗng → status='no_candidates'."""
        with patch.object(advanced_rag, "bm25_search", return_value=[]):
            with patch.object(advanced_rag, "_import_rag") as mock_rag:
                mock_rag.return_value = MagicMock(
                    load_chunks=MagicMock(return_value=([], {}))
                )
                with patch.object(advanced_rag, "load_advanced_config", return_value={
                    "gemini_api_key": "FAKE",
                    "gemini_generation_model": "test",
                    "gemini_embedding_model": "emb",
                    "bm25_candidates": 20,
                    "semantic_candidates": 20,
                    "rerank_candidates": 10,
                    "final_top_k": 5,
                    "rerank_min_score": 0.5,
                }):
                    result = query_advanced_rag("câu hỏi không có kết quả", mode="bm25")

        self.assertEqual(result["status"], "no_candidates")
        self.assertIsNone(result["answer"])
        self.assertEqual(result["evidence"], [])


# ─────────────────────────────────────────────────────────────────────────────
# 25. Generation tối đa một lần
# ─────────────────────────────────────────────────────────────────────────────
class TestGenerationOnce(unittest.TestCase):

    def test_generation_called_exactly_once(self):
        """generate_content chỉ được gọi đúng một lần cho một query."""
        fake_genai, fake_model = _fake_genai()

        with patch.object(advanced_rag, "bm25_search", return_value=BM25_FAKE):
            with patch.object(advanced_rag, "_import_rag") as mock_rag:
                mock_rag.return_value = MagicMock(
                    load_chunks=MagicMock(return_value=([], {}))
                )
                with patch.object(advanced_rag, "load_advanced_config", return_value={
                    "gemini_api_key": "FAKE_KEY",
                    "gemini_generation_model": "gemini-test",
                    "gemini_embedding_model": "emb-test",
                    "bm25_candidates": 20,
                    "semantic_candidates": 20,
                    "rerank_candidates": 10,
                    "final_top_k": 5,
                    "rerank_min_score": 0.5,
                }):
                    with patch.dict("sys.modules", {"google.generativeai": fake_genai}):
                        query_advanced_rag("Điều 7", mode="bm25")

        fake_model.generate_content.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# 26. Compare không generation (Tab 2)
# ─────────────────────────────────────────────────────────────────────────────
class TestCompareNoGeneration(unittest.TestCase):

    def test_bm25_search_alone_no_generation(self):
        """bm25_search không gọi generation."""
        import inspect
        src = inspect.getsource(advanced_rag.bm25_search)
        self.assertNotIn("generate_content", src)
        self.assertNotIn("GenerativeModel", src)

    def test_hybrid_retrieval_alone_no_generation(self):
        """hybrid_retrieval không gọi generation."""
        import inspect
        src = inspect.getsource(advanced_rag.hybrid_retrieval)
        self.assertNotIn("generate_content", src)
        self.assertNotIn("GenerativeModel", src)

    def test_rrf_fusion_no_generation(self):
        """rrf_fusion không gọi generation."""
        import inspect
        src = inspect.getsource(advanced_rag.rrf_fusion)
        self.assertNotIn("generate_content", src)


# ─────────────────────────────────────────────────────────────────────────────
# 27. Trace schema / counts
# ─────────────────────────────────────────────────────────────────────────────
class TestTraceSchema(unittest.TestCase):

    def _run_hybrid_mode(self):
        with patch.object(advanced_rag, "hybrid_retrieval", return_value=(FUSED_FAKE, HYBRID_TRACE_FAKE)):
            with patch.object(advanced_rag, "load_advanced_config", return_value={
                "gemini_api_key": "",
                "gemini_generation_model": "test",
                "gemini_embedding_model": "emb",
                "bm25_candidates": 20,
                "semantic_candidates": 20,
                "rerank_candidates": 10,
                "final_top_k": 5,
                "rerank_min_score": 0.5,
            }):
                result = query_advanced_rag("câu hỏi", mode="hybrid")
        return result

    def test_trace_contains_mode(self):
        result = self._run_hybrid_mode()
        self.assertEqual(result["trace"].get("mode"), "hybrid")

    def test_trace_contains_strategy(self):
        result = self._run_hybrid_mode()
        self.assertIn("strategy", result["trace"])

    def test_trace_has_latency_ms(self):
        result = self._run_hybrid_mode()
        self.assertIn("latency_ms", result["trace"])
        self.assertIsInstance(result["trace"]["latency_ms"], dict)

    def test_trace_has_candidate_counts(self):
        result = self._run_hybrid_mode()
        trace = result["trace"]
        self.assertIn("bm25_candidate_count", trace)
        self.assertIn("semantic_candidate_count", trace)
        self.assertIn("union_count", trace)

    def test_result_has_required_keys(self):
        result = self._run_hybrid_mode()
        for key in ("status", "mode", "answer", "citations", "evidence", "warnings", "trace"):
            self.assertIn(key, result, f"Thiếu key: {key}")


# ─────────────────────────────────────────────────────────────────────────────
# 28. Config hoạt động khác cwd
# ─────────────────────────────────────────────────────────────────────────────
class TestConfigIndependentOfCwd(unittest.TestCase):

    def test_load_advanced_config_works_from_any_cwd(self):
        """load_advanced_config phải load được .env bất kể cwd hiện tại."""
        original_cwd = os.getcwd()
        tmpdir = tempfile.mkdtemp()
        try:
            os.chdir(tmpdir)
            cfg = load_advanced_config()
            self.assertIsInstance(cfg, dict)
        except Exception as exc:
            self.fail(f"load_advanced_config() thất bại khi cwd={tmpdir}: {exc}")
        finally:
            os.chdir(original_cwd)
            try:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# 29. advanced_status không tạo resource
# ─────────────────────────────────────────────────────────────────────────────
class TestStatusReadOnly(unittest.TestCase):

    def test_advanced_status_does_not_create_collection(self):
        """advanced_status phải dùng get_collection_or_none, không create_collection."""
        import inspect
        src = inspect.getsource(advanced_rag.advanced_status)
        self.assertNotIn("create_collection", src)

    def test_advanced_status_no_download(self):
        """advanced_status chỉ kiểm tra filesystem, không download model."""
        import inspect
        src = inspect.getsource(advanced_rag.advanced_status)
        self.assertNotIn("from_pretrained", src)
        self.assertNotIn("hf_hub_download", src)

    def test_advanced_status_returns_dict_with_required_keys(self):
        """advanced_status phải trả dict với ít nhất các key cần thiết."""
        with patch.object(advanced_rag, "_import_rag") as mock_rag:
            mock_rag.return_value = MagicMock(
                load_config=MagicMock(return_value={
                    "gemini_api_key": "FAKE",
                    "gemini_embedding_model": "emb",
                    "gemini_embedding_dim": 128,
                }),
                load_chunks=MagicMock(return_value=([], {"valid_chunks": 0})),
                build_collection_name=MagicMock(return_value="test-col"),
                get_chroma_client=MagicMock(return_value=MagicMock()),
                get_collection_or_none=MagicMock(return_value=None),
            )
            result = advanced_status()

        required = {"reranker_cache_exists", "reranker_model", "collection_count"}
        missing = required - set(result.keys())
        self.assertEqual(missing, set(), f"Thiếu keys: {missing}")


# ─────────────────────────────────────────────────────────────────────────────
# 30. Không tải model khi import / test
# ─────────────────────────────────────────────────────────────────────────────
class TestNoModelLoadOnImport(unittest.TestCase):

    def test_advanced_rag_top_level_no_transformers_import(self):
        """Top-level module không import transformers/torch."""
        import inspect
        src = inspect.getsource(advanced_rag)
        # Lấy các dòng đầu (trước hàm đầu tiên có 'def ')
        lines = src.split("\n")
        header_lines = []
        for line in lines:
            if line.startswith("def ") or line.startswith("class "):
                break
            header_lines.append(line)
        header = "\n".join(header_lines)
        self.assertNotIn("from transformers", header)
        self.assertNotIn("import torch", header)

    def test_no_model_init_at_module_level(self):
        """Module không khởi tạo model hoặc tokenizer ở cấp module."""
        import inspect
        src = inspect.getsource(advanced_rag)
        # Tìm các dòng có from_pretrained ngoài hàm
        # (Sẽ chỉ xuất hiện bên trong def rerank_candidates)
        in_function = False
        for line in src.split("\n"):
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("class "):
                in_function = True
            if not in_function and "from_pretrained" in stripped:
                self.fail(f"from_pretrained ở cấp module: {line}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metric unit tests (công thức tính tay)
# ─────────────────────────────────────────────────────────────────────────────
class TestEvaluationMetrics(unittest.TestCase):
    """Tests công thức metric với ranking nhỏ tính tay được."""

    def setUp(self):
        # Import từ evaluate
        eval_dir = BUOI08_DIR
        if str(eval_dir) not in sys.path:
            sys.path.insert(0, str(eval_dir))
        import evaluate as ev_module
        self.ev = ev_module

    def test_recall_at_k_perfect(self):
        """Recall@2: retrieved=[A,B], relevant=[A,B] → 1.0"""
        self.assertAlmostEqual(
            self.ev.recall_at_k(["A", "B"], ["A", "B"], k=2), 1.0
        )

    def test_recall_at_k_half(self):
        """Recall@2: retrieved=[A,C], relevant=[A,B] → 0.5"""
        self.assertAlmostEqual(
            self.ev.recall_at_k(["A", "C"], ["A", "B"], k=2), 0.5
        )

    def test_recall_at_k_zero(self):
        """Recall@2: retrieved=[C,D], relevant=[A,B] → 0.0"""
        self.assertAlmostEqual(
            self.ev.recall_at_k(["C", "D"], ["A", "B"], k=2), 0.0
        )

    def test_recall_empty_relevant(self):
        """Recall với relevant=[] → 0.0 (không phạt)."""
        self.assertAlmostEqual(self.ev.recall_at_k(["A"], [], k=1), 0.0)

    def test_mrr_at_k_first_hit(self):
        """MRR@3: retrieved=[A,B,C], relevant=[A] → 1/1 = 1.0"""
        self.assertAlmostEqual(
            self.ev.mrr_at_k(["A", "B", "C"], ["A"], k=3), 1.0
        )

    def test_mrr_at_k_second_hit(self):
        """MRR@3: retrieved=[X,A,C], relevant=[A] → 1/2 = 0.5"""
        self.assertAlmostEqual(
            self.ev.mrr_at_k(["X", "A", "C"], ["A"], k=3), 0.5
        )

    def test_mrr_no_hit(self):
        """MRR@3: retrieved=[X,Y,Z], relevant=[A] → 0.0"""
        self.assertAlmostEqual(
            self.ev.mrr_at_k(["X", "Y", "Z"], ["A"], k=3), 0.0
        )

    def test_ndcg_at_k_perfect(self):
        """nDCG@2: retrieved=[A,B], relevant=[A,B] → 1.0"""
        import math
        dcg = 1 / math.log2(2) + 1 / math.log2(3)
        idcg = dcg
        expected = dcg / idcg
        self.assertAlmostEqual(
            self.ev.ndcg_at_k(["A", "B"], ["A", "B"], k=2), expected
        )

    def test_ndcg_at_k_single_first(self):
        """nDCG@2: retrieved=[A,X], relevant=[A] → 1.0"""
        import math
        dcg = 1 / math.log2(2)
        idcg = 1 / math.log2(2)
        self.assertAlmostEqual(
            self.ev.ndcg_at_k(["A", "X"], ["A"], k=2), 1.0
        )

    def test_ndcg_at_k_second_position(self):
        """nDCG@2: retrieved=[X,A], relevant=[A] → 1/log2(3) / 1/log2(2)"""
        import math
        dcg = 1 / math.log2(3)
        idcg = 1 / math.log2(2)
        expected = dcg / idcg
        self.assertAlmostEqual(
            self.ev.ndcg_at_k(["X", "A"], ["A"], k=2), expected, places=6
        )

    def test_evaluate_dataset_with_mock_retriever(self):
        """evaluate_dataset với mock retriever phải trả report đúng schema."""
        import evaluate as ev_module

        # Tạo file câu hỏi tạm
        qs = [
            {
                "query_id": "Q01",
                "question": "test?",
                "relevant_chunk_ids": ["C01"],
                "needs_human_review": True,
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            q_path = Path(tmpdir) / "questions.json"
            q_path.write_text(json.dumps(qs), encoding="utf-8")

            # Mock retriever trả deterministic kết quả
            def fake_retriever(question, mode, strategy, k, adv):
                return (["C01", "C02"], 5.0)

            report = ev_module.evaluate_dataset(
                questions_path=str(q_path),
                modes=["bm25"],
                strategy="hierarchical",
                top_k=2,
                retriever_fn=fake_retriever,
            )

        self.assertIn("timestamp", report)
        self.assertIn("config", report)
        self.assertIn("summary", report)
        self.assertIn("per_query", report)
        self.assertTrue(report["needs_human_review"])
        self.assertGreater(len(report["warnings"]), 0)

        # Recall@2: retrieved=[C01, C02], relevant=[C01] → 1.0
        q_row = report["per_query"][0]
        self.assertAlmostEqual(q_row["recall"], 1.0)
        self.assertAlmostEqual(q_row["mrr"], 1.0)
        self.assertEqual(q_row["status"], "ok")

    def test_evaluate_dataset_fail_per_query_not_silent(self):
        """Khi retriever raise exception, query có status='fail' và ghi error rõ."""
        import evaluate as ev_module

        qs = [{"query_id": "Q_FAIL", "question": "lỗi", "relevant_chunk_ids": ["X"]}]

        with tempfile.TemporaryDirectory() as tmpdir:
            q_path = Path(tmpdir) / "q.json"
            q_path.write_text(json.dumps(qs), encoding="utf-8")

            def failing_retriever(question, mode, strategy, k, adv):
                raise RuntimeError("Lỗi giả lập khi retrieve")

            report = ev_module.evaluate_dataset(
                questions_path=str(q_path),
                modes=["bm25"],
                strategy="hierarchical",
                top_k=5,
                retriever_fn=failing_retriever,
            )

        fail_row = report["per_query"][0]
        self.assertEqual(fail_row["status"], "fail")
        self.assertIn("error", fail_row)
        self.assertIn("Lỗi giả lập", fail_row["error"])


if __name__ == "__main__":
    unittest.main()
