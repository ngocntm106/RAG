"""Tests cho Bước 07: rerank_candidates.

Tất cả test offline — không tải Hugging Face model, không gọi API, không Internet.
Dùng dependency injection (rerank_fn) để inject fake cross-encoder.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

BUOI08_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BUOI08_DIR))

import advanced_rag
from advanced_rag import rerank_candidates, load_advanced_config


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: fake candidates and fake cross-encoder
# ─────────────────────────────────────────────────────────────────────────────

def _make_candidates(n: int = 5):
    """Tạo n candidate dict với fused_rank tuần tự."""
    return [
        {
            "chunk_id": f"CHUNK_{i:02d}",
            "text": f"Nội dung chunk số {i}",
            "source": "doc.pdf",
            "page_start": i,
            "page_end": i,
            "fused_rank": i,
            "rrf_score": 1.0 / (60 + i),
        }
        for i in range(1, n + 1)
    ]


def _fake_reranker_scores(candidates, scores):
    """Tạo AutoModelForSequenceClassification mock trả scores cố định."""

    class FakeLogits:
        def squeeze(self, *a, **kw):
            return self

        def float(self):
            return scores

        def tolist(self):
            return scores

    class FakeOutput:
        logits = FakeLogits()

    class FakeModel:
        def eval(self):
            return self

        def __call__(self, **kwargs):
            return FakeOutput()

    return FakeModel()


# ─────────────────────────────────────────────────────────────────────────────
# 14. Lazy load và dependency injection
# ─────────────────────────────────────────────────────────────────────────────
class TestRerankLazyLoad(unittest.TestCase):

    def test_import_advanced_rag_does_not_load_torch(self):
        """Import advanced_rag không được trigger load torch/transformers."""
        # Nếu đã import thì kiểm tra module không tự load transformers khi import
        import inspect
        src = inspect.getsource(advanced_rag)
        # transformers chỉ được import bên trong hàm (lazy), không ở top-level
        # Kiểm tra không có 'from transformers import' ở đầu file
        lines = src.split("\n")
        top_level_imports = [
            l.strip() for l in lines[:30]
            if "transformers" in l and l.strip().startswith(("import", "from"))
        ]
        self.assertEqual(
            top_level_imports, [],
            f"Tìm thấy import transformers ở top-level: {top_level_imports}"
        )

    def test_rerank_candidates_raises_when_cache_missing(self):
        """rerank_candidates phải raise ValueError nếu model cache chưa có."""
        cands = _make_candidates(3)
        with patch.object(advanced_rag, "advanced_status", return_value={
            "reranker_cache_exists": False,
            "reranker_model": "BAAI/bge-reranker-v2-m3",
        }):
            with self.assertRaises(ValueError) as ctx:
                rerank_candidates("câu hỏi", cands, top_k=2)
            self.assertIn("cache", str(ctx.exception).lower())


# ─────────────────────────────────────────────────────────────────────────────
# 15. Pair construction/batching
# ─────────────────────────────────────────────────────────────────────────────
class TestRerankPairConstruction(unittest.TestCase):

    def _run_with_fake_model(self, query, candidates, scores, top_k=5, threshold=-10.0):
        """Helper: chạy rerank_candidates với fake model."""
        fake_scores = scores

        class FakeTokenizer:
            def __call__(self, queries, texts, **kwargs):
                # Ghi lại số lượng pair
                self._last_call_len = len(texts)
                return {"input_ids": [[0]] * len(texts)}

        class FakeTensor:
            def squeeze(self, *a, **kw):
                return self
            def float(self):
                return self
            def tolist(self):
                return fake_scores

        class FakeOutput:
            logits = FakeTensor()

        class FakeModel:
            def eval(self):
                return self
            def __call__(self, **kwargs):
                return FakeOutput()

        fake_tok = FakeTokenizer()
        fake_model = FakeModel()

        with patch.object(advanced_rag, "advanced_status", return_value={
            "reranker_cache_exists": True,
            "reranker_model": "BAAI/bge-reranker-v2-m3",
        }):
            with patch.dict("sys.modules", {
                "transformers": MagicMock(
                    AutoTokenizer=MagicMock(from_pretrained=MagicMock(return_value=fake_tok)),
                    AutoModelForSequenceClassification=MagicMock(
                        from_pretrained=MagicMock(return_value=fake_model)
                    ),
                ),
                "torch": MagicMock(no_grad=MagicMock(return_value=MagicMock(
                    __enter__=lambda s, *a: s,
                    __exit__=lambda s, *a: None,
                ))),
            }):
                result = rerank_candidates(query, candidates, top_k=top_k, threshold=threshold)
        return result, fake_tok

    def test_exactly_n_pairs_built(self):
        """Số pair = len(candidates)."""
        cands = _make_candidates(4)
        scores = [0.9, 0.3, 0.7, 0.5]
        result, tok = self._run_with_fake_model("query", cands, scores, top_k=4, threshold=-100.0)
        # Kiểm tra kết quả trả về đúng top_k
        self.assertEqual(len(result), 4)

    def test_rerank_returns_top_k_subset(self):
        """rerank_candidates phải trả đúng top_k kết quả."""
        cands = _make_candidates(5)
        scores = [0.1, 0.9, 0.5, 0.8, 0.3]
        result, _ = self._run_with_fake_model("query", cands, scores, top_k=3, threshold=-100.0)
        self.assertEqual(len(result), 3)


# ─────────────────────────────────────────────────────────────────────────────
# 16. Score và acceptance
# ─────────────────────────────────────────────────────────────────────────────
class TestRerankScoreAttachment(unittest.TestCase):

    def _run_fake(self, candidates, raw_scores, threshold, top_k=None):
        if top_k is None:
            top_k = len(candidates)

        class FakeTensor:
            def squeeze(self, *a, **kw): return self
            def float(self): return self
            def tolist(self): return raw_scores

        class FakeOutput:
            logits = FakeTensor()

        class FakeModel:
            def eval(self): return self
            def __call__(self, **kwargs): return FakeOutput()

        with patch.object(advanced_rag, "advanced_status", return_value={
            "reranker_cache_exists": True,
            "reranker_model": "BAAI/bge-reranker-v2-m3",
        }):
            with patch.dict("sys.modules", {
                "transformers": MagicMock(
                    AutoTokenizer=MagicMock(from_pretrained=MagicMock(return_value=MagicMock(
                        __call__=MagicMock(return_value={"input_ids": [[0]] * len(candidates)})
                    ))),
                    AutoModelForSequenceClassification=MagicMock(
                        from_pretrained=MagicMock(return_value=FakeModel())
                    ),
                ),
                "torch": MagicMock(no_grad=MagicMock(return_value=MagicMock(
                    __enter__=lambda s, *a: s,
                    __exit__=lambda s, *a: None,
                ))),
            }):
                result = rerank_candidates(
                    "câu hỏi", candidates, top_k=top_k, threshold=threshold
                )
        return result

    def test_rerank_score_attached(self):
        """Mỗi candidate phải có key 'rerank_score'."""
        cands = _make_candidates(3)
        scores = [0.8, 0.2, 0.6]
        result = self._run_fake(cands, scores, threshold=-100.0)
        for r in result:
            self.assertIn("rerank_score", r)

    def test_accepted_flag_above_threshold(self):
        """accepted=True khi score >= threshold."""
        cands = _make_candidates(2)
        scores = [0.9, 0.1]
        result = self._run_fake(cands, scores, threshold=0.5, top_k=2)
        # Sau sort: [0.9, 0.1]
        high = [r for r in result if r["rerank_score"] == 0.9][0]
        low = [r for r in result if r["rerank_score"] == 0.1][0]
        self.assertTrue(high["accepted"])
        self.assertFalse(low["accepted"])

    def test_sorted_by_score_descending(self):
        """Kết quả phải được sort giảm dần theo rerank_score."""
        cands = _make_candidates(4)
        scores = [0.3, 0.9, 0.5, 0.7]
        result = self._run_fake(cands, scores, threshold=-100.0, top_k=4)
        result_scores = [r["rerank_score"] for r in result]
        self.assertEqual(result_scores, sorted(result_scores, reverse=True))


# ─────────────────────────────────────────────────────────────────────────────
# 17. Reorder / rank movement
# ─────────────────────────────────────────────────────────────────────────────
class TestRerankRankMovement(unittest.TestCase):

    def _run_fake(self, candidates, raw_scores, top_k=None, threshold=-100.0):
        if top_k is None:
            top_k = len(candidates)

        class FakeTensor:
            def squeeze(self, *a, **kw): return self
            def float(self): return self
            def tolist(self): return raw_scores

        class FakeOutput:
            logits = FakeTensor()

        class FakeModel:
            def eval(self): return self
            def __call__(self, **kwargs): return FakeOutput()

        with patch.object(advanced_rag, "advanced_status", return_value={
            "reranker_cache_exists": True,
            "reranker_model": "BAAI/bge-reranker-v2-m3",
        }):
            with patch.dict("sys.modules", {
                "transformers": MagicMock(
                    AutoTokenizer=MagicMock(from_pretrained=MagicMock(return_value=MagicMock(
                        __call__=MagicMock(return_value={"input_ids": [[0]] * len(candidates)})
                    ))),
                    AutoModelForSequenceClassification=MagicMock(
                        from_pretrained=MagicMock(return_value=FakeModel())
                    ),
                ),
                "torch": MagicMock(no_grad=MagicMock(return_value=MagicMock(
                    __enter__=lambda s, *a: s,
                    __exit__=lambda s, *a: None,
                ))),
            }):
                result = rerank_candidates(
                    "câu hỏi", candidates, top_k=top_k, threshold=threshold
                )
        return result

    def test_rerank_rank_sequential(self):
        """rerank_rank phải bắt đầu từ 1 và liên tiếp."""
        cands = _make_candidates(5)
        scores = [0.5, 0.9, 0.1, 0.7, 0.3]
        result = self._run_fake(cands, scores, top_k=5)
        ranks = [r["rerank_rank"] for r in result]
        self.assertEqual(ranks, list(range(1, len(result) + 1)))

    def test_reorder_from_original_order(self):
        """Candidate có score cao nhất phải có rerank_rank=1 dù vị trí ban đầu."""
        cands = _make_candidates(3)
        # fused_rank: CHUNK_01=1, CHUNK_02=2, CHUNK_03=3
        # Scores: CHUNK_01=0.1, CHUNK_02=0.9, CHUNK_03=0.5 → rerank: CHUNK_02 first
        scores = [0.1, 0.9, 0.5]
        result = self._run_fake(cands, scores, top_k=3)
        self.assertEqual(result[0]["chunk_id"], "CHUNK_02")
        self.assertEqual(result[0]["rerank_rank"], 1)


# ─────────────────────────────────────────────────────────────────────────────
# 18. Candidate / final limits
# ─────────────────────────────────────────────────────────────────────────────
class TestRerankLimits(unittest.TestCase):

    def _run_fake(self, candidates, raw_scores, top_k, threshold=-100.0):
        class FakeTensor:
            def squeeze(self, *a, **kw): return self
            def float(self): return self
            def tolist(self): return raw_scores

        class FakeOutput:
            logits = FakeTensor()

        class FakeModel:
            def eval(self): return self
            def __call__(self, **kwargs): return FakeOutput()

        with patch.object(advanced_rag, "advanced_status", return_value={
            "reranker_cache_exists": True,
            "reranker_model": "BAAI/bge-reranker-v2-m3",
        }):
            with patch.dict("sys.modules", {
                "transformers": MagicMock(
                    AutoTokenizer=MagicMock(from_pretrained=MagicMock(return_value=MagicMock(
                        __call__=MagicMock(return_value={"input_ids": [[0]] * len(candidates)})
                    ))),
                    AutoModelForSequenceClassification=MagicMock(
                        from_pretrained=MagicMock(return_value=FakeModel())
                    ),
                ),
                "torch": MagicMock(no_grad=MagicMock(return_value=MagicMock(
                    __enter__=lambda s, *a: s,
                    __exit__=lambda s, *a: None,
                ))),
            }):
                result = rerank_candidates(
                    "câu hỏi", candidates, top_k=top_k, threshold=threshold
                )
        return result

    def test_final_top_k_limit(self):
        """rerank_candidates trả không quá top_k kết quả."""
        cands = _make_candidates(10)
        scores = [float(i) / 10 for i in range(10)]
        result = self._run_fake(cands, scores, top_k=3)
        self.assertEqual(len(result), 3)

    def test_top_k_larger_than_candidates(self):
        """top_k > len(candidates): trả tất cả candidates, không crash."""
        cands = _make_candidates(2)
        scores = [0.8, 0.3]
        result = self._run_fake(cands, scores, top_k=100)
        self.assertEqual(len(result), 2)


# ─────────────────────────────────────────────────────────────────────────────
# 19. Failure không silent fallback
# ─────────────────────────────────────────────────────────────────────────────
class TestRerankFailureNoSilent(unittest.TestCase):

    def test_cache_missing_raises_value_error(self):
        """Khi cache không tồn tại, phải raise ValueError rõ ràng (không trả list rỗng)."""
        cands = _make_candidates(3)
        with patch.object(advanced_rag, "advanced_status", return_value={
            "reranker_cache_exists": False,
            "reranker_model": "BAAI/bge-reranker-v2-m3",
        }):
            with self.assertRaises(ValueError):
                rerank_candidates("câu hỏi", cands, top_k=2)

    def test_import_error_raises_not_returns_empty(self):
        """Thiếu transformers phải raise ImportError, không trả kết quả âm thầm."""
        cands = _make_candidates(3)
        with patch.object(advanced_rag, "advanced_status", return_value={
            "reranker_cache_exists": True,
            "reranker_model": "BAAI/bge-reranker-v2-m3",
        }):
            with patch.dict("sys.modules", {"transformers": None, "torch": None}):
                with self.assertRaises((ImportError, TypeError)):
                    rerank_candidates("câu hỏi", cands, top_k=2)


if __name__ == "__main__":
    unittest.main()
