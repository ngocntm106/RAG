"""Unit tests cho Bước 05 — Multi-Query Child Retrieval & Cross-Query RRF Fusion.

Bao gồm 12 test cases bắt buộc (100% offline, dùng fake generator & fake hybrid retriever):
1. Công thức MQ-RRF tính tay.
2. Original/variant weights.
3. Deduplicate union.
4. Missing query contribution.
5. Support query count/IDs.
6. Metadata mismatch fail.
7. Deterministic tie-break.
8. Mỗi query gọi hybrid đúng một lần.
9. Không gọi reranker/generation.
10. Q0 failure và generated-query partial status.
11. Trace counts/latency schema.
12. Tests 100% offline.
"""

import json
import unittest
from typing import Any, Dict, List, Tuple

from hierarchical_rag import (
    clear_query_expansion_cache,
    retrieve_multi_query_children,
)


class TestStep05MultiChild(unittest.TestCase):

    def setUp(self):
        clear_query_expansion_cache()
        self.config = {
            "multi_query_count": 2,
            "multi_query_max_chars": 300,
            "multi_query_original_weight": 1.5,
            "multi_query_variant_weight": 1.0,
            "multi_query_rrf_k": 60,
            "per_query_candidates": 12,
            "gemini_generation_model": "gemini-3.5-flash-lite",
        }

    def tearDown(self):
        clear_query_expansion_cache()

    def _mock_query_gen(self, prompt: str, cfg: Dict) -> str:
        """Fake query generator sinh Q1, Q2."""
        return json.dumps({
            "queries": [
                {"text": "Biến thể 1", "focus": "exact_legal_terms"},
                {"text": "Biến thể 2", "focus": "paraphrase"},
            ]
        })

    def test_01_mq_rrf_manual_calculation(self):
        """Tính tay công thức Cross-Query RRF: w / (60 + rank).

        Child A: Q0 rank 1 => 1.5 / (60 + 1) = 1.5 / 61 = 0.02459016
        Child B: Q1 rank 1 => 1.0 / (60 + 1) = 1.0 / 61 = 0.01639344
        """
        def mock_retriever(qtext: str, cfg: Dict) -> List[Dict]:
            if "Biến thể 1" in qtext:
                return [{"child_id": "child_B", "text": "Text B", "source": "src.pdf", "page_start": 1, "page_end": 1}]
            if "Câu hỏi Q0" in qtext:
                return [{"child_id": "child_A", "text": "Text A", "source": "src.pdf", "page_start": 1, "page_end": 1}]
            return []

        res = retrieve_multi_query_children(
            "Câu hỏi Q0",
            config=self.config,
            query_generator_fn=self._mock_query_gen,
            hybrid_retriever_fn=mock_retriever,
        )
        hits = res["merged_child_hits"]
        self.assertEqual(len(hits), 2)

        # child_A top 1
        self.assertEqual(hits[0]["child_id"], "child_A")
        self.assertAlmostEqual(hits[0]["multi_query_rrf_score"], round(1.5 / 61, 6), places=5)

        # child_B top 2
        self.assertEqual(hits[1]["child_id"], "child_B")
        self.assertAlmostEqual(hits[1]["multi_query_rrf_score"], round(1.0 / 61, 6), places=5)

    def test_02_original_vs_variant_weights(self):
        """Q0 có weight 1.5 cao hơn variant weight 1.0 khi cùng ở rank 1."""
        def mock_retriever(qtext: str, cfg: Dict) -> List[Dict]:
            if "Biến thể 1" in qtext:
                return [{"child_id": "child_var", "text": "Text V", "source": "src.pdf", "page_start": 1, "page_end": 1}]
            if "Câu hỏi Q0" in qtext:
                return [{"child_id": "child_orig", "text": "Text O", "source": "src.pdf", "page_start": 1, "page_end": 1}]
            return []

        res = retrieve_multi_query_children(
            "Câu hỏi Q0",
            config=self.config,
            query_generator_fn=self._mock_query_gen,
            hybrid_retriever_fn=mock_retriever,
        )
        hits = res["merged_child_hits"]
        self.assertGreater(hits[0]["multi_query_rrf_score"], hits[1]["multi_query_rrf_score"])
        self.assertEqual(hits[0]["child_id"], "child_orig")

    def test_03_deduplicate_union(self):
        """Child xuất hiện ở nhiều query được hợp nhất duy nhất 1 item với support_query_count = 3."""
        def mock_retriever(qtext: str, cfg: Dict) -> List[Dict]:
            return [{"child_id": "child_shared", "text": "Text Shared", "source": "src.pdf", "page_start": 1, "page_end": 1}]

        res = retrieve_multi_query_children(
            "Câu hỏi Q0",
            config=self.config,
            query_generator_fn=self._mock_query_gen,
            hybrid_retriever_fn=mock_retriever,
        )
        hits = res["merged_child_hits"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["child_id"], "child_shared")
        self.assertEqual(hits[0]["support_query_count"], 3)  # Q0, Q1, Q2
        self.assertEqual(hits[0]["support_query_ids"], ["Q0", "Q1", "Q2"])

    def test_04_missing_query_contribution(self):
        """Candidate chỉ xuất hiện ở Q2 vẫn được giữ trong kết quả hợp nhất."""
        def mock_retriever(qtext: str, cfg: Dict) -> List[Dict]:
            if "Biến thể 2" in qtext:
                return [{"child_id": "child_q2", "text": "Text Q2", "source": "src.pdf", "page_start": 2, "page_end": 2}]
            return []

        res = retrieve_multi_query_children(
            "Câu hỏi Q0",
            config=self.config,
            query_generator_fn=self._mock_query_gen,
            hybrid_retriever_fn=mock_retriever,
        )
        hits = res["merged_child_hits"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["child_id"], "child_q2")
        self.assertEqual(hits[0]["support_query_ids"], ["Q2"])

    def test_05_support_query_count_and_ids(self):
        """support_query_ids giữ thứ tự chuẩn Q0, Q1, Q2..."""
        def mock_retriever(qtext: str, cfg: Dict) -> List[Dict]:
            if "Biến thể 2" in qtext:
                return [{"child_id": "child_1", "text": "T1", "source": "src.pdf", "page_start": 1, "page_end": 1}]
            if "Câu hỏi Q0" in qtext:
                return [{"child_id": "child_1", "text": "T1", "source": "src.pdf", "page_start": 1, "page_end": 1}]
            return []

        res = retrieve_multi_query_children(
            "Câu hỏi Q0",
            config=self.config,
            query_generator_fn=self._mock_query_gen,
            hybrid_retriever_fn=mock_retriever,
        )
        hits = res["merged_child_hits"]
        self.assertEqual(hits[0]["support_query_ids"], ["Q0", "Q2"])
        self.assertEqual(hits[0]["support_query_count"], 2)

    def test_06_metadata_mismatch_fail(self):
        """Khi cùng child_id nhưng text/source khác nhau từ 2 query -> Raise ValueError."""
        def mock_retriever(qtext: str, cfg: Dict) -> List[Dict]:
            if "Biến thể 1" in qtext:
                return [{"child_id": "child_X", "text": "Text Conflict!", "source": "src.pdf", "page_start": 1, "page_end": 1}]
            return [{"child_id": "child_X", "text": "Text Initial", "source": "src.pdf", "page_start": 1, "page_end": 1}]

        with self.assertRaises(ValueError) as ctx:
            retrieve_multi_query_children(
                "Câu hỏi Q0",
                config=self.config,
                query_generator_fn=self._mock_query_gen,
                hybrid_retriever_fn=mock_retriever,
            )
        self.assertIn("Metadata mismatch", str(ctx.exception))

    def test_07_deterministic_tie_break(self):
        """Tie break bằng child_id lexicographical khi score, support_count, best_rank bằng nhau."""
        # Q1 trả child_Z (rank 1), Q2 trả child_A (rank 1) -> score=1.0/61, support=1, best_rank=1 cho cả hai
        def mock_retriever(qtext: str, cfg: Dict) -> List[Dict]:
            if "Biến thể 1" in qtext:
                return [{"child_id": "child_Z", "text": "Text Z", "source": "src.pdf", "page_start": 1, "page_end": 1}]
            if "Biến thể 2" in qtext:
                return [{"child_id": "child_A", "text": "Text A", "source": "src.pdf", "page_start": 1, "page_end": 1}]
            return []

        res = retrieve_multi_query_children(
            "Câu hỏi Q0",
            config=self.config,
            query_generator_fn=self._mock_query_gen,
            hybrid_retriever_fn=mock_retriever,
        )
        hits = res["merged_child_hits"]
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0]["multi_query_rrf_score"], hits[1]["multi_query_rrf_score"])
        self.assertEqual(hits[0]["support_query_count"], hits[1]["support_query_count"])
        self.assertEqual(hits[0]["child_id"], "child_A")
        self.assertEqual(hits[1]["child_id"], "child_Z")

    def test_08_single_hybrid_call_per_query(self):
        """Mỗi query được gọi hybrid_retriever đúng 1 lần (tổng số lần gọi bằng số lượng query)."""
        call_counts: Dict[str, int] = {}

        def mock_retriever(qtext: str, cfg: Dict) -> List[Dict]:
            call_counts[qtext] = call_counts.get(qtext, 0) + 1
            return []

        retrieve_multi_query_children(
            "Câu hỏi Q0",
            config=self.config,
            query_generator_fn=self._mock_query_gen,
            hybrid_retriever_fn=mock_retriever,
        )
        self.assertEqual(sum(call_counts.values()), 3)  # Q0, Q1, Q2 mỗi câu 1 lần

    def test_09_no_reranker_or_generation_call(self):
        """Kiểm tra retrieval step chỉ dừng ở level child hits hợp nhất, không gọi reranker hay answer gen."""
        res = retrieve_multi_query_children(
            "Câu hỏi Q0",
            config=self.config,
            query_generator_fn=self._mock_query_gen,
            hybrid_retriever_fn=lambda q, c: [],
        )
        self.assertIn("merged_child_hits", res)
        self.assertNotIn("parent_candidates", res)
        self.assertNotIn("answer", res)

    def test_10_q0_failure_and_generated_query_partial_status(self):
        """Q0 retrieval lỗi -> Raise exception; Generated query lỗi -> Status multi_query_partial."""
        # 1. Q0 failure
        def q0_fail_retriever(qtext: str, cfg: Dict) -> List[Dict]:
            if "Câu hỏi Q0" in qtext:
                raise RuntimeError("Chroma DB error on Q0")
            return []

        with self.assertRaises(RuntimeError) as ctx:
            retrieve_multi_query_children(
                "Câu hỏi Q0",
                config=self.config,
                query_generator_fn=self._mock_query_gen,
                hybrid_retriever_fn=q0_fail_retriever,
            )
        self.assertIn("Q0 hybrid retrieval failed", str(ctx.exception))

        # 2. Generated query failure
        def var_fail_retriever(qtext: str, cfg: Dict) -> List[Dict]:
            if "Biến thể 1" in qtext:
                raise RuntimeError("Variant 1 failed")
            return [{"child_id": "c1", "text": "T1", "source": "src.pdf", "page_start": 1, "page_end": 1}]

        res = retrieve_multi_query_children(
            "Câu hỏi Q0",
            config=self.config,
            query_generator_fn=self._mock_query_gen,
            hybrid_retriever_fn=var_fail_retriever,
        )
        self.assertEqual(res["status"], "multi_query_partial")
        self.assertIn("Q1", res["trace"]["query_errors"])

    def test_11_trace_counts_and_latency_schema(self):
        """Trace trả về đúng schema chỉ số: query_count, latencies, overlap_distribution..."""
        res = retrieve_multi_query_children(
            "Câu hỏi Q0",
            config=self.config,
            query_generator_fn=self._mock_query_gen,
            hybrid_retriever_fn=lambda q, c: [{"child_id": "c1", "text": "T1", "source": "src.pdf", "page_start": 1, "page_end": 1}],
        )
        trace = res["trace"]
        self.assertIn("query_count", trace)
        self.assertIn("generation_latency_ms", trace)
        self.assertIn("per_query_retrieval_latency_ms", trace)
        self.assertIn("union_child_count", trace)
        self.assertIn("overlap_distribution", trace)
        self.assertEqual(trace["query_count"]["executed"], 3)

    def test_12_unit_test_100_percent_offline(self):
        """100% offline: không cần kết nối Chroma DB thật hay Gemini API thật."""
        res = retrieve_multi_query_children(
            "Câu hỏi Q0",
            config=self.config,
            query_generator_fn=self._mock_query_gen,
            hybrid_retriever_fn=lambda q, c: [{"child_id": "c_off", "text": "T_off", "source": "src.pdf", "page_start": 1, "page_end": 1}],
        )
        self.assertEqual(res["status"], "ready")
        self.assertEqual(len(res["merged_child_hits"]), 1)


if __name__ == "__main__":
    unittest.main()
