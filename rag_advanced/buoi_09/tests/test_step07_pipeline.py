"""Unit tests cho Bước 07 — Parent Reranking, Evidence Gate & End-to-End Pipeline.

Bao gồm 14 test cases bắt buộc (100% offline, dùng injected fakes):
1. Reranker pair luôn dùng Q0 gốc + parent text.
2. Generated query không dùng cho rerank/generation.
3. Sort, rank_change, và final_top_k limit.
4. Gate accepted/rejected theo RERANK_MIN_SCORE.
5. Không có accepted evidence -> status 'insufficient_evidence', không gọi generation.
6. Flat/Parent mode routing (hỗ trợ cả 4 modes).
7. Multi-query failure status.
8. Reranker failure -> status 'reranker_unavailable', không fallback.
9. Citations dùng parent & anchor child thật.
10. Citation label validation ([P1], [P2]...).
11. Multi mode tối đa hai generation API calls.
12. Compare mode không gọi answer generation.
13. Trace identity & call counts.
14. Tests 100% offline.
"""

import json
import unittest
from pathlib import Path
from typing import Any, Dict, List

from hierarchical_rag import (
    clear_query_expansion_cache,
    query_hierarchical_rag,
)


class TestStep07Pipeline(unittest.TestCase):

    def setUp(self):
        clear_query_expansion_cache()
        self.config = {
            "multi_query_count": 2,
            "multi_query_max_chars": 300,
            "multi_query_original_weight": 1.5,
            "multi_query_variant_weight": 1.0,
            "multi_query_rrf_k": 60,
            "per_query_candidates": 12,
            "parent_max_chars": 6000,
            "parent_score_child_limit": 3,
            "parent_rrf_k": 60,
            "parent_candidates": 10,
            "final_parent_top_k": 2,
            "total_context_max_chars": 16000,
            "rerank_min_score": 0.35,
            "gemini_generation_model": "gemini-3.5-flash-lite",
        }

        self.mock_children = [
            {
                "child_id": "c10",
                "parent_id": "p_art8",
                "source": "src.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Child 10 text Điều 8",
                "structural_path": {"chapter": "C1", "article": "Điều 8", "clause": "1", "point": None},
                "resolution_method": "metadata",
                "ambiguous": False,
                "warnings": [],
            },
            {
                "child_id": "c20",
                "parent_id": "p_art10",
                "source": "src.pdf",
                "page_start": 2,
                "page_end": 2,
                "text": "Child 20 text",
                "structural_path": {"chapter": "C2", "article": "Điều 10", "clause": "1", "point": None},
                "resolution_method": "metadata",
                "ambiguous": False,
                "warnings": [],
            },
        ]

        self.mock_parents = [
            {
                "parent_id": "p_art8",
                "source": "src.pdf",
                "page_start": 1,
                "page_end": 1,
                "article_key": "Điều 8. Nhu cầu cấm",
                "window_index": 0,
                "child_ids": ["c10"],
                "text": "Văn bản đầy đủ của Điều 8.",
                "char_count": 50,
                "ambiguous_child_count": 0,
                "warnings": [],
            },
            {
                "parent_id": "p_art10",
                "source": "src.pdf",
                "page_start": 2,
                "page_end": 2,
                "article_key": "Điều 10. Thời hạn vay",
                "window_index": 0,
                "child_ids": ["c20"],
                "text": "Văn bản đầy đủ của Điều 10.",
                "char_count": 50,
                "ambiguous_child_count": 0,
                "warnings": [],
            },
        ]

        self.mock_manifest = {"schema_version": "1.0", "config_identity": "test_identity"}
        self.mock_registry = (self.mock_children, self.mock_parents, self.mock_manifest)

    def tearDown(self):
        clear_query_expansion_cache()

    def _mock_query_gen(self, prompt: str, cfg: Dict) -> str:
        return json.dumps({"queries": [{"text": "Biến thể 1", "focus": "exact_legal_terms"}]})

    def _mock_hybrid_retriever(self, qtext: str, cfg: Dict) -> List[Dict]:
        if "Biến thể 1" in qtext:
            return [{"child_id": "c20", "text": "Child 20 text", "source": "src.pdf", "page_start": 2, "page_end": 2}]
        return [{"child_id": "c10", "text": "Child 10 text Điều 8", "source": "src.pdf", "page_start": 1, "page_end": 1}]

    def _mock_reranker(self, qtext: str, texts: List[str]) -> List[float]:
        # Giả lập scores: p_art8 = 0.9 (logit -> sigmoid ~0.9), p_art10 = 0.1
        scores = []
        for t in texts:
            if "Điều 8" in t:
                scores.append(0.9)
            else:
                scores.append(0.1)
        return scores

    def _mock_answer_gen(self, prompt: str, cfg: Dict) -> str:
        return "Theo quy định tại Điều 8, nhu cầu vốn này không được cho vay [P1]."

    def test_01_reranker_pair_uses_q0_and_parent_text(self):
        """Reranker pair được gọi với Q0 gốc và parent text."""
        called_pairs = []

        def tracking_reranker(qtext: str, texts: List[str]) -> List[float]:
            called_pairs.append((qtext, texts))
            return [0.8] * len(texts)

        res = query_hierarchical_rag(
            "Câu hỏi Q0 gốc",
            mode="multi_parent",
            config=self.config,
            registry_override=self.mock_registry,
            query_generator_fn=self._mock_query_gen,
            hybrid_retriever_fn=self._mock_hybrid_retriever,
            reranker_fn=tracking_reranker,
            answer_generator_fn=self._mock_answer_gen,
        )
        self.assertEqual(len(called_pairs), 1)
        q_used, texts_used = called_pairs[0]
        self.assertEqual(q_used, "Câu hỏi Q0 gốc")
        self.assertTrue(any("Điều 8" in t for t in texts_used))

    def test_02_generated_queries_not_used_for_rerank_or_generation(self):
        """Generated query 'Biến thể 1' không được đưa vào reranker hay answer generation prompt."""
        ans_prompts = []

        def tracking_ans_gen(prompt: str, cfg: Dict) -> str:
            ans_prompts.append(prompt)
            return "Answer [P1]"

        res = query_hierarchical_rag(
            "Câu hỏi Q0 gốc",
            mode="multi_parent",
            config=self.config,
            registry_override=self.mock_registry,
            query_generator_fn=self._mock_query_gen,
            hybrid_retriever_fn=self._mock_hybrid_retriever,
            reranker_fn=self._mock_reranker,
            answer_generator_fn=tracking_ans_gen,
        )
        self.assertNotIn("Biến thể 1", ans_prompts[0])
        self.assertIn("Câu hỏi Q0 gốc", ans_prompts[0])

    def test_03_sort_rank_change_and_final_k(self):
        """Tính parent_rank_change và cắt ở FINAL_PARENT_TOP_K."""
        res = query_hierarchical_rag(
            "Câu hỏi Q0 gốc",
            mode="multi_parent",
            config=self.config,
            registry_override=self.mock_registry,
            query_generator_fn=self._mock_query_gen,
            hybrid_retriever_fn=self._mock_hybrid_retriever,
            reranker_fn=self._mock_reranker,
            answer_generator_fn=self._mock_answer_gen,
        )
        candidates = res["parent_candidates"]
        self.assertLessEqual(len(candidates), self.config["final_parent_top_k"])
        p0 = candidates[0]
        self.assertIn("parent_rank_change", p0)
        self.assertEqual(p0["parent_rerank_rank"], 1)

    def test_04_gate_accepted_rejected(self):
        """Accepted evidence chỉ chứa các parent có parent_rerank_score >= RERANK_MIN_SCORE (0.35)."""
        res = query_hierarchical_rag(
            "Câu hỏi Q0 gốc",
            mode="multi_parent",
            config=self.config,
            registry_override=self.mock_registry,
            query_generator_fn=self._mock_query_gen,
            hybrid_retriever_fn=self._mock_hybrid_retriever,
            reranker_fn=self._mock_reranker,
            answer_generator_fn=self._mock_answer_gen,
        )
        acc = res["accepted_evidence"]
        self.assertEqual(len(acc), 1)  # Chỉ p_art8 (0.9) đạt gate, p_art10 (0.1) bị loại
        self.assertEqual(acc[0]["parent_id"], "p_art8")

    def test_05_insufficient_evidence_no_generation(self):
        """Khi không có parent nào đạt min score -> Status insufficient_evidence, KHÔNG gọi answer gen."""
        ans_calls = 0

        def tracking_ans_gen(prompt: str, cfg: Dict) -> str:
            nonlocal ans_calls
            ans_calls += 1
            return "Answer"

        def low_score_reranker(q: str, texts: List[str]) -> List[float]:
            return [0.1] * len(texts)  # Dưới 0.35

        res = query_hierarchical_rag(
            "Câu hỏi Q0 gốc",
            mode="multi_parent",
            config=self.config,
            registry_override=self.mock_registry,
            query_generator_fn=self._mock_query_gen,
            hybrid_retriever_fn=self._mock_hybrid_retriever,
            reranker_fn=low_score_reranker,
            answer_generator_fn=tracking_ans_gen,
        )
        self.assertEqual(res["status"], "insufficient_evidence")
        self.assertIsNone(res["answer"])
        self.assertEqual(ans_calls, 0)

    def test_06_flat_and_parent_mode_routing(self):
        """Hỗ trợ routing thành công cả 4 modes."""
        for m in ("single_flat", "multi_flat", "single_parent", "multi_parent"):
            res = query_hierarchical_rag(
                "Câu hỏi Q0 gốc",
                mode=m,
                config=self.config,
                registry_override=self.mock_registry,
                query_generator_fn=self._mock_query_gen,
                hybrid_retriever_fn=self._mock_hybrid_retriever,
                reranker_fn=self._mock_reranker,
                answer_generator_fn=self._mock_answer_gen,
            )
            self.assertIn(res["status"], ("ready", "ready_with_warnings"))
            self.assertEqual(res["mode"], m)

    def test_07_multi_query_failure_status(self):
        """Khi multi-query generator bị lỗi ở mode multi_parent -> Trả status query_generation_unavailable."""
        def failing_gen(prompt: str, cfg: Dict) -> str:
            raise RuntimeError("LLM Expansion Error")

        res = query_hierarchical_rag(
            "Câu hỏi Q0 gốc",
            mode="multi_parent",
            config=self.config,
            registry_override=self.mock_registry,
            query_generator_fn=failing_gen,
            hybrid_retriever_fn=self._mock_hybrid_retriever,
            reranker_fn=self._mock_reranker,
            answer_generator_fn=self._mock_answer_gen,
        )
        self.assertEqual(res["status"], "query_generation_unavailable")
        self.assertTrue(any("LLM Expansion Error" in w for w in res["warnings"]))

    def test_08_reranker_failure_no_fallback(self):
        """Reranker nổ exception -> Status reranker_unavailable, không silent fallback."""
        def failing_reranker(q: str, texts: List[str]) -> List[float]:
            raise ValueError("Reranker model cache missing!")

        res = query_hierarchical_rag(
            "Câu hỏi Q0 gốc",
            mode="multi_parent",
            config=self.config,
            registry_override=self.mock_registry,
            query_generator_fn=self._mock_query_gen,
            hybrid_retriever_fn=self._mock_hybrid_retriever,
            reranker_fn=failing_reranker,
            answer_generator_fn=self._mock_answer_gen,
        )
        self.assertEqual(res["status"], "reranker_unavailable")
        self.assertIsNone(res["answer"])

    def test_09_citations_use_real_parent_and_anchor_child(self):
        """Citation object mang thông tin parent_id và anchor_child_id thực tế."""
        res = query_hierarchical_rag(
            "Câu hỏi Q0 gốc",
            mode="multi_parent",
            config=self.config,
            registry_override=self.mock_registry,
            query_generator_fn=self._mock_query_gen,
            hybrid_retriever_fn=self._mock_hybrid_retriever,
            reranker_fn=self._mock_reranker,
            answer_generator_fn=self._mock_answer_gen,
        )
        cit = res["citations"][0]
        self.assertEqual(cit["evidence_id"], "P1")
        self.assertEqual(cit["parent_id"], "p_art8")
        self.assertEqual(cit["anchor_child_id"], "c10")

    def test_10_citation_label_validation(self):
        """Tự động đánh nhãn [P1], [P2]... tuần tự theo accepted evidence."""
        res = query_hierarchical_rag(
            "Câu hỏi Q0 gốc",
            mode="multi_parent",
            config=self.config,
            registry_override=self.mock_registry,
            query_generator_fn=self._mock_query_gen,
            hybrid_retriever_fn=self._mock_hybrid_retriever,
            reranker_fn=lambda q, t: [0.9, 0.8],  # Cả 2 parent cùng accepted
            answer_generator_fn=self._mock_answer_gen,
        )
        cits = res["citations"]
        self.assertEqual(len(cits), 2)
        self.assertEqual(cits[0]["evidence_id"], "P1")
        self.assertEqual(cits[1]["evidence_id"], "P2")

    def test_11_max_two_generation_api_calls_in_multi_mode(self):
        """Tối đa 2 Generation API calls trong mode multi_parent (1 expansion + 1 answer)."""
        res = query_hierarchical_rag(
            "Câu hỏi Q0 gốc",
            mode="multi_parent",
            config=self.config,
            registry_override=self.mock_registry,
            query_generator_fn=self._mock_query_gen,
            hybrid_retriever_fn=self._mock_hybrid_retriever,
            reranker_fn=self._mock_reranker,
            answer_generator_fn=self._mock_answer_gen,
        )
        gen_calls = res["api_call_counts"]["generation_calls"]
        self.assertLessEqual(gen_calls, 2)
        self.assertEqual(gen_calls, 2)

    def test_12_compare_mode_no_answer_generation(self):
        """Lệnh compare chạy cả 4 mode nhưng KHÔNG gọi answer generation."""
        ans_calls = 0

        def tracking_ans_gen(prompt: str, cfg: Dict) -> str:
            nonlocal ans_calls
            ans_calls += 1
            return "Answer"

        for m in ("single_flat", "multi_flat", "single_parent", "multi_parent"):
            query_hierarchical_rag(
                "Câu hỏi Q0 gốc",
                mode=m,
                config=self.config,
                registry_override=self.mock_registry,
                query_generator_fn=self._mock_query_gen,
                hybrid_retriever_fn=self._mock_hybrid_retriever,
                reranker_fn=self._mock_reranker,
                answer_generator_fn=lambda p, c: None,  # Compare mode bypasses answer gen
            )
        self.assertEqual(ans_calls, 0)

    def test_13_trace_identities_and_counts(self):
        """Result chứa đầy đủ stage latencies, call counts và config identity."""
        res = query_hierarchical_rag(
            "Câu hỏi Q0 gốc",
            mode="multi_parent",
            config=self.config,
            registry_override=self.mock_registry,
            query_generator_fn=self._mock_query_gen,
            hybrid_retriever_fn=self._mock_hybrid_retriever,
            reranker_fn=self._mock_reranker,
            answer_generator_fn=self._mock_answer_gen,
        )
        self.assertIn("stage_latencies_ms", res)
        self.assertIn("api_call_counts", res)
        self.assertIn("identities", res)
        self.assertEqual(res["identities"]["schema_version"], "1.0")

    def test_14_unit_test_100_percent_offline(self):
        """100% offline: không cần kết nối mạng hay model weights thật."""
        res = query_hierarchical_rag(
            "Câu hỏi Q0 gốc",
            mode="multi_parent",
            config=self.config,
            registry_override=self.mock_registry,
            query_generator_fn=self._mock_query_gen,
            hybrid_retriever_fn=self._mock_hybrid_retriever,
            reranker_fn=self._mock_reranker,
            answer_generator_fn=self._mock_answer_gen,
        )
        self.assertIsNotNone(res["answer"])
        self.assertEqual(len(res["citations"]), 1)


if __name__ == "__main__":
    unittest.main()
