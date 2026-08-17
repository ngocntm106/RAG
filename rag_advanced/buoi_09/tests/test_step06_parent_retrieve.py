"""Unit tests cho Bước 06 — Parent Document Resolution & Score Aggregation.

Bao gồm 12 test cases bắt buộc (100% offline, dùng registry mock):
1. Child map đúng parent_id.
2. Missing / stale hierarchy status handling.
3. Parent aggregation formula tính tay.
4. Child score cap ở PARENT_SCORE_CHILD_LIMIT.
5. Supporting, scoring, và anchor child separation.
6. Parent deduplication (nhiều child cùng parent).
7. Sort / tie-break deterministic.
8. Candidate limit truncation.
9. Context budget truncation chỉ ở parent boundary.
10. Oversized first parent warning.
11. Expansion factor & count trace accuracy.
12. Không gọi reranker/generation.
"""

import json
import unittest
from pathlib import Path
from typing import Any, Dict, List

from hierarchical_rag import (
    clear_query_expansion_cache,
    retrieve_parent_documents,
)


class TestStep06ParentRetrieve(unittest.TestCase):

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
            "parent_score_child_limit": 2,  # Cap at 2 for testing
            "parent_rrf_k": 60,
            "parent_candidates": 10,
            "final_parent_top_k": 3,
            "total_context_max_chars": 500,  # Small budget for testing truncation
            "gemini_generation_model": "gemini-3.5-flash-lite",
        }

        # Mock Children & Parents Registry
        self.mock_children = [
            {
                "child_id": "c10",
                "parent_id": "p_art8",
                "source": "src.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Child 10 text",
                "structural_path": {"chapter": "C1", "article": "Điều 8", "clause": "1", "point": None},
                "resolution_method": "metadata",
                "ambiguous": False,
                "warnings": [],
            },
            {
                "child_id": "c11",
                "parent_id": "p_art8",
                "source": "src.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Child 11 text",
                "structural_path": {"chapter": "C1", "article": "Điều 8", "clause": "2", "point": None},
                "resolution_method": "metadata",
                "ambiguous": False,
                "warnings": [],
            },
            {
                "child_id": "c12",
                "parent_id": "p_art8",
                "source": "src.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Child 12 text",
                "structural_path": {"chapter": "C1", "article": "Điều 8", "clause": "3", "point": None},
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
                "child_ids": ["c10", "c11", "c12"],
                "text": "Văn bản đầy đủ của Điều 8 (Child 10 text\nChild 11 text\nChild 12 text).",
                "char_count": 70,
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
                "text": "Văn bản đầy đủ của Điều 10 (Child 20 text).",
                "char_count": 45,
                "ambiguous_child_count": 0,
                "warnings": [],
            },
        ]

        self.mock_manifest = {
            "schema_version": "1.0",
            "config_identity": "test_identity",
        }

        self.mock_registry = (self.mock_children, self.mock_parents, self.mock_manifest)

    def tearDown(self):
        clear_query_expansion_cache()

    def _mock_gen(self, prompt: str, cfg: Dict) -> str:
        return json.dumps({"queries": [{"text": "Biến thể 1", "focus": "exact_legal_terms"}]})

    def test_01_child_maps_correct_parent(self):
        """Child hit được map chính xác vào parent_id trong registry."""
        def mock_retriever(qtext: str, cfg: Dict) -> List[Dict]:
            return [{"child_id": "c10", "text": "Child 10 text", "source": "src.pdf", "page_start": 1, "page_end": 1}]

        res = retrieve_parent_documents(
            "Câu hỏi Q0",
            mode="multi_parent",
            config=self.config,
            registry_override=self.mock_registry,
            query_generator_fn=self._mock_gen,
            hybrid_retriever_fn=mock_retriever,
        )
        self.assertEqual(res["status"], "ready")
        self.assertEqual(len(res["accepted_parents"]), 1)
        p = res["accepted_parents"][0]
        self.assertEqual(p["parent_id"], "p_art8")
        self.assertEqual(p["anchor_child_id"], "c10")

    def test_02_missing_or_stale_hierarchy_status(self):
        """Trả status hierarchy_not_ready khi registry không tồn tại."""
        res = retrieve_parent_documents(
            "Câu hỏi Q0",
            mode="multi_parent",
            config=self.config,
            storage_dir=Path("/non_existent_path_xyz"),
            query_generator_fn=self._mock_gen,
            hybrid_retriever_fn=lambda q, c: [],
        )
        self.assertEqual(res["status"], "hierarchy_not_ready")

    def test_03_parent_aggregation_formula_manual(self):
        """Tính tay parent_rrf_score: sum(1 / (60 + multi_query_rank)).

        Với c10 ở rank 1: 1 / (60 + 1) = 1/61 = 0.016393
        """
        def mock_retriever(qtext: str, cfg: Dict) -> List[Dict]:
            return [{"child_id": "c10", "text": "Child 10 text", "source": "src.pdf", "page_start": 1, "page_end": 1}]

        res = retrieve_parent_documents(
            "Câu hỏi Q0",
            mode="multi_parent",
            config=self.config,
            registry_override=self.mock_registry,
            query_generator_fn=self._mock_gen,
            hybrid_retriever_fn=mock_retriever,
        )
        p = res["accepted_parents"][0]
        self.assertAlmostEqual(p["parent_rrf_score"], round(1.0 / 61, 6), places=5)

    def test_04_child_score_cap(self):
        """Cap số child dùng tính score ở tối đa PARENT_SCORE_CHILD_LIMIT (ở config này là 2)."""
        def mock_retriever(qtext: str, cfg: Dict) -> List[Dict]:
            # Tra ve 3 child c10 (rank 1), c11 (rank 2), c12 (rank 3)
            return [
                {"child_id": "c10", "text": "Child 10 text", "source": "src.pdf", "page_start": 1, "page_end": 1},
                {"child_id": "c11", "text": "Child 11 text", "source": "src.pdf", "page_start": 1, "page_end": 1},
                {"child_id": "c12", "text": "Child 12 text", "source": "src.pdf", "page_start": 1, "page_end": 1},
            ]

        res = retrieve_parent_documents(
            "Câu hỏi Q0",
            mode="multi_parent",
            config=self.config,
            registry_override=self.mock_registry,
            query_generator_fn=self._mock_gen,
            hybrid_retriever_fn=mock_retriever,
        )
        p = res["accepted_parents"][0]
        self.assertEqual(len(p["scoring_child_ids"]), 2)  # Cap at 2
        self.assertEqual(len(p["supporting_child_ids"]), 3)  # Supporting keeps all 3
        expected_score = round(1.0 / 61 + 1.0 / 62, 6)
        self.assertAlmostEqual(p["parent_rrf_score"], expected_score, places=5)

    def test_05_supporting_and_scoring_child_separation(self):
        """Tách bạch scoring_child_ids, supporting_child_ids, và anchor_child_id."""
        def mock_retriever(qtext: str, cfg: Dict) -> List[Dict]:
            return [
                {"child_id": "c10", "text": "Child 10 text", "source": "src.pdf", "page_start": 1, "page_end": 1},
                {"child_id": "c11", "text": "Child 11 text", "source": "src.pdf", "page_start": 1, "page_end": 1},
                {"child_id": "c12", "text": "Child 12 text", "source": "src.pdf", "page_start": 1, "page_end": 1},
            ]

        res = retrieve_parent_documents(
            "Câu hỏi Q0",
            mode="multi_parent",
            config=self.config,
            registry_override=self.mock_registry,
            query_generator_fn=self._mock_gen,
            hybrid_retriever_fn=mock_retriever,
        )
        p = res["accepted_parents"][0]
        self.assertEqual(p["anchor_child_id"], "c10")
        self.assertEqual(p["scoring_child_ids"], ["c10", "c11"])
        self.assertEqual(p["supporting_child_ids"], ["c10", "c11", "c12"])

    def test_06_parent_deduplication(self):
        """Nhiều child thuộc cùng 1 parent (c10 và c11) gom vào 1 parent candidate duy nhất."""
        def mock_retriever(qtext: str, cfg: Dict) -> List[Dict]:
            return [
                {"child_id": "c10", "text": "Child 10 text", "source": "src.pdf", "page_start": 1, "page_end": 1},
                {"child_id": "c11", "text": "Child 11 text", "source": "src.pdf", "page_start": 1, "page_end": 1},
            ]

        res = retrieve_parent_documents(
            "Câu hỏi Q0",
            mode="multi_parent",
            config=self.config,
            registry_override=self.mock_registry,
            query_generator_fn=self._mock_gen,
            hybrid_retriever_fn=mock_retriever,
        )
        self.assertEqual(len(res["accepted_parents"]), 1)

    def test_07_deterministic_parent_sort(self):
        """Parent được sắp xếp deterministic theo score desc."""
        def mock_retriever(qtext: str, cfg: Dict) -> List[Dict]:
            if "Biến thể 1" in qtext:
                return [{"child_id": "c20", "text": "Child 20 text", "source": "src.pdf", "page_start": 2, "page_end": 2}]
            if "Câu hỏi Q0" in qtext:
                return [{"child_id": "c10", "text": "Child 10 text", "source": "src.pdf", "page_start": 1, "page_end": 1}]
            return []

        res = retrieve_parent_documents(
            "Câu hỏi Q0",
            mode="multi_parent",
            config=self.config,
            registry_override=self.mock_registry,
            query_generator_fn=self._mock_gen,
            hybrid_retriever_fn=mock_retriever,
        )
        parents = res["accepted_parents"]
        self.assertEqual(len(parents), 2)
        # p_art8 có c10 xuất hiện ở Q0 (w=1.5) -> score cao hơn p_art10 có c20 ở Q1 (w=1.0)
        self.assertEqual(parents[0]["parent_id"], "p_art8")
        self.assertEqual(parents[1]["parent_id"], "p_art10")

    def test_08_candidate_limit_truncation(self):
        """Giới hạn PARENT_CANDIDATES items (ở đây đặt limit 1 để test)."""
        config = dict(self.config)
        config["parent_candidates"] = 1

        def mock_retriever(qtext: str, cfg: Dict) -> List[Dict]:
            return [
                {"child_id": "c10", "text": "Child 10 text", "source": "src.pdf", "page_start": 1, "page_end": 1},
                {"child_id": "c20", "text": "Child 20 text", "source": "src.pdf", "page_start": 2, "page_end": 2},
            ]

        res = retrieve_parent_documents(
            "Câu hỏi Q0",
            mode="multi_parent",
            config=config,
            registry_override=self.mock_registry,
            query_generator_fn=self._mock_gen,
            hybrid_retriever_fn=mock_retriever,
        )
        self.assertEqual(len(res["accepted_parents"]), 1)
        self.assertEqual(len(res["dropped_parents"]), 1)
        self.assertEqual(res["trace"]["parents_dropped"]["by_candidate_limit"], 1)

    def test_09_context_budget_truncation_at_boundary(self):
        """Truncate ở ranh giới nguyên parent document khi tổng độ dài vượt total_context_max_chars."""
        config = dict(self.config)
        config["total_context_max_chars"] = 80  # Chỉ vừa p_art8 (70 chars), p_art10 (45 chars) sẽ bị dropped

        def mock_retriever(qtext: str, cfg: Dict) -> List[Dict]:
            return [
                {"child_id": "c10", "text": "Child 10 text", "source": "src.pdf", "page_start": 1, "page_end": 1},
                {"child_id": "c20", "text": "Child 20 text", "source": "src.pdf", "page_start": 2, "page_end": 2},
            ]

        res = retrieve_parent_documents(
            "Câu hỏi Q0",
            mode="multi_parent",
            config=config,
            registry_override=self.mock_registry,
            query_generator_fn=self._mock_gen,
            hybrid_retriever_fn=mock_retriever,
        )
        self.assertEqual(len(res["accepted_parents"]), 1)
        self.assertEqual(res["accepted_parents"][0]["parent_id"], "p_art8")
        self.assertEqual(len(res["dropped_parents"]), 1)

    def test_10_oversized_first_parent_warning(self):
        """Giữ parent đầu tiên nếu vượt total_context_max_chars và đánh warning oversized_first_parent_kept_exceeding_budget."""
        config = dict(self.config)
        config["total_context_max_chars"] = 30  # Nhỏ hơn p_art8 (70 chars)

        def mock_retriever(qtext: str, cfg: Dict) -> List[Dict]:
            return [{"child_id": "c10", "text": "Child 10 text", "source": "src.pdf", "page_start": 1, "page_end": 1}]

        res = retrieve_parent_documents(
            "Câu hỏi Q0",
            mode="multi_parent",
            config=config,
            registry_override=self.mock_registry,
            query_generator_fn=self._mock_gen,
            hybrid_retriever_fn=mock_retriever,
        )
        self.assertEqual(len(res["accepted_parents"]), 1)
        p = res["accepted_parents"][0]
        self.assertTrue(any("oversized_first_parent_kept_exceeding_budget" in w for w in p["warnings"]))

    def test_11_expansion_factor_and_count_trace(self):
        """Tính đúng child_chars, expanded_parent_chars, context_expansion_factor trong trace."""
        def mock_retriever(qtext: str, cfg: Dict) -> List[Dict]:
            return [{"child_id": "c10", "text": "1234567890", "source": "src.pdf", "page_start": 1, "page_end": 1}]  # 10 chars

        res = retrieve_parent_documents(
            "Câu hỏi Q0",
            mode="multi_parent",
            config=self.config,
            registry_override=self.mock_registry,
            query_generator_fn=self._mock_gen,
            hybrid_retriever_fn=mock_retriever,
        )
        trace = res["trace"]
        self.assertEqual(trace["child_chars"], 10)
        self.assertEqual(trace["expanded_parent_chars"], 70)  # p_art8 text length = 70
        self.assertEqual(trace["context_expansion_factor"], 7.0)

    def test_12_no_reranker_or_generation_call(self):
        """Kiểm tra parent retrieval dừng ở level parent documents, không đụng tới cross-encoder hay answer gen."""
        res = retrieve_parent_documents(
            "Câu hỏi Q0",
            mode="multi_parent",
            config=self.config,
            registry_override=self.mock_registry,
            query_generator_fn=self._mock_gen,
            hybrid_retriever_fn=lambda q, c: [],
        )
        self.assertIn("accepted_parents", res)
        self.assertNotIn("reranked_parents", res)
        self.assertNotIn("answer", res)


if __name__ == "__main__":
    unittest.main()
