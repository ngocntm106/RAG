"""Unit tests cho Bước 08 — UI Helper Functions.

Bao gồm 6 test cases bắt buộc (100% offline, thuần Python, không đụng Streamlit hay API call):
1. Build query-child matrix structure.
2. Format parent tree node data.
3. Format citation display list.
4. Build mode comparison row.
5. Map status badge types and descriptions.
6. Context expansion factor computation.
"""

import unittest
from ui_helpers import (
    build_mode_comparison_row,
    build_query_child_matrix,
    format_citation_display,
    format_parent_tree_node,
    map_status_badge,
)


class TestStep08UIHelpers(unittest.TestCase):

    def test_01_build_query_child_matrix(self):
        """Ma trận Query–Child trả về đúng danh sách qids và rows với rank tương ứng."""
        queries = [
            {"query_id": "Q0", "text": "Q0 text"},
            {"query_id": "Q1", "text": "Q1 text"},
        ]
        hits = [
            {
                "child_id": "c1",
                "source": "src1.pdf",
                "support_query_count": 2,
                "multi_query_rrf_score": 0.04,
                "per_query_ranks": {"Q0": 1, "Q1": 3},
            },
            {
                "child_id": "c2",
                "source": "src1.pdf",
                "support_query_count": 1,
                "multi_query_rrf_score": 0.02,
                "per_query_ranks": {"Q0": 2},
            },
        ]

        matrix = build_query_child_matrix(queries, hits)
        self.assertEqual(matrix["qids"], ["Q0", "Q1"])
        self.assertEqual(len(matrix["rows"]), 2)

        row0 = matrix["rows"][0]
        self.assertEqual(row0["child_id"], "c1")
        self.assertEqual(row0["ranks"]["Q0"], 1)
        self.assertEqual(row0["ranks"]["Q1"], 3)

        row1 = matrix["rows"][1]
        self.assertEqual(row1["child_id"], "c2")
        self.assertEqual(row1["ranks"]["Q0"], 2)
        self.assertIsNone(row1["ranks"]["Q1"])

    def test_02_format_parent_tree_node(self):
        """Format node cây Parent–Child hiển thị đầy đủ thông tin rank, score và list child."""
        parent = {
            "parent_id": "p1",
            "source": "src.pdf",
            "page_start": 1,
            "page_end": 2,
            "parent_rank": 3,
            "parent_rerank_rank": 1,
            "parent_rank_change": 2,
            "parent_rrf_score": 0.03,
            "parent_rerank_score": 0.85,
            "structural_path": {"article": "Điều 8. Nhu cầu cấm", "chapter": "Chương II"},
            "anchor_child_id": "c10",
            "scoring_child_ids": ["c10"],
            "supporting_child_ids": ["c10", "c11"],
            "support_query_ids": ["Q0", "Q1"],
            "text": "Parent text content here...",
            "ambiguous": False,
            "warnings": [],
        }

        node = format_parent_tree_node(parent)
        self.assertEqual(node["parent_id"], "p1")
        self.assertIn("Điều 8", node["article"])
        self.assertIn("+2", node["rank_summary"])
        self.assertEqual(node["anchor_child_id"], "c10")
        self.assertEqual(node["char_count"], len("Parent text content here..."))

    def test_03_format_citation_display(self):
        """Format danh sách citation đính kèm label [P1], [P2]."""
        cits = [
            {
                "evidence_id": "P1",
                "source": "src.pdf",
                "page_start": 1,
                "page_end": 3,
                "structural_path": {"article": "Điều 8. Nhu cầu cấm"},
                "parent_rerank_score": 0.8542,
                "parent_id": "p1",
                "anchor_child_id": "c10",
                "ambiguous": False,
            }
        ]

        formatted = format_citation_display(cits)
        self.assertEqual(len(formatted), 1)
        item = formatted[0]
        self.assertEqual(item["label"], "[P1]")
        self.assertIn("Điều 8", item["title"])
        self.assertEqual(item["score"], "0.8542")

    def test_04_build_mode_comparison_row(self):
        """Tạo hàng dữ liệu so sánh 4 modes trong Tab 4."""
        mock_res = {
            "status": "ready",
            "accepted_evidence": [
                {"evidence_id": "P1", "parent_rerank_score": 0.88, "text": "Parent 1 text"},
                {"evidence_id": "P2", "parent_rerank_score": 0.75, "text": "Parent 2 text"},
            ],
            "stage_latencies_ms": {"retrieval": 15.0, "rerank": 20.0},
            "api_call_counts": {"generation_calls": 2, "embedding_calls": 0},
            "trace": {
                "union_child_count": 12,
                "unique_parent_count": 5,
                "expanded_parent_chars": 8000,
                "context_expansion_factor": 4.5,
            },
            "warnings": [],
        }

        row = build_mode_comparison_row("multi_parent", mock_res)
        self.assertEqual(row["mode"], "multi_parent")
        self.assertEqual(row["unit_type"], "parent")
        self.assertEqual(row["evidence_count"], 2)
        self.assertEqual(row["evidence_ids"], "P1, P2")
        self.assertEqual(row["top_rerank_score"], 0.88)
        self.assertEqual(row["expansion_factor"], 4.5)
        self.assertEqual(row["total_latency_ms"], 35.0)

    def test_05_map_status_badge(self):
        """Map các status code thành UI badge type và description."""
        badge_ready = map_status_badge("ready")
        self.assertEqual(badge_ready[0], "success")

        badge_missing = map_status_badge("hierarchy_not_ready")
        self.assertEqual(badge_missing[0], "error")
        self.assertIn("Hierarchy Store Chưa Sẵn Sàng", badge_missing[1])

        badge_gate = map_status_badge("insufficient_evidence")
        self.assertEqual(badge_gate[0], "warning")
        self.assertIn("Không Đủ Căn Cứ Đạt Gate", badge_gate[1])

    def test_06_expansion_factor_calculation(self):
        """Context expansion factor được tính chính xác."""
        mock_res = {
            "accepted_evidence": [{"text": "1234567890"}],
            "trace": {"expanded_parent_chars": 100, "context_expansion_factor": 10.0},
        }
        row = build_mode_comparison_row("single_parent", mock_res)
        self.assertEqual(row["expansion_factor"], 10.0)


if __name__ == "__main__":
    unittest.main()
