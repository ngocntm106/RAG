"""Unit tests cho Bước 04 — Multi-Query Expansion.

Bao gồm 11 test cases bắt buộc (100% offline, dùng fake query_generator_fn):
1. Q0 luôn đứng đầu và giữ nguyên nội dung.
2. Strict schema validation (đủ tất cả trường theo hợp đồng).
3. NFC / trim / max length check.
4. Duplicate removal (trùng lặp với Q0 hoặc giữa các Qi).
5. Legal reference preservation check.
6. Reject invented articles (loại bỏ số Điều bịa thêm).
7. Deterministic query IDs (Q0, Q1, Q2...).
8. Single generator call (chỉ 1 API call).
9. Cache hit không gọi lại generator.
10. API lỗi trả explicit status 'query_generation_unavailable'.
11. Unit test 100% offline (mock generator).
"""

import json
import unittest
from typing import Any, Dict

from hierarchical_rag import (
    clear_query_expansion_cache,
    generate_multi_queries,
)


class TestStep04MultiQuery(unittest.TestCase):

    def setUp(self):
        clear_query_expansion_cache()
        self.config = {
            "multi_query_count": 3,
            "multi_query_max_chars": 300,
            "multi_query_temperature": 0.2,
            "gemini_generation_model": "gemini-3.5-flash-lite",
        }

    def tearDown(self):
        clear_query_expansion_cache()

    def test_01_q0_first_and_preserved(self):
        """Q0 luôn đứng đầu danh sách, origin='original', focus='original_intent', giữ nguyên nội dung."""
        q0_text = "  Những nhu cầu vốn nào không được cho vay?  "

        def fake_gen(prompt: str, cfg: Dict) -> str:
            return json.dumps({
                "queries": [
                    {"text": "Quy định về nhu cầu vốn bị cấm cho vay", "focus": "exact_legal_terms"}
                ]
            })

        res = generate_multi_queries(q0_text, config=self.config, query_generator_fn=fake_gen)
        self.assertEqual(res["status"], "ready")
        self.assertEqual(res["original_question"], "Những nhu cầu vốn nào không được cho vay?")
        self.assertEqual(len(res["queries"]), 2)
        q0 = res["queries"][0]
        self.assertEqual(q0["query_id"], "Q0")
        self.assertEqual(q0["origin"], "original")
        self.assertEqual(q0["focus"], "original_intent")
        self.assertEqual(q0["text"], "Những nhu cầu vốn nào không được cho vay?")

    def test_02_strict_schema_validation(self):
        """Trả về đầy đủ các trường dữ liệu theo đúng Query Set Contract."""
        def fake_gen(prompt: str, cfg: Dict) -> str:
            return json.dumps({"queries": []})

        res = generate_multi_queries("Test question", config=self.config, query_generator_fn=fake_gen)
        expected_keys = {
            "original_question",
            "queries",
            "model",
            "generation_latency_ms",
            "status",
            "warnings",
            "cache_hit",
            "dropped_duplicate_count",
        }
        self.assertTrue(expected_keys.issubset(res.keys()))

    def test_03_nfc_trim_max_length(self):
        """Chuẩn hóa Unicode NFC, trim khoảng trắng và loại bỏ query sinh ra dài hơn MULTI_QUERY_MAX_CHARS."""
        config = dict(self.config)
        config["multi_query_max_chars"] = 20

        def fake_gen(prompt: str, cfg: Dict) -> str:
            return json.dumps({
                "queries": [
                    {"text": "  Câu ngắn  ", "focus": "paraphrase"},
                    {"text": "Câu này quá dài vượt quá giới hạn hai mươi ký tự cho phép", "focus": "paraphrase"},
                ]
            })

        res = generate_multi_queries("Câu hỏi gốc?", config=config, query_generator_fn=fake_gen)
        queries = res["queries"]
        self.assertEqual(len(queries), 2)  # Q0 + "Câu ngắn"
        self.assertEqual(queries[1]["text"], "Câu ngắn")

    def test_04_duplicate_removal(self):
        """Loại bỏ các query sinh ra bị trùng lặp với Q0 hoặc trùng lặp lẫn nhau."""
        def fake_gen(prompt: str, cfg: Dict) -> str:
            return json.dumps({
                "queries": [
                    {"text": "Nhu cầu vốn cấm?", "focus": "paraphrase"},  # Trùng Q0
                    {"text": "Biến thể 1", "focus": "exact_legal_terms"},
                    {"text": "Biến thể 1", "focus": "missing_aspect"},   # Trùng Biến thể 1
                ]
            })

        res = generate_multi_queries("Nhu cầu vốn cấm?", config=self.config, query_generator_fn=fake_gen)
        self.assertEqual(res["dropped_duplicate_count"], 2)
        self.assertEqual(len(res["queries"]), 2)  # Q0 + Biến thể 1

    def test_05_legal_reference_preservation(self):
        """Giữ nguyên số Điều có trong Q0."""
        q0 = "Theo Điều 8 Thông tư 39, nhu cầu vốn nào bị cấm?"

        def fake_gen(prompt: str, cfg: Dict) -> str:
            return json.dumps({
                "queries": [
                    {"text": "Quy định tại Điều 8 về nhu cầu vốn cấm", "focus": "exact_legal_terms"},
                ]
            })

        res = generate_multi_queries(q0, config=self.config, query_generator_fn=fake_gen)
        self.assertEqual(len(res["queries"]), 2)
        self.assertIn("Điều 8", res["queries"][1]["text"])

    def test_06_reject_invented_articles(self):
        """Loại bỏ variant chứa số Điều phát minh không xuất hiện trong Q0."""
        q0 = "Nhu cầu vốn cấm theo Điều 8?"

        def fake_gen(prompt: str, cfg: Dict) -> str:
            return json.dumps({
                "queries": [
                    {"text": "Thời hạn cho vay theo Điều 10?", "focus": "exact_legal_terms"},  # Điều 10 bịa thêm
                    {"text": "Các nhu cầu vốn cấm vay theo Điều 8", "focus": "paraphrase"},
                ]
            })

        res = generate_multi_queries(q0, config=self.config, query_generator_fn=fake_gen)
        self.assertEqual(len(res["queries"]), 2)  # Q0 + Điều 8 variant
        self.assertEqual(res["queries"][1]["text"], "Các nhu cầu vốn cấm vay theo Điều 8")
        self.assertTrue(any("invented article" in w for w in res["warnings"]))

    def test_07_deterministic_ids(self):
        """Đánh số query_id theo thứ tự deterministic Q0, Q1, Q2..."""
        def fake_gen(prompt: str, cfg: Dict) -> str:
            return json.dumps({
                "queries": [
                    {"text": "Biến thể A", "focus": "exact_legal_terms"},
                    {"text": "Biến thể B", "focus": "paraphrase"},
                ]
            })

        res = generate_multi_queries("Câu hỏi", config=self.config, query_generator_fn=fake_gen)
        ids = [q["query_id"] for q in res["queries"]]
        self.assertEqual(ids, ["Q0", "Q1", "Q2"])

    def test_08_single_generator_call(self):
        """Đảm bảo chỉ gọi query_generator_fn đúng 1 lần duy nhất."""
        call_count = 0

        def fake_gen(prompt: str, cfg: Dict) -> str:
            nonlocal call_count
            call_count += 1
            return json.dumps({"queries": [{"text": "Variant 1", "focus": "paraphrase"}]})

        generate_multi_queries("Test single call", config=self.config, query_generator_fn=fake_gen)
        self.assertEqual(call_count, 1)

    def test_09_cache_hit_no_second_call(self):
        """Cache hit trong process: gọi lại cùng câu hỏi không thực hiện generator call thứ hai."""
        call_count = 0

        def fake_gen(prompt: str, cfg: Dict) -> str:
            nonlocal call_count
            call_count += 1
            return json.dumps({"queries": [{"text": "Variant 1", "focus": "paraphrase"}]})

        res1 = generate_multi_queries("Cache question", config=self.config, query_generator_fn=fake_gen)
        self.assertFalse(res1["cache_hit"])
        self.assertEqual(call_count, 1)

        res2 = generate_multi_queries("Cache question", config=self.config, query_generator_fn=fake_gen)
        self.assertTrue(res2["cache_hit"])
        self.assertEqual(call_count, 1)  # Không tăng lên 2
        self.assertEqual(res1["queries"], res2["queries"])

    def test_10_explicit_status_on_api_error(self):
        """Khi API/Generator gặp lỗi, trả explicit status 'query_generation_unavailable' và chỉ chứa Q0."""
        def failing_gen(prompt: str, cfg: Dict) -> str:
            raise RuntimeError("API Connection timeout")

        res = generate_multi_queries("Lỗi API?", config=self.config, query_generator_fn=failing_gen)
        self.assertEqual(res["status"], "query_generation_unavailable")
        self.assertEqual(len(res["queries"]), 1)
        self.assertEqual(res["queries"][0]["query_id"], "Q0")
        self.assertTrue(any("API Connection timeout" in w for w in res["warnings"]))

    def test_11_offline_no_network(self):
        """100% offline: không có API key trong môi trường vẫn chạy bình thường với mock generator."""
        def mock_gen(prompt: str, cfg: Dict) -> str:
            return json.dumps({"queries": [{"text": "Mock response", "focus": "paraphrase"}]})

        res = generate_multi_queries("Offline test", config=self.config, query_generator_fn=mock_gen)
        self.assertEqual(res["status"], "ready")
        self.assertEqual(len(res["queries"]), 2)


if __name__ == "__main__":
    unittest.main()
