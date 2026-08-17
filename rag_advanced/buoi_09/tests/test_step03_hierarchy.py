"""Unit tests cho Bước 03 — Hierarchical Registry Builder & Storage.

Bao gồm 14 test cases bắt buộc (100% offline, dùng temporary directories & fixtures):
1. Metadata precedence
2. Heading inferred ở đầu chunk
3. Carry forward trong cùng source
4. Không carry forward qua source khác
5. Inline 'Điều N' không bị nhận nhầm là heading
6. Conflict đặt ambiguous/warning
7. Numeric chunk ordering
8. Stable parent ID (deterministic)
9. Parent split tại child boundary (với max_chars limit)
10. Oversized child warning
11. Mỗi child thuộc đúng một parent
12. Parent pages/count/text đúng
13. Atomic build và manifest fingerprint
14. Status không tạo/sửa file (read-only)
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from hierarchical_rag import (
    build_parents,
    hierarchy_status,
    load_hierarchical_chunks,
    load_hierarchy,
    resolve_hierarchy,
    save_hierarchy,
)


class TestHierarchyStep03(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.default_config = {
            "parent_max_chars": 500,
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_metadata_precedence(self):
        """Metadata structure hợp lệ được ưu tiên trên hết."""
        chunks = [{
            "chunk_id": "src1:0001",
            "source": "src1.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "Khoản 1. Nội dung khoản 1.",
            "structure": {"article": "Điều 5. Quy định chung"}
        }]
        children = resolve_hierarchy(chunks)
        self.assertEqual(len(children), 1)
        c = children[0]
        self.assertEqual(c["resolution_method"], "metadata")
        self.assertEqual(c["structural_path"]["article"], "Điều 5. Quy định chung")
        self.assertFalse(c["ambiguous"])

    def test_02_heading_inferred(self):
        """Heading 'Điều N' ở đầu chunk được tự động nhận diện nếu không có metadata."""
        chunks = [{
            "chunk_id": "src1:0001",
            "source": "src1.pdf",
            "page_start": 2,
            "page_end": 2,
            "text": "Điều 10. Thời hạn cho vay ngắn hạn.",
            "structure": {}
        }]
        children = resolve_hierarchy(chunks)
        c = children[0]
        self.assertEqual(c["resolution_method"], "heading_inferred")
        self.assertIn("Điều 10", c["structural_path"]["article"])
        self.assertFalse(c["ambiguous"])

    def test_03_carry_forward_same_source(self):
        """Kế thừa article từ chunk trước trong CÙNG source khi chunk sau không có metadata/heading."""
        chunks = [
            {
                "chunk_id": "src1:0001",
                "source": "src1.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Điều 8. Nhu cầu cấm cho vay.",
                "structure": {}
            },
            {
                "chunk_id": "src1:0002",
                "source": "src1.pdf",
                "page_start": 1,
                "page_end": 2,
                "text": "Điểm a) Vay để kinh doanh hàng cấm.",
                "structure": {}
            }
        ]
        children = resolve_hierarchy(chunks)
        self.assertEqual(children[0]["resolution_method"], "heading_inferred")
        self.assertEqual(children[1]["resolution_method"], "carried_forward")
        self.assertEqual(children[1]["structural_path"]["article"], children[0]["structural_path"]["article"])

    def test_04_no_carry_across_sources(self):
        """Không carry forward sang source mới — source mới rơi vào document_fallback nếu không có info."""
        chunks = [
            {
                "chunk_id": "src1:0001",
                "source": "src1.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Điều 8. Nhu cầu cấm cho vay.",
                "structure": {}
            },
            {
                "chunk_id": "src2:0001",
                "source": "src2.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Nội dung chung không có điều.",
                "structure": {}
            }
        ]
        children = resolve_hierarchy(chunks)
        self.assertEqual(children[1]["source"], "src2.pdf")
        self.assertEqual(children[1]["resolution_method"], "document_fallback")
        self.assertIsNone(children[1]["structural_path"]["article"])

    def test_05_inline_dieu_n_not_heading(self):
        """Cụm 'Điều N' nằm giữa câu không bị nhận nhầm là heading."""
        chunks = [{
            "chunk_id": "src1:0001",
            "source": "src1.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "Khách hàng phải đáp ứng các điều kiện quy định tại Điều 7 và Điều 8 của Thông tư này.",
            "structure": {}
        }]
        children = resolve_hierarchy(chunks)
        c = children[0]
        self.assertEqual(c["resolution_method"], "document_fallback")
        self.assertIsNone(c["structural_path"]["article"])

    def test_06_conflict_sets_ambiguous(self):
        """Khi metadata bảo Điều A nhưng text heading bảo Điều B -> đặt ambiguous=True + warning."""
        chunks = [{
            "chunk_id": "src1:0001",
            "source": "src1.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "Điều 10. Tiêu đề khác.",
            "structure": {"article": "Điều 8. Tiêu đề metadata"}
        }]
        children = resolve_hierarchy(chunks)
        c = children[0]
        self.assertTrue(c["ambiguous"])
        self.assertEqual(c["structural_path"]["article"], "Điều 8. Tiêu đề metadata")
        self.assertTrue(any("Article conflict" in w for w in c["warnings"]))

    def test_07_numeric_chunk_ordering(self):
        """Sắp xếp chunk theo chuỗi số cuối của chunk_id (:2 xếp trước :10) thay vì sắp xếp lexical."""
        chunk_file = self.temp_dir / "sample__hierarchical.json"
        data = [
            {"chunk_id": "src1:0010", "strategy": "hierarchical", "source": "src1.pdf", "page_start": 5, "page_end": 5, "text": "Chunk 10"},
            {"chunk_id": "src1:0002", "strategy": "hierarchical", "source": "src1.pdf", "page_start": 1, "page_end": 1, "text": "Chunk 2"},
            {"chunk_id": "src1:0001", "strategy": "hierarchical", "source": "src1.pdf", "page_start": 1, "page_end": 1, "text": "Chunk 1"},
        ]
        chunk_file.write_text(json.dumps(data), encoding="utf-8")

        loaded, stats, _ = load_hierarchical_chunks(self.temp_dir)
        ids = [c["chunk_id"] for c in loaded]
        self.assertEqual(ids, ["src1:0001", "src1:0002", "src1:0010"])

    def test_08_stable_parent_id(self):
        """Cùng input & config luôn cho ra parent_id giống hệt nhau (deterministic)."""
        chunks = [{
            "chunk_id": "src1:0001",
            "source": "src1.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "Điều 8. Nội dung.",
            "structure": {}
        }]
        children1 = resolve_hierarchy(chunks)
        parents1, _ = build_parents(children1, self.default_config)

        children2 = resolve_hierarchy(chunks)
        parents2, _ = build_parents(children2, self.default_config)

        self.assertEqual(parents1[0]["parent_id"], parents2[0]["parent_id"])

    def test_09_parent_split_at_child_boundary(self):
        """Khi tổng độ dài vượt parent_max_chars, chia parent window tại ranh giới child."""
        config = {"parent_max_chars": 50}
        chunks = [
            {"chunk_id": "src1:0001", "source": "src1.pdf", "page_start": 1, "page_end": 1, "text": "A" * 30, "structure": {"article": "Điều 1"}},
            {"chunk_id": "src1:0002", "source": "src1.pdf", "page_start": 1, "page_end": 1, "text": "B" * 30, "structure": {"article": "Điều 1"}},
        ]
        children = resolve_hierarchy(chunks)
        parents, updated_children = build_parents(children, config)

        self.assertEqual(len(parents), 2)
        self.assertEqual(parents[0]["child_ids"], ["src1:0001"])
        self.assertEqual(parents[1]["child_ids"], ["src1:0002"])
        self.assertNotEqual(updated_children[0]["parent_id"], updated_children[1]["parent_id"])

    def test_10_oversized_child_warning(self):
        """Một child đơn lẻ vượt parent_max_chars vẫn được giữ nguyên và đánh warning oversized_single_child."""
        config = {"parent_max_chars": 20}
        chunks = [{
            "chunk_id": "src1:0001",
            "source": "src1.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "X" * 50,
            "structure": {"article": "Điều 1"}
        }]
        children = resolve_hierarchy(chunks)
        parents, _ = build_parents(children, config)

        self.assertEqual(len(parents), 1)
        self.assertEqual(parents[0]["char_count"], 50)
        self.assertTrue(any("oversized_single_child" in w for w in parents[0]["warnings"]))

    def test_11_each_child_exactly_one_parent(self):
        """Mỗi child phải gán đúng 1 parent_id và thuộc đúng 1 parent window."""
        chunks = [
            {"chunk_id": "src1:0001", "source": "src1.pdf", "page_start": 1, "page_end": 1, "text": "Text 1", "structure": {"article": "Điều 1"}},
            {"chunk_id": "src1:0002", "source": "src1.pdf", "page_start": 1, "page_end": 2, "text": "Text 2", "structure": {"article": "Điều 1"}},
        ]
        children = resolve_hierarchy(chunks)
        parents, updated_children = build_parents(children, self.default_config)

        for c in updated_children:
            self.assertIsNotNone(c["parent_id"])
            matching = [p for p in parents if c["child_id"] in p["child_ids"]]
            self.assertEqual(len(matching), 1)

    def test_12_parent_pages_count_text_correct(self):
        """Kiểm tra parent page_start, page_end, char_count và text được tính chính xác."""
        chunks = [
            {"chunk_id": "src1:0001", "source": "src1.pdf", "page_start": 2, "page_end": 3, "text": "Do đoạn 1.", "structure": {"article": "Điều 1"}},
            {"chunk_id": "src1:0002", "source": "src1.pdf", "page_start": 3, "page_end": 5, "text": "Do đoạn 2.", "structure": {"article": "Điều 1"}},
        ]
        children = resolve_hierarchy(chunks)
        parents, _ = build_parents(children, self.default_config)

        p = parents[0]
        self.assertEqual(p["page_start"], 2)
        self.assertEqual(p["page_end"], 5)
        self.assertEqual(p["text"], "Do đoạn 1.\nDo đoạn 2.")
        self.assertEqual(p["char_count"], len(p["text"]))

    def test_13_atomic_build_and_manifest_fingerprint(self):
        """Build ghi dữ liệu atomically và sinh manifest fingerprint chuẩn."""
        store_dir = self.temp_dir / "storage"
        chunks = [{"chunk_id": "src1:0001", "source": "src1.pdf", "page_start": 1, "page_end": 1, "text": "Text", "structure": {}}]
        children = resolve_hierarchy(chunks)
        parents, children = build_parents(children, self.default_config)

        load_stats = {"sources": 1}
        fps = {"src1.pdf": "hash123"}
        manifest = save_hierarchy(children, parents, load_stats, fps, self.default_config, storage_dir=store_dir)

        c_loaded, p_loaded, m_loaded = load_hierarchy(storage_dir=store_dir)
        self.assertEqual(len(c_loaded), 1)
        self.assertEqual(len(p_loaded), 1)
        self.assertEqual(m_loaded["input_fingerprints"]["src1.pdf"], "hash123")

    def test_14_status_read_only(self):
        """Chạy hierarchy_status khi thư mục không tồn tại không làm sinh file/thư mục mới."""
        non_existent = self.temp_dir / "no_store"
        status = hierarchy_status(storage_dir=non_existent)

        self.assertFalse(status["registry_exists"])
        self.assertFalse(non_existent.exists())


if __name__ == "__main__":
    unittest.main()
