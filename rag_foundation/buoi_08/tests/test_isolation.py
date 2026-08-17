import os
import sys
import unittest
import tempfile
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
BUOI_08_DIR = TEST_DIR.parent
if str(BUOI_08_DIR) not in sys.path:
    sys.path.insert(0, str(BUOI_08_DIR))

import advanced_rag
import rag


class TestSystemIsolationAndUIHelpers(unittest.TestCase):
    # Test 28: Config hoạt động độc lập bất chấp CWD hiện tại
    def test_28_config_independent_of_cwd(self):
        current_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
                os.chdir(tmp_dir)
                cfg = advanced_rag.get_advanced_config()
                self.assertIn("RRF_K", cfg)
                self.assertIn("FINAL_TOP_K", cfg)
        finally:
            os.chdir(current_cwd)

    # Test 29: Lệnh Status không tự ý khởi tạo resource hay collection mới
    def test_29_status_command_read_only(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_storage:
            st = advanced_rag.run_advanced_status(
                strategy="hierarchical", storage_dir=Path(tmp_storage)
            )
            self.assertFalse(st["collection_exists"])
            self.assertEqual(st["collection_count"], 0)
            client = rag.get_chroma_client(Path(tmp_storage))
            self.assertEqual(len(client.list_collections()), 0)

    # Test 30: Không tải HuggingFace / Reranker model khi import module
    def test_30_no_model_loading_on_import(self):
        self.assertIsNone(advanced_rag._RERANKER_CACHE["model"])
        self.assertIsNone(advanced_rag._RERANKER_CACHE["tokenizer"])


if __name__ == "__main__":
    unittest.main()
