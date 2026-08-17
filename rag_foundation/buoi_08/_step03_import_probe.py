"""One-off Bước 03 import probe — không tải model HF, không gọi Gemini."""
import importlib
import os
import sys
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PACKAGES = [
    ("streamlit", "streamlit"),
    ("google.genai", "google-genai"),
    ("chromadb", "chromadb"),
    ("dotenv", "python-dotenv"),
    ("rank_bm25", "rank-bm25"),
    ("transformers", "transformers"),
    ("torch", "torch"),
]

print("IMPORT_PROBE")
for module_name, dist_name in PACKAGES:
    importlib.import_module(module_name)
    print(f"OK {dist_name} {version(dist_name)}")

from advanced_rag import load_advanced_config, resolve_rerank_pool_size

cfg = load_advanced_config(env_path=ROOT / ".env.example")
assert resolve_rerank_pool_size(5, cfg) == 5
assert resolve_rerank_pool_size(100, cfg) == cfg["rerank_candidates"]
safe = {k: v for k, v in cfg.items() if k != "gemini_api_key"}
safe["gemini_api_key"] = "SET" if cfg["gemini_api_key"] else "EMPTY"
print("CONFIG_OK", safe)
