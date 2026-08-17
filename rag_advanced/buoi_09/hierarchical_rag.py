"""Buổi 09 — Hierarchical RAG module (Bước 03).

Triển khai Bước 03:
  - load_hierarchical_config()   : load & validate tất cả biến .env Buổi 09
  - load_hierarchical_chunks()   : load hierarchical chunks từ Buổi 05 output
  - resolve_hierarchy()          : gán article/chapter label, resolution_method cho mỗi child
  - build_parents()              : gom children thành parent windows
  - save_hierarchy()             : ghi registry atomic vào storage/hierarchy/
  - load_hierarchy()             : load registry từ storage
  - hierarchy_status()           : read-only status (không tạo file)
  - run_hierarchy_audit()        : audit report (không ghi file)
  - CLI: hierarchy-audit, build-hierarchy, hierarchy-status

Chưa triển khai (Bước 04+):
  - multi_query_generate(), cross_query_rrf(), aggregate_to_parents(),
    rerank_parents(), query_hierarchical_rag()
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = BASE_DIR / ".env"
BUOI05_CHUNKS_DIR = (
    BASE_DIR.parent.parent / "rag_foundation" / "buoi_05" / "output" / "chunks"
)
STORAGE_DIR = BASE_DIR / "storage" / "hierarchy"
SCHEMA_VERSION = "1.0"

# ─── Regex patterns ────────────────────────────────────────────────────────────
# Heading thực sự ở ĐẦU chunk: "Điều N" sau dấu markdown/space tùy chọn, rồi dấu chấm + tiêu đề.
# Không match nếu text bắt đầu bằng số, chữ khác, hoặc nếu "Điều N" xuất hiện giữa câu.
_RE_HEADING_START = re.compile(
    r'^[\s\*\#""\u201c\u201d]*Đi[eề]u\s+(\d+)\s*\.?\s*(.*)',
    re.IGNORECASE,
)
# Trích xuất số thứ tự từ chunk_id: ...:<digits>
_RE_CHUNK_SEQ = re.compile(r":(\d+)$")


# ──────────────────────────────────────────────────────────────────────────────
# § 1. Config
# ──────────────────────────────────────────────────────────────────────────────

def load_hierarchical_config(env_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load và validate toàn bộ config từ .env (relative to __file__, không phụ thuộc cwd)."""
    if env_path is None:
        env_path = DEFAULT_ENV_PATH
    load_dotenv(env_path)

    def _int(name: str, default: int, lo: int, hi: int) -> int:
        raw = os.getenv(name, str(default)).strip()
        try:
            v = int(raw)
        except ValueError:
            raise ValueError(f"{name} phải là số nguyên, nhận: '{raw}'")
        if not (lo <= v <= hi):
            raise ValueError(f"{name} phải trong [{lo}, {hi}], nhận: {v}")
        return v

    def _float(name: str, default: float, lo: float, hi: float) -> float:
        raw = os.getenv(name, str(default)).strip()
        try:
            v = float(raw)
        except ValueError:
            raise ValueError(f"{name} phải là số thực, nhận: '{raw}'")
        if not (lo <= v <= hi):
            raise ValueError(f"{name} phải trong [{lo}, {hi}], nhận: {v}")
        return v

    def _str_ne(name: str, default: str) -> str:
        v = os.getenv(name, default).strip()
        if not v:
            raise ValueError(f"{name} không được để rỗng")
        return v

    mq_count = _int("MULTI_QUERY_COUNT", 3, 1, 5)
    mq_max_chars = _int("MULTI_QUERY_MAX_CHARS", 300, 50, 1000)
    mq_temp = _float("MULTI_QUERY_TEMPERATURE", 0.2, 0.0, 1.0)
    orig_w = _float("MULTI_QUERY_ORIGINAL_WEIGHT", 1.5, 0.0, 100.0)
    var_w = _float("MULTI_QUERY_VARIANT_WEIGHT", 1.0, 0.0, 100.0)
    if orig_w == 0.0 and var_w == 0.0:
        raise ValueError(
            "MULTI_QUERY_ORIGINAL_WEIGHT và MULTI_QUERY_VARIANT_WEIGHT không được đồng thời bằng 0"
        )
    mq_rrf_k = _int("MULTI_QUERY_RRF_K", 60, 1, 10_000)
    per_q_cand = _int("PER_QUERY_CANDIDATES", 12, 1, 100)
    parent_max = _int("PARENT_MAX_CHARS", 6000, 1000, 20_000)
    parent_child_lim = _int("PARENT_SCORE_CHILD_LIMIT", 3, 1, 20)
    parent_rrf_k = _int("PARENT_RRF_K", 60, 1, 10_000)
    parent_cand = _int("PARENT_CANDIDATES", 10, 1, 100)
    final_top_k = _int("FINAL_PARENT_TOP_K", 3, 1, 100)
    if final_top_k > parent_cand:
        raise ValueError(
            f"FINAL_PARENT_TOP_K ({final_top_k}) > PARENT_CANDIDATES ({parent_cand})"
        )
    total_ctx = _int("TOTAL_CONTEXT_MAX_CHARS", 16_000, 1000, 200_000)
    if total_ctx < parent_max:
        raise ValueError(
            f"TOTAL_CONTEXT_MAX_CHARS ({total_ctx}) < PARENT_MAX_CHARS ({parent_max})"
        )
    reranker_model = _str_ne("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    emb_model = _str_ne("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    gen_model = _str_ne("GEMINI_GENERATION_MODEL", "gemini-3.5-flash-lite")
    bm25_cand = _int("BM25_CANDIDATES", 20, 1, 100)
    sem_cand = _int("SEMANTIC_CANDIDATES", 20, 1, 100)
    rrf_k = _int("RRF_K", 60, 1, 10_000)

    rerank_min_score = _float("RERANK_MIN_SCORE", 0.35, 0.0, 1.0)

    # Set defaults in os.environ for downstream modules if missing
    os.environ.setdefault("GEMINI_EMBEDDING_MODEL", emb_model)
    os.environ.setdefault("GEMINI_GENERATION_MODEL", gen_model)
    os.environ.setdefault("GEMINI_EMBEDDING_DIM", os.getenv("GEMINI_EMBEDDING_DIM", "768"))
    os.environ.setdefault("DEFAULT_TOP_K", os.getenv("DEFAULT_TOP_K", "5"))
    os.environ.setdefault("RAG_MAX_DISTANCE", os.getenv("RAG_MAX_DISTANCE", "0.45"))

    return {
        "multi_query_count": mq_count,
        "multi_query_max_chars": mq_max_chars,
        "multi_query_temperature": mq_temp,
        "multi_query_original_weight": orig_w,
        "multi_query_variant_weight": var_w,
        "multi_query_rrf_k": mq_rrf_k,
        "per_query_candidates": per_q_cand,
        "parent_max_chars": parent_max,
        "parent_score_child_limit": parent_child_lim,
        "parent_rrf_k": parent_rrf_k,
        "parent_candidates": parent_cand,
        "final_parent_top_k": final_top_k,
        "total_context_max_chars": total_ctx,
        "rerank_min_score": rerank_min_score,
        "reranker_model": reranker_model,
        "gemini_embedding_model": emb_model,
        "gemini_generation_model": gen_model,
        "bm25_candidates": bm25_cand,
        "semantic_candidates": sem_cand,
        "rrf_k": rrf_k,
    }


def _config_identity(config: Dict[str, Any]) -> str:
    """Stable hash của các config key ảnh hưởng đến hierarchy structure."""
    relevant = {k: config[k] for k in ("parent_max_chars",) if k in config}
    blob = json.dumps(relevant, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ──────────────────────────────────────────────────────────────────────────────
# § 2. Chunk loading
# ──────────────────────────────────────────────────────────────────────────────

def _chunk_seq_num(chunk_id: str) -> int:
    """Trả về số thứ tự cuối chunk_id (ví dụ: '...0042' → 42).
    Dùng để sort numeric, không sort lexical.
    """
    m = _RE_CHUNK_SEQ.search(chunk_id)
    return int(m.group(1)) if m else 0


def load_hierarchical_chunks(
    input_dir: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, str]]:
    """Load tất cả hierarchical chunks từ thư mục input.

    Returns:
        (chunks, load_stats, file_fingerprints)
        chunks được group by source, sort by numeric chunk_id sequence.
    """
    if input_dir is None:
        input_dir = BUOI05_CHUNKS_DIR
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise ValueError(f"Input directory not found: {input_dir}")

    h_files = sorted(input_dir.glob("*__hierarchical.json"))
    if not h_files:
        # Also accept any .json for test fixtures
        h_files = sorted(input_dir.glob("*.json"))
    if not h_files:
        raise ValueError(f"No hierarchical JSON files found in: {input_dir}")

    all_chunks: List[Dict[str, Any]] = []
    seen_ids: Dict[str, str] = {}
    file_fingerprints: Dict[str, str] = {}
    stats: Dict[str, Any] = {
        "files": len(h_files),
        "total_records": 0,
        "hierarchical_selected": 0,
        "skipped_strategy": 0,
        "skipped_empty_text": 0,
        "valid": 0,
    }

    for fpath in h_files:
        raw = fpath.read_bytes()
        file_fingerprints[fpath.name] = hashlib.sha256(raw).hexdigest()

        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{fpath.name}: invalid JSON — {exc}")

        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict) and "chunks" in payload:
            records = payload["chunks"]
        else:
            raise ValueError(f"{fpath.name}: expected JSON list or object with 'chunks'")

        for idx, rec in enumerate(records):
            stats["total_records"] += 1

            if not isinstance(rec, dict):
                raise ValueError(f"{fpath.name} record {idx}: not a JSON object")

            strategy = str(rec.get("strategy", "")).strip()
            if strategy != "hierarchical":
                stats["skipped_strategy"] += 1
                continue

            stats["hierarchical_selected"] += 1

            # Validate required fields
            for field in ("chunk_id", "source", "page_start", "page_end", "text"):
                if field not in rec:
                    raise ValueError(
                        f"{fpath.name} record {idx}: missing required field '{field}'"
                    )

            chunk_id = str(rec["chunk_id"]).strip()
            if not chunk_id:
                raise ValueError(f"{fpath.name} record {idx}: empty chunk_id")

            if chunk_id in seen_ids:
                raise ValueError(
                    f"Duplicate chunk_id '{chunk_id}': "
                    f"first seen in {seen_ids[chunk_id]}, "
                    f"repeated in {fpath.name} record {idx}"
                )
            seen_ids[chunk_id] = fpath.name

            source = str(rec["source"]).strip()
            if not source:
                raise ValueError(f"{fpath.name} record {idx}: empty source")

            page_start = rec["page_start"]
            page_end = rec["page_end"]
            if isinstance(page_start, bool) or not isinstance(page_start, int):
                raise ValueError(
                    f"{fpath.name} record {idx}: page_start must be integer, got {type(page_start).__name__}"
                )
            if isinstance(page_end, bool) or not isinstance(page_end, int):
                raise ValueError(
                    f"{fpath.name} record {idx}: page_end must be integer, got {type(page_end).__name__}"
                )
            if page_start < 1 or page_end < 1:
                raise ValueError(
                    f"{fpath.name} record {idx}: page numbers must be >= 1, got [{page_start}, {page_end}]"
                )
            if page_start > page_end:
                raise ValueError(
                    f"{fpath.name} record {idx}: page_start ({page_start}) > page_end ({page_end})"
                )

            text = str(rec.get("text", "")).strip()
            if not text:
                stats["skipped_empty_text"] += 1
                continue

            structure = rec.get("structure")
            if structure is not None and not isinstance(structure, dict):
                raise ValueError(
                    f"{fpath.name} record {idx}: 'structure' must be a JSON object or null"
                )

            all_chunks.append({
                "chunk_id": chunk_id,
                "source": source,
                "page_start": page_start,
                "page_end": page_end,
                "text": text,
                "structure": structure or {},
                "_file": fpath.name,
            })
            stats["valid"] += 1

    # Group by source, sort each group numerically by chunk_id sequence
    by_source: Dict[str, List[Dict]] = {}
    for chunk in all_chunks:
        by_source.setdefault(chunk["source"], []).append(chunk)

    ordered: List[Dict[str, Any]] = []
    for src in sorted(by_source.keys()):
        grp = sorted(by_source[src], key=lambda c: _chunk_seq_num(c["chunk_id"]))
        ordered.extend(grp)

    stats["sources"] = len(by_source)
    return ordered, stats, file_fingerprints


# ──────────────────────────────────────────────────────────────────────────────
# § 3. Hierarchy resolution
# ──────────────────────────────────────────────────────────────────────────────

def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _extract_article_num(article_str: str) -> Optional[int]:
    """Trích số Điều từ chuỗi như 'Điều 8. Tiêu đề' → 8."""
    if not article_str:
        return None
    m = re.search(r"Đi[eề]u\s+(\d+)", article_str, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _infer_heading(text: str) -> Optional[Tuple[int, str]]:
    """Nhận diện heading Điều ở ĐẦU chunk.

    Returns (article_num, full_label) nếu tìm thấy, None nếu không.
    Không nhận diện "Điều N" nằm giữa câu văn.
    """
    norm = _nfc(text)
    m = _RE_HEADING_START.match(norm)
    if not m:
        return None
    num = int(m.group(1))
    rest = m.group(2).strip().rstrip('*"\u201d').strip()
    label = f"Điều {num}. {rest}" if rest else f"Điều {num}"
    return (num, label)


def resolve_hierarchy(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Gán hierarchy metadata cho mỗi child chunk.

    Thứ tự ưu tiên per chunk:
      1. metadata (structure.article tồn tại và hợp lệ)
      2. heading_inferred (text bắt đầu bằng 'Điều N')
      3. carried_forward (article gần nhất TRONG CÙNG source)
      4. document_fallback

    Không carry forward qua source khác.
    Ambiguous = True nếu metadata.article_num ≠ heading.article_num.
    """
    # Carry state per source
    carry: Dict[str, Dict[str, Any]] = {}
    # carry[source] = {"chapter": str|None, "article_key": str|None, "article_num": int|None}

    result: List[Dict[str, Any]] = []

    for chunk in chunks:
        source = chunk["source"]
        struct = chunk.get("structure") or {}
        text = chunk["text"]
        warnings: List[str] = []
        ambiguous = False

        # Extract metadata fields
        meta_chapter = str(struct.get("chapter") or "").strip() or None
        meta_article = str(struct.get("article") or "").strip() or None
        meta_clause = str(struct.get("clause") or "").strip() or None
        meta_point = str(struct.get("point") or "").strip() or None
        meta_article_num = _extract_article_num(meta_article) if meta_article else None

        # Infer heading from text
        head = _infer_heading(text)
        head_num, head_label = head if head else (None, None)

        # Initialize carry state for new source
        if source not in carry:
            carry[source] = {"chapter": None, "article_key": None, "article_num": None}

        cs = carry[source]

        # Always update chapter from metadata if present
        if meta_chapter:
            cs["chapter"] = meta_chapter

        # ── Priority 1: metadata article ──
        if meta_article:
            resolution_method = "metadata"
            resolved_article = meta_article
            resolved_article_num = meta_article_num
            # Update carry
            cs["article_key"] = resolved_article
            cs["article_num"] = resolved_article_num

            # Conflict check: metadata article_num vs heading article_num
            if head_num is not None and head_num != meta_article_num:
                ambiguous = True
                warnings.append(
                    f"Article conflict: metadata says 'Điều {meta_article_num}' "
                    f"but text heading says 'Điều {head_num}'; kept metadata"
                )

        # ── Priority 2: heading inferred ──
        elif head_num is not None:
            resolution_method = "heading_inferred"
            resolved_article = head_label
            resolved_article_num = head_num
            # Update carry
            cs["article_key"] = resolved_article
            cs["article_num"] = resolved_article_num

        # ── Priority 3: carry forward (same source only) ──
        elif cs["article_key"] is not None:
            resolution_method = "carried_forward"
            resolved_article = cs["article_key"]
            resolved_article_num = cs["article_num"]

        # ── Priority 4: document fallback ──
        else:
            resolution_method = "document_fallback"
            resolved_article = None
            resolved_article_num = None

        resolved_chapter = cs["chapter"]

        structural_path = {
            "chapter": resolved_chapter,
            "article": resolved_article,
            "clause": meta_clause,
            "point": meta_point,
        }

        result.append({
            "child_id": chunk["chunk_id"],
            "parent_id": None,   # set by build_parents
            "source": source,
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "text": text,
            "structural_path": structural_path,
            # internal keys removed before persisting
            "_article_key": resolved_article,
            "_article_num": resolved_article_num,
            "resolution_method": resolution_method,
            "ambiguous": ambiguous,
            "warnings": warnings,
        })

    return result


# ──────────────────────────────────────────────────────────────────────────────
# § 4. Parent building
# ──────────────────────────────────────────────────────────────────────────────

_FALLBACK_GROUP = "__DOCUMENT_FALLBACK__"


def _make_parent_id(source: str, article_key: Optional[str], window_index: int) -> str:
    """Stable deterministic parent ID (hash-based, byte-equivalent for same input)."""
    art = article_key if article_key is not None else _FALLBACK_GROUP
    key = f"{source}||{art}||{window_index}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"p:{h}"


def _split_into_windows(
    children: List[Dict[str, Any]], max_chars: int
) -> List[List[Dict[str, Any]]]:
    """Split children list into windows respecting max_chars at child boundaries.

    Rules:
    - Không cắt giữa child.
    - Nếu một child đơn lẻ vượt max_chars → để nguyên (sẽ warning oversized).
    - Mỗi child thuộc đúng một window.
    """
    windows: List[List[Dict]] = []
    current: List[Dict] = []
    current_chars = 0

    for child in children:
        child_chars = len(child["text"])
        if current and current_chars + child_chars > max_chars:
            windows.append(current)
            current = [child]
            current_chars = child_chars
        else:
            current.append(child)
            current_chars += child_chars

    if current:
        windows.append(current)

    return windows


def build_parents(
    children: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Group children into parent windows, set parent_id on children in-place.

    Returns:
        (parents, updated_children)  # children have parent_id filled
    """
    parent_max = config["parent_max_chars"]

    # Group children by (source, article_key)
    groups: Dict[Tuple[str, Optional[str]], List[Dict]] = {}
    for child in children:
        art = child.get("_article_key")
        gk = (child["source"], art)  # art may be None
        groups.setdefault(gk, []).append(child)

    parents: List[Dict[str, Any]] = []

    for (source, art_key), grp in groups.items():
        windows = _split_into_windows(grp, parent_max)
        for win_idx, window in enumerate(windows):
            pid = _make_parent_id(source, art_key, win_idx)
            parent_text = "\n".join(c["text"] for c in window)
            char_count = len(parent_text)
            page_start = min(c["page_start"] for c in window)
            page_end = max(c["page_end"] for c in window)
            ambiguous_cnt = sum(1 for c in window if c["ambiguous"])
            parent_warnings: List[str] = []

            # Oversized single child warning
            if len(window) == 1 and len(window[0]["text"]) > parent_max:
                parent_warnings.append(
                    f"oversized_single_child: {window[0]['child_id']} "
                    f"({len(window[0]['text'])} chars > {parent_max})"
                )

            parents.append({
                "parent_id": pid,
                "source": source,
                "page_start": page_start,
                "page_end": page_end,
                "article_key": art_key,
                "window_index": win_idx,
                "child_ids": [c["child_id"] for c in window],
                "text": parent_text,
                "char_count": char_count,
                "ambiguous_child_count": ambiguous_cnt,
                "warnings": parent_warnings,
            })

            for c in window:
                c["parent_id"] = pid

    return parents, children


# ──────────────────────────────────────────────────────────────────────────────
# § 5. Storage — atomic save / load / status
# ──────────────────────────────────────────────────────────────────────────────

def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically: write to .tmp in same dir, then replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        text = json.dumps(data, ensure_ascii=False, indent=2)
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def _clean_child(child: Dict[str, Any]) -> Dict[str, Any]:
    """Remove internal keys (_article_key, _article_num, _file) before persisting."""
    return {k: v for k, v in child.items() if not k.startswith("_")}


def save_hierarchy(
    children: List[Dict[str, Any]],
    parents: List[Dict[str, Any]],
    load_stats: Dict[str, Any],
    file_fingerprints: Dict[str, str],
    config: Dict[str, Any],
    storage_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Ghi registry atomic vào storage/hierarchy/.

    Không xóa store cũ trước khi build mới thành công.
    Returns manifest dict.
    """
    if storage_dir is None:
        storage_dir = STORAGE_DIR

    storage_dir = Path(storage_dir)

    clean_children = [_clean_child(c) for c in children]

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "strategy": "hierarchical",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "config_identity": _config_identity(config),
        "input_fingerprints": file_fingerprints,
        "counts": {
            "children": len(children),
            "parents": len(parents),
            "sources": load_stats.get("sources", 0),
            "ambiguous_children": sum(1 for c in children if c.get("ambiguous")),
            "warned_children": sum(1 for c in children if c.get("warnings")),
            "orphan_children": sum(
                1 for c in children if c.get("_article_key") is None
            ),
            "warned_parents": sum(1 for p in parents if p.get("warnings")),
            "resolution_breakdown": _resolution_breakdown(children),
        },
    }

    # Atomic writes: children → parents → manifest (manifest last = commit point)
    _atomic_write_json(storage_dir / "children.json", clean_children)
    _atomic_write_json(storage_dir / "parents.json", parents)
    _atomic_write_json(storage_dir / "manifest.json", manifest)

    return manifest


def _resolution_breakdown(children: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for c in children:
        m = c.get("resolution_method", "unknown")
        counts[m] = counts.get(m, 0) + 1
    return counts


def load_hierarchy(
    storage_dir: Optional[Path] = None,
) -> Tuple[List[Dict], List[Dict], Dict]:
    """Load registry từ storage. Raises ValueError nếu không tồn tại hoặc corrupt."""
    if storage_dir is None:
        storage_dir = STORAGE_DIR
    storage_dir = Path(storage_dir)

    for fname in ("manifest.json", "children.json", "parents.json"):
        p = storage_dir / fname
        if not p.exists():
            raise ValueError(
                f"Hierarchy registry không tồn tại tại {storage_dir}. "
                f"Hãy chạy 'build-hierarchy' trước. Thiếu: {fname}"
            )

    manifest = json.loads((storage_dir / "manifest.json").read_text("utf-8"))
    children = json.loads((storage_dir / "children.json").read_text("utf-8"))
    parents = json.loads((storage_dir / "parents.json").read_text("utf-8"))
    return children, parents, manifest


def hierarchy_status(storage_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Read-only status. Không mkdir, không build, không sửa file."""
    if storage_dir is None:
        storage_dir = STORAGE_DIR
    storage_dir = Path(storage_dir)

    files = {
        "manifest.json": (storage_dir / "manifest.json").exists(),
        "children.json": (storage_dir / "children.json").exists(),
        "parents.json": (storage_dir / "parents.json").exists(),
    }
    registry_exists = all(files.values())

    if not registry_exists:
        return {
            "registry_exists": False,
            "storage_dir": str(storage_dir),
            "files": files,
            "manifest": None,
        }

    try:
        manifest = json.loads((storage_dir / "manifest.json").read_text("utf-8"))
    except Exception as exc:
        return {
            "registry_exists": False,
            "storage_dir": str(storage_dir),
            "files": files,
            "manifest": None,
            "error": str(exc),
        }

    return {
        "registry_exists": True,
        "storage_dir": str(storage_dir),
        "files": files,
        "manifest": manifest,
    }


# ──────────────────────────────────────────────────────────────────────────────
# § 6. Audit (read-only, no file write)
# ──────────────────────────────────────────────────────────────────────────────

def run_hierarchy_audit(
    input_dir: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load chunks, resolve hierarchy, build parents — không ghi file.

    Returns audit report dict.
    """
    if config is None:
        config = load_hierarchical_config()

    chunks, load_stats, fingerprints = load_hierarchical_chunks(input_dir)
    children = resolve_hierarchy(chunks)
    parents, children = build_parents(children, config)

    # Parent size distribution
    sizes = sorted(p["char_count"] for p in parents)
    n = len(sizes)

    def pct(k):
        if not sizes:
            return 0
        idx = min(int(len(sizes) * k / 100), len(sizes) - 1)
        return sizes[idx]

    parent_size_dist = {
        "min": sizes[0] if sizes else 0,
        "p25": pct(25),
        "median": pct(50),
        "p75": pct(75),
        "p95": pct(95),
        "max": sizes[-1] if sizes else 0,
        "mean": round(sum(sizes) / n, 1) if n else 0,
    }

    # Warning examples
    ambiguous_examples = [
        {"child_id": c["child_id"], "warnings": c["warnings"]}
        for c in children if c.get("ambiguous")
    ][:5]

    parent_warning_examples = [
        {"parent_id": p["parent_id"], "article_key": p["article_key"], "warnings": p["warnings"]}
        for p in parents if p.get("warnings")
    ][:5]

    resolution_breakdown = _resolution_breakdown(children)
    orphan_count = sum(1 for c in children if c.get("_article_key") is None)

    return {
        "load_stats": load_stats,
        "resolution_breakdown": resolution_breakdown,
        "totals": {
            "children": len(children),
            "parents": len(parents),
            "ambiguous_children": sum(1 for c in children if c.get("ambiguous")),
            "orphan_children": orphan_count,
            "warned_parents": sum(1 for p in parents if p.get("warnings")),
        },
        "parent_size_distribution": parent_size_dist,
        "ambiguous_examples": ambiguous_examples,
        "parent_warning_examples": parent_warning_examples,
    }


# ──────────────────────────────────────────────────────────────────────────────
# § 8. Multi-Query Generator (Bước 04)
# ──────────────────────────────────────────────────────────────────────────────

_QUERY_EXPANSION_CACHE: Dict[str, Dict[str, Any]] = {}


def clear_query_expansion_cache() -> None:
    """Clear process-level query expansion cache (used in unit tests)."""
    global _QUERY_EXPANSION_CACHE
    _QUERY_EXPANSION_CACHE.clear()


def _normalize_for_dedup(text: str) -> str:
    """Chuẩn hóa text để deduplicate: Unicode NFC + casefold + loại bỏ punctuation & extra whitespace."""
    s = unicodedata.normalize("NFC", text).casefold()
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())


def _extract_articles_numbers(text: str) -> set[int]:
    """Trích xuất tất cả số Điều có trong câu text."""
    norm = unicodedata.normalize("NFC", text)
    matches = re.findall(r"Đi[eề]u\s+(\d+)", norm, re.IGNORECASE)
    return {int(m) for m in matches}


def _default_gemini_query_generator(
    prompt: str,
    config: Dict[str, Any],
) -> str:
    """Gọi Gemini API thực tế để sinh multi-query variants với structured JSON output."""
    from google import genai
    from google.genai import types
    from pydantic import BaseModel, Field

    class QueryVariantItem(BaseModel):
        text: str = Field(description="Nội dung câu hỏi biến thể")
        focus: str = Field(
            description="Loại biến thể: exact_legal_terms, paraphrase, hoặc missing_aspect"
        )

    class MultiQueryResponse(BaseModel):
        queries: List[QueryVariantItem]

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY không được tìm thấy trong môi trường")

    client = genai.Client(api_key=api_key)
    model_name = config.get("gemini_generation_model", "gemini-3.5-flash-lite")
    temperature = config.get("multi_query_temperature", 0.2)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=MultiQueryResponse,
        ),
    )

    if not response.text:
        raise ValueError("Gemini API trả về response rỗng")

    return response.text


def generate_multi_queries(
    question: str,
    config: Optional[Dict[str, Any]] = None,
    query_generator_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Sinh Multi-Query expansion có kiểm soát từ câu hỏi gốc Q0.

    Args:
        question: Câu hỏi gốc người dùng
        config: Dict config từ load_hierarchical_config()
        query_generator_fn: Optional callable(prompt: str, config: Dict) -> str (dùng cho testing mock)

    Returns:
        Dict theo Query Set Contract
    """
    import time

    if config is None:
        config = load_hierarchical_config()

    q0_clean = unicodedata.normalize("NFC", question.strip())
    mq_max_chars = config["multi_query_max_chars"]
    mq_count = config["multi_query_count"]
    model_name = config.get("gemini_generation_model", "gemini-3.5-flash-lite")
    temp = config.get("multi_query_temperature", 0.2)

    # Validation Q0
    if not q0_clean:
        return {
            "original_question": "",
            "queries": [],
            "model": model_name,
            "generation_latency_ms": 0.0,
            "status": "query_generation_unavailable",
            "warnings": ["Câu hỏi gốc rỗng"],
            "cache_hit": False,
            "dropped_duplicate_count": 0,
        }

    if len(q0_clean) > mq_max_chars:
        # Nếu Q0 quá dài nhưng không rỗng, vẫn giữ Q0
        pass

    # Standard Q0 item
    q0_item = {
        "query_id": "Q0",
        "text": q0_clean,
        "origin": "original",
        "focus": "original_intent",
    }

    # Check cache
    cache_key_raw = f"{q0_clean}||{model_name}||{mq_count}||{temp}||{mq_max_chars}"
    cache_key = hashlib.sha256(cache_key_raw.encode("utf-8")).hexdigest()

    if cache_key in _QUERY_EXPANSION_CACHE:
        cached = json.loads(json.dumps(_QUERY_EXPANSION_CACHE[cache_key]))
        cached["cache_hit"] = True
        cached["generation_latency_ms"] = 0.0
        return cached

    # Prompt hợp đồng
    prompt = f"""Bạn là chuyên gia tra cứu văn bản pháp luật ngân hàng Việt Nam.
Nhiệm vụ: Tạo ra tối đa {mq_count} câu hỏi tìm kiếm biến thể (query variants) từ câu hỏi gốc dưới đây.

QUY TẮC BẮT BUỘC:
1. KHÔNG trả lời câu hỏi gốc. Chỉ tạo các câu hỏi tìm kiếm.
2. Các biến thể phải bao phủ:
   - exact_legal_terms: Sử dụng thuật ngữ pháp lý chính xác trong văn bản quy phạm pháp luật.
   - paraphrase: Diễn đạt lại theo cách khác có cùng ý nghĩa.
   - missing_aspect: Khía cạnh bổ sung hoặc chi tiết liên quan nếu câu hỏi gốc có nhiều ý.
3. KHÔNG thêm bớt sự kiện, thông tin giả định hoặc kết luận pháp lý ngoài câu hỏi gốc.
4. Nếu câu hỏi gốc có chứa số Điều, Khoản, Điểm hoặc tên Thông tư/Nghị định, giữ nguyên các thông số đó. TUYỆT ĐỐI KHÔNG tự phát minh ra số Điều/Khoản khác không có trong câu hỏi gốc.
5. Mỗi câu hỏi không quá {mq_max_chars} ký tự.

CÂU HỎI GỐC: "{q0_clean}"
"""

    gen_fn = query_generator_fn or _default_gemini_query_generator
    t0 = time.perf_counter()
    raw_response_text = ""
    warnings: List[str] = []

    try:
        raw_response_text = gen_fn(prompt, config)
    except Exception as exc:
        err_msg = f"Multi-query generation error: {exc}"
        return {
            "original_question": q0_clean,
            "queries": [q0_item],
            "model": model_name,
            "generation_latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "status": "query_generation_unavailable",
            "warnings": [err_msg],
            "cache_hit": False,
            "dropped_duplicate_count": 0,
        }

    latency_ms = (time.perf_counter() - t0) * 1000

    # Parse JSON
    try:
        parsed_data = json.loads(raw_response_text)
    except json.JSONDecodeError as exc:
        return {
            "original_question": q0_clean,
            "queries": [q0_item],
            "model": model_name,
            "generation_latency_ms": round(latency_ms, 2),
            "status": "query_generation_unavailable",
            "warnings": [f"Invalid JSON returned from model: {exc}"],
            "cache_hit": False,
            "dropped_duplicate_count": 0,
        }

    if not isinstance(parsed_data, dict) or "queries" not in parsed_data:
        return {
            "original_question": q0_clean,
            "queries": [q0_item],
            "model": model_name,
            "generation_latency_ms": round(latency_ms, 2),
            "status": "query_generation_unavailable",
            "warnings": ["Response JSON missing 'queries' key"],
            "cache_hit": False,
            "dropped_duplicate_count": 0,
        }

    raw_queries = parsed_data["queries"]
    if not isinstance(raw_queries, list):
        return {
            "original_question": q0_clean,
            "queries": [q0_item],
            "model": model_name,
            "generation_latency_ms": round(latency_ms, 2),
            "status": "query_generation_unavailable",
            "warnings": ["'queries' field is not a list"],
            "cache_hit": False,
            "dropped_duplicate_count": 0,
        }

    # Filtering & Validation
    seen_dedup_keys = {_normalize_for_dedup(q0_clean)}
    q0_articles = _extract_articles_numbers(q0_clean)
    valid_variants: List[Dict[str, Any]] = []
    dropped_dup_count = 0

    valid_focuses = {"exact_legal_terms", "paraphrase", "missing_aspect"}

    for item in raw_queries:
        if len(valid_variants) >= mq_count:
            break

        if not isinstance(item, dict):
            continue

        text_raw = str(item.get("text", "")).strip()
        text_clean = unicodedata.normalize("NFC", text_raw)

        if not text_clean or len(text_clean) > mq_max_chars:
            continue

        # Check deduplication
        dedup_k = _normalize_for_dedup(text_clean)
        if dedup_k in seen_dedup_keys:
            dropped_dup_count += 1
            continue

        # Check legal references inventing (prohibit inventing new articles not present in Q0)
        item_articles = _extract_articles_numbers(text_clean)
        invented_articles = item_articles - q0_articles
        if invented_articles:
            warnings.append(
                f"Dropped variant containing invented article(s) {invented_articles}: '{text_clean}'"
            )
            continue

        # Focus normalization
        focus_raw = str(item.get("focus", "")).strip()
        focus_clean = focus_raw if focus_raw in valid_focuses else "paraphrase"

        seen_dedup_keys.add(dedup_k)
        valid_variants.append({"text": text_clean, "focus": focus_clean})

    # Assemble final query list
    final_queries = [q0_item]
    for idx, v in enumerate(valid_variants, start=1):
        final_queries.append({
            "query_id": f"Q{idx}",
            "text": v["text"],
            "origin": "generated",
            "focus": v["focus"],
        })

    result = {
        "original_question": q0_clean,
        "queries": final_queries,
        "model": model_name,
        "generation_latency_ms": round(latency_ms, 2),
        "status": "ready",
        "warnings": warnings,
        "cache_hit": False,
        "dropped_duplicate_count": dropped_dup_count,
    }

    # Store in process cache
    _QUERY_EXPANSION_CACHE[cache_key] = json.loads(json.dumps(result))

    return result


# ──────────────────────────────────────────────────────────────────────────────
# § 10. Per-Query Hybrid Retrieval & Cross-Query RRF (Bước 05)
# ──────────────────────────────────────────────────────────────────────────────

def _default_hybrid_retriever(
    query_text: str,
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Gọi hybrid_retrieval từ advanced_rag.py snapshot của Buổi 09."""
    import sys
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    import rag
    import advanced_rag

    if not rag.DEFAULT_INPUT_DIR.exists() and BUOI05_CHUNKS_DIR.exists():
        rag.DEFAULT_INPUT_DIR = BUOI05_CHUNKS_DIR

    hits, trace = advanced_rag.hybrid_retrieval(
        question=query_text,
        strategy="hierarchical",
    )
    return hits, trace


def retrieve_multi_query_children(
    question: str,
    config: Optional[Dict[str, Any]] = None,
    query_generator_fn: Optional[Any] = None,
    hybrid_retriever_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Thực hiện per-query hybrid retrieval cho toàn bộ query variants (Q0..Qn)
    và hợp nhất kết quả bằng Cross-Query RRF.

    Args:
        question: Câu hỏi gốc người dùng
        config: Config dict
        query_generator_fn: Optional mock generator cho multi-query expansion
        hybrid_retriever_fn: Optional mock retriever(query_text, config) -> (hits, trace) hoặc hits list

    Returns:
        Dict chứa merged child hits và full execution trace.
    """
    import time

    if config is None:
        config = load_hierarchical_config()

    orig_weight = config.get("multi_query_original_weight", 1.5)
    var_weight = config.get("multi_query_variant_weight", 1.0)
    rrf_k = config.get("multi_query_rrf_k", 60)
    per_q_candidates = config.get("per_query_candidates", 12)

    # 1. Multi-query Expansion (Bước 04)
    t0_gen = time.perf_counter()
    query_set = generate_multi_queries(
        question=question,
        config=config,
        query_generator_fn=query_generator_fn,
    )
    gen_latency_ms = query_set.get(
        "generation_latency_ms", round((time.perf_counter() - t0_gen) * 1000, 2)
    )

    queries = query_set.get("queries", [])
    if not queries:
        q0_clean = unicodedata.normalize("NFC", question.strip())
        queries = [
            {
                "query_id": "Q0",
                "text": q0_clean,
                "origin": "original",
                "focus": "original_intent",
            }
        ]

    # 2. Per-Query Hybrid Retrieval
    retriever = hybrid_retriever_fn or _default_hybrid_retriever

    query_hits_map: Dict[str, List[Dict[str, Any]]] = {}
    query_traces_map: Dict[str, Dict[str, Any]] = {}
    per_query_latency: Dict[str, float] = {}
    result_count_per_query: Dict[str, int] = {}
    query_errors: Dict[str, str] = {}

    executed_count = 0
    failed_count = 0

    for q_item in queries:
        qid = q_item["query_id"]
        qtext = q_item["text"]
        t0_q = time.perf_counter()

        try:
            res = retriever(qtext, config)
            t_q_lat = round((time.perf_counter() - t0_q) * 1000, 2)
            per_query_latency[qid] = t_q_lat

            if isinstance(res, tuple):
                hits, inner_trace = res
            else:
                hits, inner_trace = res, {}

            # Chuẩn hóa hits: giới hạn tối đa per_query_candidates
            clean_hits: List[Dict[str, Any]] = []
            for h in hits[:per_q_candidates]:
                cid = str(h.get("child_id") or h.get("chunk_id") or "").strip()
                if not cid:
                    continue
                hit_item = {
                    "child_id": cid,
                    "text": h.get("text", ""),
                    "source": h.get("source", ""),
                    "page_start": h.get("page_start", 1),
                    "page_end": h.get("page_end", 1),
                    "bm25_rank": h.get("bm25_rank"),
                    "semantic_rank": h.get("semantic_rank"),
                    "inner_rrf_rank": h.get("inner_rrf_rank", h.get("rank")),
                }
                clean_hits.append(hit_item)

            query_hits_map[qid] = clean_hits
            query_traces_map[qid] = inner_trace
            result_count_per_query[qid] = len(clean_hits)
            executed_count += 1

        except Exception as exc:
            t_q_lat = round((time.perf_counter() - t0_q) * 1000, 2)
            per_query_latency[qid] = t_q_lat
            failed_count += 1
            err_str = str(exc)
            query_errors[qid] = err_str

            if qid == "Q0":
                raise RuntimeError(f"Q0 hybrid retrieval failed: {err_str}") from exc

            query_hits_map[qid] = []
            result_count_per_query[qid] = 0

    # Determine overall status
    if query_set.get("status") == "query_generation_unavailable":
        overall_status = "query_generation_unavailable"
    elif failed_count > 0 or (
        len(queries) > 1
        and any(result_count_per_query.get(q["query_id"], 0) == 0 for q in queries[1:])
    ):
        overall_status = "multi_query_partial"
    else:
        overall_status = "ready"

    # 3. Cross-Query RRF Fusion & Union
    t0_fusion = time.perf_counter()

    child_meta: Dict[str, Dict[str, Any]] = {}
    child_per_q_ranks: Dict[str, Dict[str, int]] = {}
    child_per_q_trace: Dict[str, Dict[str, Any]] = {}
    child_weights: Dict[str, float] = {}

    for q_item in queries:
        qid = q_item["query_id"]
        w = (
            orig_weight
            if q_item.get("origin") == "original" or qid == "Q0"
            else var_weight
        )

        hits = query_hits_map.get(qid, [])
        for rank_idx, hit in enumerate(hits, start=1):
            cid = hit["child_id"]

            # Metadata validation contract
            if cid in child_meta:
                existing = child_meta[cid]
                if (
                    existing["text"] != hit["text"]
                    or existing["source"] != hit["source"]
                    or existing["page_start"] != hit["page_start"]
                    or existing["page_end"] != hit["page_end"]
                ):
                    raise ValueError(
                        f"Metadata mismatch for child_id '{cid}': "
                        f"First seen ({existing['source']}, p{existing['page_start']}), "
                        f"conflict with ({hit['source']}, p{hit['page_start']})"
                    )
            else:
                child_meta[cid] = {
                    "text": hit["text"],
                    "source": hit["source"],
                    "page_start": hit["page_start"],
                    "page_end": hit["page_end"],
                }

            if cid not in child_per_q_ranks:
                child_per_q_ranks[cid] = {}
                child_per_q_trace[cid] = {}
                child_weights[cid] = 0.0

            child_per_q_ranks[cid][qid] = rank_idx
            child_per_q_trace[cid][qid] = {
                "bm25_rank": hit.get("bm25_rank"),
                "semantic_rank": hit.get("semantic_rank"),
                "inner_rrf_rank": hit.get("inner_rrf_rank"),
            }

            rrf_term = w / (rrf_k + rank_idx)
            child_weights[cid] += rrf_term

    candidates: List[Dict[str, Any]] = []
    for cid, meta in child_meta.items():
        pq_ranks = child_per_q_ranks[cid]
        all_qids = [q["query_id"] for q in queries]
        supp_qids = [q for q in all_qids if q in pq_ranks]
        best_rank = min(pq_ranks.values()) if pq_ranks else 999999
        mq_rrf_score = child_weights[cid]

        candidates.append({
            "child_id": cid,
            "text": meta["text"],
            "source": meta["source"],
            "page_start": meta["page_start"],
            "page_end": meta["page_end"],
            "multi_query_rrf_score": round(mq_rrf_score, 6),
            "support_query_count": len(supp_qids),
            "support_query_ids": supp_qids,
            "per_query_ranks": pq_ranks,
            "per_query_trace": child_per_q_trace[cid],
            "_best_rank": best_rank,
        })

    # Sort 4 criteria:
    # 1. multi_query_rrf_score desc
    # 2. support_query_count desc
    # 3. best_rank asc
    # 4. child_id asc
    candidates.sort(
        key=lambda c: (
            -c["multi_query_rrf_score"],
            -c["support_query_count"],
            c["_best_rank"],
            c["child_id"],
        )
    )

    for rank_idx, cand in enumerate(candidates, start=1):
        cand["multi_query_rank"] = rank_idx
        del cand["_best_rank"]

    fusion_latency_ms = round((time.perf_counter() - t0_fusion) * 1000, 2)

    overlap_dist: Dict[str, int] = {}
    for cand in candidates:
        cnt = cand["support_query_count"]
        key = f"hit_by_{cnt}"
        overlap_dist[key] = overlap_dist.get(key, 0) + 1

    trace = {
        "query_count": {
            "requested": len(queries),
            "valid": len(queries),
            "executed": executed_count,
            "failed": failed_count,
        },
        "generation_latency_ms": gen_latency_ms,
        "per_query_retrieval_latency_ms": per_query_latency,
        "result_count_per_query": result_count_per_query,
        "union_child_count": len(candidates),
        "overlap_distribution": overlap_dist,
        "fusion_latency_ms": fusion_latency_ms,
        "query_expansion_call_count": 0 if query_set.get("cache_hit") else 1,
        "query_errors": query_errors,
        "warnings": query_set.get("warnings", []),
    }

    return {
        "status": overall_status,
        "original_question": query_set.get("original_question", question),
        "queries": queries,
        "merged_child_hits": candidates,
        "warnings": query_set.get("warnings", []),
        "trace": trace,
    }


# ──────────────────────────────────────────────────────────────────────────────
# § 11. Parent Document Resolution & Score Aggregation (Bước 06)
# ──────────────────────────────────────────────────────────────────────────────

def retrieve_parent_documents(
    question: str,
    mode: str = "multi_parent",
    config: Optional[Dict[str, Any]] = None,
    storage_dir: Optional[Path] = None,
    registry_override: Optional[Tuple[List[Dict], List[Dict], Dict]] = None,
    query_generator_fn: Optional[Any] = None,
    hybrid_retriever_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Chuyển đổi Fused Child Hits sang Parent Documents, tính Parent Score,
    áp dụng Context Budget và trả về danh sách Parent Candidates.

    Args:
        question: Câu hỏi người dùng
        mode: "single_parent" hoặc "multi_parent"
        config: Dict config từ load_hierarchical_config()
        storage_dir: Đường dẫn thư mục lưu hierarchy registry
        registry_override: Optional (children, parents, manifest) cho testing mock
        query_generator_fn: Optional mock generator cho multi-query
        hybrid_retriever_fn: Optional mock hybrid retriever

    Returns:
        Dict kết quả theo Parent Candidates contract.
    """
    import time

    if config is None:
        config = load_hierarchical_config()

    if mode not in ("single_parent", "multi_parent"):
        raise ValueError(
            f"Mode '{mode}' không hợp lệ cho Bước 06. Nhận: single_parent hoặc multi_parent."
        )

    # 1. Load Registry & Verify Preconditions
    t0_map = time.perf_counter()
    if registry_override is not None:
        children_reg, parents_reg, manifest_reg = registry_override
    else:
        status_info = hierarchy_status(storage_dir)
        if not status_info.get("registry_exists"):
            return {
                "status": "hierarchy_not_ready",
                "original_question": question,
                "mode": mode,
                "accepted_parents": [],
                "dropped_parents": [],
                "warnings": [
                    "Hierarchy registry không tồn tại. Hãy chạy build-hierarchy trước."
                ],
                "trace": {},
            }
        try:
            children_reg, parents_reg, manifest_reg = load_hierarchy(storage_dir)
        except Exception as exc:
            return {
                "status": "hierarchy_not_ready",
                "original_question": question,
                "mode": mode,
                "accepted_parents": [],
                "dropped_parents": [],
                "warnings": [f"Lỗi nạp hierarchy registry: {exc}"],
                "trace": {},
            }

        # Verify manifest config identity
        current_cfg_id = _config_identity(config)
        stored_cfg_id = manifest_reg.get("config_identity")
        if stored_cfg_id and stored_cfg_id != current_cfg_id:
            return {
                "status": "hierarchy_not_ready",
                "original_question": question,
                "mode": mode,
                "accepted_parents": [],
                "dropped_parents": [],
                "warnings": [
                    f"Hierarchy manifest stale! Stored config identity '{stored_cfg_id}' "
                    f"khác current config identity '{current_cfg_id}'. Chạy build-hierarchy lại."
                ],
                "trace": {},
            }

    # Registry lookup tables
    child_map = {c["child_id"]: c for c in children_reg}
    parent_map = {p["parent_id"]: p for p in parents_reg}

    # Config parameters
    parent_child_lim = config.get("parent_score_child_limit", 3)
    parent_rrf_k = config.get("parent_rrf_k", 60)
    parent_candidates_lim = config.get("parent_candidates", 10)
    total_ctx_max = config.get("total_context_max_chars", 16000)

    # 2. Per-Query & Multi-Query Child Retrieval (Bước 05)
    effective_config = dict(config)
    if mode == "single_parent":
        effective_config["multi_query_count"] = 0

    child_retrieval_res = retrieve_multi_query_children(
        question=question,
        config=effective_config,
        query_generator_fn=query_generator_fn
        if mode == "multi_parent"
        else (lambda p, c: '{"queries":[]}'),
        hybrid_retriever_fn=hybrid_retriever_fn,
    )

    if child_retrieval_res.get("status") == "failed":
        raise RuntimeError(f"Child retrieval failed for mode {mode}")

    fused_child_hits = child_retrieval_res.get("merged_child_hits", [])
    child_retrieval_trace = child_retrieval_res.get("trace", {})

    # 3. Child to Parent Resolution & Grouping
    parent_hits_groups: Dict[str, List[Dict[str, Any]]] = {}
    child_to_parent_table: Dict[str, str] = {}

    for hit in fused_child_hits:
        cid = hit["child_id"]
        if cid not in child_map:
            raise ValueError(
                f"Child ID '{cid}' không tồn tại trong hierarchy registry"
            )
        child_info = child_map[cid]
        pid = child_info.get("parent_id")
        if not pid or pid not in parent_map:
            raise ValueError(
                f"Parent ID '{pid}' cho child '{cid}' không tồn tại trong hierarchy registry"
            )

        child_to_parent_table[cid] = pid
        parent_hits_groups.setdefault(pid, []).append(hit)

    # 4. Parent Score Aggregation
    raw_parent_candidates: List[Dict[str, Any]] = []

    for pid, hits_in_parent in parent_hits_groups.items():
        p_info = parent_map[pid]

        sorted_hits = sorted(hits_in_parent, key=lambda h: h["multi_query_rank"])

        anchor_child_id = sorted_hits[0]["child_id"]
        best_child_rank = sorted_hits[0]["multi_query_rank"]

        scoring_hits = sorted_hits[:parent_child_lim]
        scoring_child_ids = [h["child_id"] for h in scoring_hits]
        supporting_child_ids = [h["child_id"] for h in sorted_hits]

        all_qids = [q["query_id"] for q in child_retrieval_res.get("queries", [])]
        supp_qids_set = set()
        for h in sorted_hits:
            supp_qids_set.update(h.get("support_query_ids", []))
        ordered_supp_qids = [q for q in all_qids if q in supp_qids_set]

        p_score = sum(
            1.0 / (parent_rrf_k + h["multi_query_rank"]) for h in scoring_hits
        )

        is_ambiguous = any(
            child_map[h["child_id"]].get("ambiguous", False) for h in sorted_hits
        )
        p_warnings = list(p_info.get("warnings", []))

        raw_parent_candidates.append({
            "parent_id": pid,
            "source": p_info["source"],
            "page_start": p_info["page_start"],
            "page_end": p_info["page_end"],
            "structural_path": {
                "chapter": child_map[anchor_child_id]["structural_path"].get(
                    "chapter"
                ),
                "article": p_info.get("article_key"),
                "clause": child_map[anchor_child_id]["structural_path"].get(
                    "clause"
                ),
                "point": child_map[anchor_child_id]["structural_path"].get("point"),
            },
            "text": p_info["text"],
            "parent_rrf_score": round(p_score, 6),
            "anchor_child_id": anchor_child_id,
            "scoring_child_ids": scoring_child_ids,
            "supporting_child_ids": supporting_child_ids,
            "support_query_ids": ordered_supp_qids,
            "best_child_rank": best_child_rank,
            "ambiguous": is_ambiguous,
            "warnings": p_warnings,
        })

    # 5. Sort Parent Candidates (4 criteria)
    # 1. parent_rrf_score desc
    # 2. len(support_query_ids) desc
    # 3. best_child_rank asc
    # 4. parent_id asc
    raw_parent_candidates.sort(
        key=lambda p: (
            -p["parent_rrf_score"],
            -len(p["support_query_ids"]),
            p["best_child_rank"],
            p["parent_id"],
        )
    )

    for rank_idx, p_cand in enumerate(raw_parent_candidates, start=1):
        p_cand["parent_rank"] = rank_idx

    cand_limited_parents = raw_parent_candidates[:parent_candidates_lim]
    dropped_by_cand_limit = raw_parent_candidates[parent_candidates_lim:]

    # 6. Context Budget Truncation
    accepted_parents: List[Dict[str, Any]] = []
    dropped_by_budget: List[Dict[str, Any]] = []
    current_ctx_chars = 0

    for idx, p_cand in enumerate(cand_limited_parents):
        p_len = len(p_cand["text"])
        if current_ctx_chars + p_len <= total_ctx_max:
            accepted_parents.append(p_cand)
            current_ctx_chars += p_len
        else:
            if idx == 0:
                p_cand["warnings"].append(
                    f"oversized_first_parent_kept_exceeding_budget: ({p_len} chars > max {total_ctx_max})"
                )
                accepted_parents.append(p_cand)
                current_ctx_chars += p_len
            else:
                dropped_by_budget.append(p_cand)

    mapping_latency_ms = round((time.perf_counter() - t0_map) * 1000, 2)

    input_child_chars = sum(len(h.get("text", "")) for h in fused_child_hits)
    exp_factor = (
        round(current_ctx_chars / input_child_chars, 2)
        if input_child_chars > 0
        else 0.0
    )

    child_count_per_p = {
        p["parent_id"]: len(p["supporting_child_ids"]) for p in accepted_parents
    }

    parent_score_comps = {
        p["parent_id"]: {
            "parent_rrf_score": p["parent_rrf_score"],
            "scoring_child_ranks": [
                next(
                    h["multi_query_rank"]
                    for h in fused_child_hits
                    if h["child_id"] == cid
                )
                for cid in p["scoring_child_ids"]
            ],
        }
        for p in accepted_parents
    }

    all_dropped = dropped_by_cand_limit + dropped_by_budget

    trace = {
        "mode": mode,
        "input_child_hit_count": len(fused_child_hits),
        "unique_parent_count": len(parent_hits_groups),
        "child_count_per_parent": child_count_per_p,
        "child_to_parent_mapping": child_to_parent_table,
        "parent_score_components": parent_score_comps,
        "parents_dropped": {
            "by_candidate_limit": len(dropped_by_cand_limit),
            "by_context_budget": len(dropped_by_budget),
            "total_dropped": len(all_dropped),
        },
        "child_chars": input_child_chars,
        "expanded_parent_chars": current_ctx_chars,
        "context_expansion_factor": exp_factor,
        "ambiguous_parent_count": sum(1 for p in accepted_parents if p["ambiguous"]),
        "mapping_and_aggregation_latency_ms": mapping_latency_ms,
        "child_retrieval_trace": child_retrieval_trace,
    }

    overall_status = child_retrieval_res.get("status", "ready")

    return {
        "status": overall_status,
        "original_question": question,
        "mode": mode,
        "accepted_parents": accepted_parents,
        "dropped_parents": all_dropped,
        "warnings": child_retrieval_res.get("warnings", []),
        "trace": trace,
    }


# ──────────────────────────────────────────────────────────────────────────────
# § 12. CLI
# ──────────────────────────────────────────────────────────────────────────────

def _run_audit_cli(args: argparse.Namespace) -> int:
    try:
        config = load_hierarchical_config()
    except ValueError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1

    input_dir = Path(args.input_dir) if args.input_dir else None
    try:
        report = run_hierarchy_audit(input_dir, config)
    except ValueError as exc:
        print(f"AUDIT ERROR: {exc}", file=sys.stderr)
        return 1

    ls = report["load_stats"]
    tot = report["totals"]
    dist = report["parent_size_distribution"]

    print("\n══ Hierarchy Audit ══════════════════════════════════════════")
    print(f"  Files read   : {ls['files']}")
    print(f"  Total records: {ls['total_records']}")
    print(f"  Valid chunks : {ls['valid']}  (sources: {ls.get('sources', '?')})")
    print(f"\n── Resolution Breakdown ─────────────────────────────────────")
    for method, cnt in sorted(report["resolution_breakdown"].items()):
        print(f"  {method:<20}: {cnt}")
    print(f"\n── Totals ───────────────────────────────────────────────────")
    print(f"  Children     : {tot['children']}")
    print(f"  Parents      : {tot['parents']}")
    print(f"  Ambiguous    : {tot['ambiguous_children']}")
    print(f"  Orphans      : {tot['orphan_children']}")
    print(f"  Parent warns : {tot['warned_parents']}")
    print(f"\n── Parent Size Distribution (chars) ─────────────────────────")
    for k, v in dist.items():
        print(f"  {k:<8}: {v}")
    if report["ambiguous_examples"]:
        print(f"\n── Ambiguous Examples ────────────────────────────────────")
        for ex in report["ambiguous_examples"]:
            print(f"  {ex['child_id']}: {ex['warnings']}")
    if report["parent_warning_examples"]:
        print(f"\n── Parent Warnings ──────────────────────────────────────")
        for ex in report["parent_warning_examples"]:
            print(f"  {ex['parent_id']}: {ex['warnings']}")
    print("═" * 60)
    return 0


def _run_build_cli(args: argparse.Namespace) -> int:
    try:
        config = load_hierarchical_config()
    except ValueError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1

    input_dir = Path(args.input_dir) if args.input_dir else None
    storage_dir = Path(args.storage_dir) if args.storage_dir else None

    print("Loading chunks…")
    try:
        chunks, load_stats, fingerprints = load_hierarchical_chunks(input_dir)
    except ValueError as exc:
        print(f"LOAD ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"  {load_stats['valid']} hierarchical chunks from {load_stats['sources']} sources")
    print("Resolving hierarchy…")
    children = resolve_hierarchy(chunks)

    print("Building parents…")
    parents, children = build_parents(children, config)
    print(f"  {len(parents)} parent windows built")

    print("Saving registry…")
    try:
        manifest = save_hierarchy(children, parents, load_stats, fingerprints, config, storage_dir)
    except Exception as exc:
        print(f"SAVE ERROR: {exc}", file=sys.stderr)
        return 1

    counts = manifest["counts"]
    print("\nBUILD COMPLETE")
    print(f"  children       : {counts['children']}")
    print(f"  parents        : {counts['parents']}")
    print(f"  ambiguous      : {counts['ambiguous_children']}")
    print(f"  orphans        : {counts['orphan_children']}")
    print(f"  parent warnings: {counts['warned_parents']}")
    print(f"  built_at       : {manifest['built_at']}")
    return 0


def _run_status_cli(args: argparse.Namespace) -> int:
    storage_dir = Path(args.storage_dir) if args.storage_dir else None
    status = hierarchy_status(storage_dir)

    print("\n══ Hierarchy Status (read-only) ══════════════════════════════")
    print(f"  registry_exists : {status['registry_exists']}")
    print(f"  storage_dir     : {status['storage_dir']}")
    for fname, exists in status.get("files", {}).items():
        mark = "✓" if exists else "✗"
        print(f"  {mark} {fname}")
    if status.get("manifest"):
        m = status["manifest"]
        c = m.get("counts", {})
        print(f"  built_at        : {m.get('built_at', '?')}")
        print(f"  schema_version  : {m.get('schema_version', '?')}")
        print(f"  children        : {c.get('children', '?')}")
        print(f"  parents         : {c.get('parents', '?')}")
        print(f"  ambiguous       : {c.get('ambiguous_children', '?')}")
        print(f"  orphans         : {c.get('orphan_children', '?')}")
    elif status.get("error"):
        print(f"  error: {status['error']}")
    print("═" * 60)
    return 0


def _run_expand_query_cli(args: argparse.Namespace) -> int:
    try:
        config = load_hierarchical_config()
    except ValueError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1

    question = args.question
    print(f"Expanding query: '{question}'...\n")
    res = generate_multi_queries(question, config=config)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def _run_multi_child_cli(args: argparse.Namespace) -> int:
    try:
        config = load_hierarchical_config()
    except ValueError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1

    question = args.question
    print(f"Running Multi-Query Child Retrieval for: '{question}'...\n")
    try:
        res = retrieve_multi_query_children(question, config=config)
    except Exception as exc:
        print(f"RETRIEVAL ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Status  : {res['status']}")
    print(f"Queries : {len(res['queries'])}")
    for q in res['queries']:
        print(f"  [{q['query_id']}] ({q['origin']}/{q['focus']}): {q['text']}")

    hits = res["merged_child_hits"]
    print(f"\nMerged Child Hits ({len(hits)} total):\n")
    print(f"{'Rank':<6} {'Child ID':<35} {'Supp':<6} {'QIDs':<16} {'MQ-RRF Score':<14} {'Per-Query Ranks'}")
    print("-" * 100)
    for h in hits:
        qids_str = ",".join(h["support_query_ids"])
        pq_str = str(h["per_query_ranks"])
        print(f"{h['multi_query_rank']:<6} {h['child_id']:<35} {h['support_query_count']:<6} {qids_str:<16} {h['multi_query_rrf_score']:<14.6f} {pq_str}")

    print("\nTrace summary:")
    print(json.dumps(res["trace"], ensure_ascii=False, indent=2))
    return 0


def _run_parent_retrieve_cli(args: argparse.Namespace) -> int:
    try:
        config = load_hierarchical_config()
    except ValueError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1

    question = args.question
    mode = args.mode
    print(f"Running Parent Retrieval ({mode}) for: '{question}'...\n")
    try:
        res = retrieve_parent_documents(question, mode=mode, config=config)
    except Exception as exc:
        print(f"PARENT RETRIEVAL ERROR: {exc}", file=sys.stderr)
        return 1

    if res["status"] == "hierarchy_not_ready":
        print(f"STATUS: {res['status']}")
        for w in res.get("warnings", []):
            print(f"WARNING: {w}")
        return 1

    print(f"Status           : {res['status']}")
    print(f"Accepted Parents : {len(res['accepted_parents'])}")
    print(f"Dropped Parents  : {len(res['dropped_parents'])}\n")

    for p in res["accepted_parents"]:
        art = p["structural_path"].get("article") or "Document Block"
        print(f"Parent [{p['parent_id']}] (Rank: {p['parent_rank']}, Score: {p['parent_rrf_score']:.6f}, Source: {p['source']}, Chars: {len(p['text'])})")
        print(f"  Article: {art}")
        for cid in p["supporting_child_ids"]:
            is_scoring = " (Scoring)" if cid in p["scoring_child_ids"] else ""
            is_anchor = " [Anchor]" if cid == p["anchor_child_id"] else ""
            print(f"  ├── Child [{cid}]{is_anchor}{is_scoring}")

    print("\nTrace Summary:")
    print(json.dumps(res["trace"], ensure_ascii=False, indent=2))
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# § 13. Parent Reranking, Evidence Gate & Answer Generation (Bước 07)
# ──────────────────────────────────────────────────────────────────────────────

def rerank_parent_candidates(
    original_question: str,
    parent_candidates: List[Dict[str, Any]],
    config: Dict[str, Any],
    reranker_fn: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Cross-encoder rerank trên danh sách parent candidates.

    Luôn sử dụng cặp (original_question, parent_text). Không rerank bằng variant query.
    """
    import math

    if not parent_candidates:
        return []

    min_score = config.get("rerank_min_score", 0.35)

    if reranker_fn is not None:
        scores = reranker_fn(original_question, [p["text"] for p in parent_candidates])
    else:
        import sys
        if str(BASE_DIR) not in sys.path:
            sys.path.insert(0, str(BASE_DIR))
        import advanced_rag

        status = advanced_rag.advanced_status()
        if not status.get("reranker_cache_exists"):
            raise ValueError("Reranker model cache not found. Run script download model first.")

        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        model_name = status["reranker_model"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()

        texts = [p["text"] for p in parent_candidates]
        inputs = tokenizer([original_question] * len(texts), texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(-1).float()
        scores = logits.tolist()
        if isinstance(scores, float):
            scores = [scores]

    reranked_parents: List[Dict[str, Any]] = []
    for cand, sc in zip(parent_candidates, scores):
        p_item = dict(cand)
        if 0.0 <= sc <= 1.0:
            sig_score = sc
            raw_logit = sc
        else:
            raw_logit = sc
            sig_score = 1.0 / (1.0 + math.exp(-sc))

        p_item["parent_rerank_raw_score"] = round(raw_logit, 6)
        p_item["parent_rerank_score"] = round(sig_score, 6)
        p_item["accepted"] = sig_score >= min_score
        reranked_parents.append(p_item)

    # Sort: score desc, parent_rank asc, parent_id asc
    reranked_parents.sort(
        key=lambda p: (
            -p["parent_rerank_score"],
            p["parent_rank"],
            p["parent_id"],
        )
    )

    for idx, p in enumerate(reranked_parents, start=1):
        p["parent_rerank_rank"] = idx
        p["parent_rank_change"] = p["parent_rank"] - idx

    return reranked_parents


def _default_answer_generator(
    prompt: str,
    config: Dict[str, Any],
) -> str:
    """Gọi Gemini API thực tế để sinh answer từ accepted evidence."""
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY bị thiếu. Hãy điền API key vào .env")

    client = genai.Client(api_key=api_key)
    model_name = config.get("gemini_generation_model", "gemini-3.5-flash-lite")

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    if not response.text:
        raise ValueError("Gemini API trả về answer rỗng")
    return response.text


def _generate_answer_from_evidence(
    question: str,
    accepted_evidence: List[Dict[str, Any]],
    config: Dict[str, Any],
    answer_generator_fn: Optional[Any] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Xây dựng prompt và gọi Gemini sinh câu trả lời đính kèm citations [P1], [P2]..."""
    ev_blocks = []
    for idx, ev in enumerate(accepted_evidence, start=1):
        ev_id = f"P{idx}"
        art = ev.get("structural_path", {}).get("article") or "Văn bản"
        src = ev.get("source", "")
        pages = f"tr. {ev.get('page_start')}-{ev.get('page_end')}"
        ev_blocks.append(f"[{ev_id}] Nguồn: {src}, {art}, {pages}\n{ev['text']}")

    evidence_str = "\n\n".join(ev_blocks)

    prompt = f"""Bạn là trợ lý tư vấn pháp luật ngân hàng Việt Nam.
Nhiệm vụ: Trả lời câu hỏi gốc của người dùng DỰA HOÀN TOÀN VÀO CÁC TÀI LIỆU DẪN CHỨNG DƯỚI ĐÂY.

CÂU HỎI GỐC: "{question}"

TÀI LIỆU DẪN CHỨNG (ACCEPTED EVIDENCE):
{evidence_str}

QUY TẮC BẮT BUỘC:
1. Chỉ sử dụng thông tin trong TÀI LIỆU DẪN CHỨNG để trả lời. Không tự suy diễn hay tư vấn ngoài căn cứ pháp lý.
2. Mỗi khẳng định, nhận định hoặc quy định được trình bày BẮT BUỘC phải đính kèm ký hiệu dẫn chứng tương ứng ở cuối câu (ví dụ: [P1], [P2]).
3. Nếu tài liệu dẫn chứng có điểm mâu thuẫn hoặc chưa rõ ràng (ambiguous), phải nêu rõ giới hạn đó cho người dùng.
4. Tuyệt đối không bịa đặt nguồn, số trang, số Điều/Khoản hay mã định danh không có trong dẫn chứng.
"""

    gen_fn = answer_generator_fn or _default_answer_generator
    try:
        ans = gen_fn(prompt, config)
        return ans.strip(), None
    except Exception as exc:
        return None, f"Lỗi sinh câu trả lời từ Gemini: {exc}"


def query_hierarchical_rag(
    question: str,
    mode: str = "multi_parent",
    strategy: str = "hierarchical",
    config: Optional[Dict[str, Any]] = None,
    storage_dir: Optional[Path] = None,
    registry_override: Optional[Tuple[List[Dict], List[Dict], Dict]] = None,
    query_generator_fn: Optional[Any] = None,
    hybrid_retriever_fn: Optional[Any] = None,
    reranker_fn: Optional[Any] = None,
    answer_generator_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """End-to-end RAG pipeline cho Buổi 09 hỗ trợ cả 4 modes.

    Modes:
      - single_flat     : Q0 → hybrid → rerank child
      - multi_flat      : Q0 + variants → per-query hybrid → MQ-RRF → rerank child
      - single_parent   : Q0 → hybrid → child-to-parent → parent aggregation → rerank parent
      - multi_parent    : Q0 + variants → per-query hybrid → MQ-RRF → child-to-parent → parent aggregation → rerank parent
    """
    import time
    t0_start = time.perf_counter()

    if config is None:
        config = load_hierarchical_config()

    q0_clean = unicodedata.normalize("NFC", question.strip())

    VALID_MODES = {"single_flat", "multi_flat", "single_parent", "multi_parent"}
    if mode not in VALID_MODES:
        raise ValueError(f"mode phải là một trong {sorted(VALID_MODES)}, nhận: '{mode}'")

    final_top_k = config.get("final_parent_top_k", 3)
    min_score = config.get("rerank_min_score", 0.35)

    gen_calls_count = 0
    emb_calls_count = 0
    warnings_list: List[str] = []
    stage_latencies: Dict[str, float] = {}

    # ── PARENT MODES (single_parent, multi_parent) ───────────────────────────
    if mode in ("single_parent", "multi_parent"):
        t0_ret = time.perf_counter()
        ret_res = retrieve_parent_documents(
            question=q0_clean,
            mode=mode,
            config=config,
            storage_dir=storage_dir,
            registry_override=registry_override,
            query_generator_fn=query_generator_fn,
            hybrid_retriever_fn=hybrid_retriever_fn,
        )
        stage_latencies["retrieval_and_aggregation"] = round((time.perf_counter() - t0_ret) * 1000, 2)

        if ret_res.get("status") == "hierarchy_not_ready":
            return ret_res

        if mode == "multi_parent" and ret_res.get("status") != "query_generation_unavailable":
            gen_calls_count += 1  # Multi-query expansion API call count

        parent_candidates = ret_res.get("accepted_parents", [])
        warnings_list.extend(ret_res.get("warnings", []))

        if not parent_candidates:
            return {
                "status": "insufficient_evidence",
                "mode": mode,
                "original_question": q0_clean,
                "query_set": ret_res.get("queries"),
                "child_hits": [],
                "parent_candidates": [],
                "accepted_evidence": [],
                "answer": None,
                "citations": [],
                "stage_latencies_ms": stage_latencies,
                "api_call_counts": {"generation_calls": gen_calls_count, "embedding_calls": emb_calls_count},
                "warnings": warnings_list + ["Không tìm thấy parent document phù hợp"],
                "trace": ret_res.get("trace", {}),
            }

        # Rerank Parent Candidates
        t0_rr = time.perf_counter()
        try:
            reranked_parents = rerank_parent_candidates(
                original_question=q0_clean,
                parent_candidates=parent_candidates,
                config=config,
                reranker_fn=reranker_fn,
            )
            stage_latencies["rerank"] = round((time.perf_counter() - t0_rr) * 1000, 2)
        except Exception as exc:
            return {
                "status": "reranker_unavailable",
                "mode": mode,
                "original_question": q0_clean,
                "query_set": ret_res.get("queries"),
                "child_hits": [],
                "parent_candidates": parent_candidates,
                "accepted_evidence": [],
                "answer": None,
                "citations": [],
                "stage_latencies_ms": stage_latencies,
                "api_call_counts": {"generation_calls": gen_calls_count, "embedding_calls": emb_calls_count},
                "warnings": warnings_list + [f"Lỗi reranker: {exc}"],
                "trace": ret_res.get("trace", {}),
            }

        # Evidence Gate
        final_pool = reranked_parents[:final_top_k]
        accepted_evidence = [p for p in final_pool if p["accepted"]]

        if not accepted_evidence:
            return {
                "status": "insufficient_evidence",
                "mode": mode,
                "original_question": q0_clean,
                "query_set": ret_res.get("queries"),
                "child_hits": [],
                "parent_candidates": final_pool,
                "accepted_evidence": [],
                "answer": None,
                "citations": [],
                "stage_latencies_ms": stage_latencies,
                "api_call_counts": {"generation_calls": gen_calls_count, "embedding_calls": emb_calls_count},
                "warnings": warnings_list + [f"Không có parent nào đạt RERANK_MIN_SCORE ({min_score})"],
                "trace": ret_res.get("trace", {}),
            }

        # Format Citations
        citations = []
        for idx, ev in enumerate(accepted_evidence, start=1):
            evidence_id = f"P{idx}"
            ev["evidence_id"] = evidence_id
            cit_item = {
                "evidence_id": evidence_id,
                "parent_id": ev["parent_id"],
                "anchor_child_id": ev["anchor_child_id"],
                "supporting_child_ids": ev["supporting_child_ids"],
                "source": ev["source"],
                "page_start": ev["page_start"],
                "page_end": ev["page_end"],
                "structural_path": ev["structural_path"],
                "parent_rerank_score": ev["parent_rerank_score"],
                "ambiguous": ev["ambiguous"],
                "warnings": ev["warnings"],
            }
            citations.append(cit_item)

        # Answer Generation (Call 2)
        t0_ans = time.perf_counter()
        answer_text, gen_err = _generate_answer_from_evidence(
            question=q0_clean,
            accepted_evidence=accepted_evidence,
            config=config,
            answer_generator_fn=answer_generator_fn,
        )
        stage_latencies["answer_generation"] = round((time.perf_counter() - t0_ans) * 1000, 2)
        if gen_err:
            warnings_list.append(gen_err)
        else:
            gen_calls_count += 1

        overall_status = ret_res.get("status", "ready")

        return {
            "status": overall_status,
            "mode": mode,
            "original_question": q0_clean,
            "query_set": ret_res.get("queries"),
            "child_hits": [],
            "parent_candidates": final_pool,
            "accepted_evidence": accepted_evidence,
            "answer": answer_text,
            "citations": citations,
            "stage_latencies_ms": stage_latencies,
            "api_call_counts": {"generation_calls": gen_calls_count, "embedding_calls": emb_calls_count},
            "identities": {
                "config_identity": _config_identity(config),
                "schema_version": SCHEMA_VERSION,
            },
            "warnings": warnings_list,
            "trace": ret_res.get("trace", {}),
        }

    # ── FLAT MODES (single_flat, multi_flat) ──────────────────────────────────
    else:
        t0_ret = time.perf_counter()
        effective_config = dict(config)
        if mode == "single_flat":
            effective_config["multi_query_count"] = 0

        child_res = retrieve_multi_query_children(
            question=q0_clean,
            config=effective_config,
            query_generator_fn=query_generator_fn if mode == "multi_flat" else (lambda p, c: '{"queries":[]}'),
            hybrid_retriever_fn=hybrid_retriever_fn,
        )
        stage_latencies["retrieval_and_fusion"] = round((time.perf_counter() - t0_ret) * 1000, 2)

        if mode == "multi_flat" and child_res.get("status") != "query_generation_unavailable":
            gen_calls_count += 1

        fused_child_hits = child_res.get("merged_child_hits", [])
        warnings_list.extend(child_res.get("warnings", []))

        if not fused_child_hits:
            return {
                "status": "insufficient_evidence",
                "mode": mode,
                "original_question": q0_clean,
                "query_set": child_res.get("queries"),
                "child_hits": [],
                "parent_candidates": [],
                "accepted_evidence": [],
                "answer": None,
                "citations": [],
                "stage_latencies_ms": stage_latencies,
                "api_call_counts": {"generation_calls": gen_calls_count, "embedding_calls": emb_calls_count},
                "warnings": warnings_list + ["Không tìm thấy child chunk nào"],
                "trace": child_res.get("trace", {}),
            }

        # Rerank Flat Child Candidates bằng Q0
        t0_rr = time.perf_counter()
        try:
            import math
            if reranker_fn is not None:
                scores = reranker_fn(q0_clean, [c["text"] for c in fused_child_hits])
            else:
                import sys
                if str(BASE_DIR) not in sys.path:
                    sys.path.insert(0, str(BASE_DIR))
                import advanced_rag
                st = advanced_rag.advanced_status()
                if not st.get("reranker_cache_exists"):
                    raise ValueError("Reranker model cache not found.")
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                import torch
                tok = AutoTokenizer.from_pretrained(st["reranker_model"])
                mdl = AutoModelForSequenceClassification.from_pretrained(st["reranker_model"])
                mdl.eval()
                txts = [c["text"] for c in fused_child_hits]
                inps = tok([q0_clean] * len(txts), txts, padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    lg = mdl(**inps).logits.squeeze(-1).float()
                scores = lg.tolist()
                if isinstance(scores, float):
                    scores = [scores]

            reranked_children = []
            for cand, sc in zip(fused_child_hits, scores):
                c_item = dict(cand)
                sig_sc = sc if (0.0 <= sc <= 1.0) else (1.0 / (1.0 + math.exp(-sc)))
                c_item["child_rerank_score"] = round(sig_sc, 6)
                c_item["accepted"] = sig_sc >= min_score
                reranked_children.append(c_item)

            reranked_children.sort(key=lambda c: (-c["child_rerank_score"], c["multi_query_rank"]))
            for idx, c in enumerate(reranked_children, start=1):
                c["child_rerank_rank"] = idx

            stage_latencies["rerank"] = round((time.perf_counter() - t0_rr) * 1000, 2)
        except Exception as exc:
            return {
                "status": "reranker_unavailable",
                "mode": mode,
                "original_question": q0_clean,
                "query_set": child_res.get("queries"),
                "child_hits": fused_child_hits,
                "parent_candidates": [],
                "accepted_evidence": [],
                "answer": None,
                "citations": [],
                "stage_latencies_ms": stage_latencies,
                "api_call_counts": {"generation_calls": gen_calls_count, "embedding_calls": emb_calls_count},
                "warnings": warnings_list + [f"Lỗi reranker: {exc}"],
                "trace": child_res.get("trace", {}),
            }

        final_pool = reranked_children[:final_top_k]
        accepted_evidence = [c for c in final_pool if c["accepted"]]

        if not accepted_evidence:
            return {
                "status": "insufficient_evidence",
                "mode": mode,
                "original_question": q0_clean,
                "query_set": child_res.get("queries"),
                "child_hits": final_pool,
                "parent_candidates": [],
                "accepted_evidence": [],
                "answer": None,
                "citations": [],
                "stage_latencies_ms": stage_latencies,
                "api_call_counts": {"generation_calls": gen_calls_count, "embedding_calls": emb_calls_count},
                "warnings": warnings_list + [f"Không có child nào đạt RERANK_MIN_SCORE ({min_score})"],
                "trace": child_res.get("trace", {}),
            }

        # Format Citations for Flat Modes
        citations = []
        for idx, ev in enumerate(accepted_evidence, start=1):
            evidence_id = f"C{idx}"
            ev["evidence_id"] = evidence_id
            cit_item = {
                "evidence_id": evidence_id,
                "child_id": ev["child_id"],
                "source": ev["source"],
                "page_start": ev["page_start"],
                "page_end": ev["page_end"],
                "child_rerank_score": ev["child_rerank_score"],
            }
            citations.append(cit_item)

        # Answer Generation
        t0_ans = time.perf_counter()
        answer_text, gen_err = _generate_answer_from_evidence(
            question=q0_clean,
            accepted_evidence=accepted_evidence,
            config=config,
            answer_generator_fn=answer_generator_fn,
        )
        stage_latencies["answer_generation"] = round((time.perf_counter() - t0_ans) * 1000, 2)
        if gen_err:
            warnings_list.append(gen_err)
        else:
            gen_calls_count += 1

        overall_status = child_res.get("status", "ready")

        return {
            "status": overall_status,
            "mode": mode,
            "original_question": q0_clean,
            "query_set": child_res.get("queries"),
            "child_hits": final_pool,
            "parent_candidates": [],
            "accepted_evidence": accepted_evidence,
            "answer": answer_text,
            "citations": citations,
            "stage_latencies_ms": stage_latencies,
            "api_call_counts": {"generation_calls": gen_calls_count, "embedding_calls": emb_calls_count},
            "identities": {
                "config_identity": _config_identity(config),
                "schema_version": SCHEMA_VERSION,
            },
            "warnings": warnings_list,
            "trace": child_res.get("trace", {}),
        }


# ──────────────────────────────────────────────────────────────────────────────
# § 14. CLI
# ──────────────────────────────────────────────────────────────────────────────

def _run_audit_cli(args: argparse.Namespace) -> int:
    try:
        config = load_hierarchical_config()
    except ValueError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1

    input_dir = Path(args.input_dir) if args.input_dir else None
    try:
        report = run_hierarchy_audit(input_dir, config)
    except ValueError as exc:
        print(f"AUDIT ERROR: {exc}", file=sys.stderr)
        return 1

    ls = report["load_stats"]
    tot = report["totals"]
    dist = report["parent_size_distribution"]

    print("\n══ Hierarchy Audit ══════════════════════════════════════════")
    print(f"  Files read   : {ls['files']}")
    print(f"  Total records: {ls['total_records']}")
    print(f"  Valid chunks : {ls['valid']}  (sources: {ls.get('sources', '?')})")
    print(f"\n── Resolution Breakdown ─────────────────────────────────────")
    for method, cnt in sorted(report["resolution_breakdown"].items()):
        print(f"  {method:<20}: {cnt}")
    print(f"\n── Totals ───────────────────────────────────────────────────")
    print(f"  Children     : {tot['children']}")
    print(f"  Parents      : {tot['parents']}")
    print(f"  Ambiguous    : {tot['ambiguous_children']}")
    print(f"  Orphans      : {tot['orphan_children']}")
    print(f"  Parent warns : {tot['warned_parents']}")
    print(f"\n── Parent Size Distribution (chars) ─────────────────────────")
    for k, v in dist.items():
        print(f"  {k:<8}: {v}")
    if report["ambiguous_examples"]:
        print(f"\n── Ambiguous Examples ────────────────────────────────────")
        for ex in report["ambiguous_examples"]:
            print(f"  {ex['child_id']}: {ex['warnings']}")
    if report["parent_warning_examples"]:
        print(f"\n── Parent Warnings ──────────────────────────────────────")
        for ex in report["parent_warning_examples"]:
            print(f"  {ex['parent_id']}: {ex['warnings']}")
    print("═" * 60)
    return 0


def _run_build_cli(args: argparse.Namespace) -> int:
    try:
        config = load_hierarchical_config()
    except ValueError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1

    input_dir = Path(args.input_dir) if args.input_dir else None
    storage_dir = Path(args.storage_dir) if args.storage_dir else None

    print("Loading chunks…")
    try:
        chunks, load_stats, fingerprints = load_hierarchical_chunks(input_dir)
    except ValueError as exc:
        print(f"LOAD ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"  {load_stats['valid']} hierarchical chunks from {load_stats['sources']} sources")
    print("Resolving hierarchy…")
    children = resolve_hierarchy(chunks)

    print("Building parents…")
    parents, children = build_parents(children, config)
    print(f"  {len(parents)} parent windows built")

    print("Saving registry…")
    try:
        manifest = save_hierarchy(children, parents, load_stats, fingerprints, config, storage_dir)
    except Exception as exc:
        print(f"SAVE ERROR: {exc}", file=sys.stderr)
        return 1

    counts = manifest["counts"]
    print("\nBUILD COMPLETE")
    print(f"  children       : {counts['children']}")
    print(f"  parents        : {counts['parents']}")
    print(f"  ambiguous      : {counts['ambiguous_children']}")
    print(f"  orphans        : {counts['orphan_children']}")
    print(f"  parent warnings: {counts['warned_parents']}")
    print(f"  built_at       : {manifest['built_at']}")
    return 0


def _run_status_cli(args: argparse.Namespace) -> int:
    storage_dir = Path(args.storage_dir) if args.storage_dir else None
    status = hierarchy_status(storage_dir)

    print("\n══ Hierarchy Status (read-only) ══════════════════════════════")
    print(f"  registry_exists : {status['registry_exists']}")
    print(f"  storage_dir     : {status['storage_dir']}")
    for fname, exists in status.get("files", {}).items():
        mark = "✓" if exists else "✗"
        print(f"  {mark} {fname}")
    if status.get("manifest"):
        m = status["manifest"]
        c = m.get("counts", {})
        print(f"  built_at        : {m.get('built_at', '?')}")
        print(f"  schema_version  : {m.get('schema_version', '?')}")
        print(f"  children        : {c.get('children', '?')}")
        print(f"  parents         : {c.get('parents', '?')}")
        print(f"  ambiguous       : {c.get('ambiguous_children', '?')}")
        print(f"  orphans         : {c.get('orphan_children', '?')}")
    elif status.get("error"):
        print(f"  error: {status['error']}")
    print("═" * 60)
    return 0


def _run_expand_query_cli(args: argparse.Namespace) -> int:
    try:
        config = load_hierarchical_config()
    except ValueError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1

    question = args.question
    print(f"Expanding query: '{question}'...\n")
    res = generate_multi_queries(question, config=config)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def _run_multi_child_cli(args: argparse.Namespace) -> int:
    try:
        config = load_hierarchical_config()
    except ValueError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1

    question = args.question
    print(f"Running Multi-Query Child Retrieval for: '{question}'...\n")
    try:
        res = retrieve_multi_query_children(question, config=config)
    except Exception as exc:
        print(f"RETRIEVAL ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Status  : {res['status']}")
    print(f"Queries : {len(res['queries'])}")
    for q in res['queries']:
        print(f"  [{q['query_id']}] ({q['origin']}/{q['focus']}): {q['text']}")

    hits = res["merged_child_hits"]
    print(f"\nMerged Child Hits ({len(hits)} total):\n")
    print(f"{'Rank':<6} {'Child ID':<35} {'Supp':<6} {'QIDs':<16} {'MQ-RRF Score':<14} {'Per-Query Ranks'}")
    print("-" * 100)
    for h in hits:
        qids_str = ",".join(h["support_query_ids"])
        pq_str = str(h["per_query_ranks"])
        print(f"{h['multi_query_rank']:<6} {h['child_id']:<35} {h['support_query_count']:<6} {qids_str:<16} {h['multi_query_rrf_score']:<14.6f} {pq_str}")

    print("\nTrace summary:")
    print(json.dumps(res["trace"], ensure_ascii=False, indent=2))
    return 0


def _run_parent_retrieve_cli(args: argparse.Namespace) -> int:
    try:
        config = load_hierarchical_config()
    except ValueError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1

    question = args.question
    mode = args.mode
    print(f"Running Parent Retrieval ({mode}) for: '{question}'...\n")
    try:
        res = retrieve_parent_documents(question, mode=mode, config=config)
    except Exception as exc:
        print(f"PARENT RETRIEVAL ERROR: {exc}", file=sys.stderr)
        return 1

    if res["status"] == "hierarchy_not_ready":
        print(f"STATUS: {res['status']}")
        for w in res.get("warnings", []):
            print(f"WARNING: {w}")
        return 1

    print(f"Status           : {res['status']}")
    print(f"Accepted Parents : {len(res['accepted_parents'])}")
    print(f"Dropped Parents  : {len(res['dropped_parents'])}\n")

    for p in res["accepted_parents"]:
        art = p["structural_path"].get("article") or "Document Block"
        print(f"Parent [{p['parent_id']}] (Rank: {p['parent_rank']}, Score: {p['parent_rrf_score']:.6f}, Source: {p['source']}, Chars: {len(p['text'])})")
        print(f"  Article: {art}")
        for cid in p["supporting_child_ids"]:
            is_scoring = " (Scoring)" if cid in p["scoring_child_ids"] else ""
            is_anchor = " [Anchor]" if cid == p["anchor_child_id"] else ""
            print(f"  ├── Child [{cid}]{is_anchor}{is_scoring}")

    print("\nTrace Summary:")
    print(json.dumps(res["trace"], ensure_ascii=False, indent=2))
    return 0


def _run_query_cli(args: argparse.Namespace) -> int:
    try:
        config = load_hierarchical_config()
    except ValueError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1

    question = args.question
    mode = args.mode
    print(f"Executing End-to-End Query ({mode}) for: '{question}'...\n")

    try:
        res = query_hierarchical_rag(question, mode=mode, config=config)
    except Exception as exc:
        print(f"QUERY ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Status            : {res['status']}")
    print(f"Mode              : {res['mode']}")
    print(f"Accepted Evidence : {len(res['accepted_evidence'])}")
    print(f"Generation Calls  : {res['api_call_counts']['generation_calls']}")

    if res.get("answer"):
        print("\n── ANSWER ───────────────────────────────────────────────────")
        print(res["answer"])

    if res.get("citations"):
        print("\n── CITATIONS ────────────────────────────────────────────────")
        for cit in res["citations"]:
            art = cit.get("structural_path", {}).get("article") or "Document Block"
            score = cit.get("parent_rerank_score", cit.get("child_rerank_score", 0.0))
            print(f"[{cit['evidence_id']}] {cit.get('source')} | {art} | Score: {score:.4f}")

    if res.get("warnings"):
        print("\n── WARNINGS ─────────────────────────────────────────────────")
        for w in res["warnings"]:
            print(f"  • {w}")

    print("\n── STAGE LATENCIES (ms) ─────────────────────────────────────")
    for k, v in res.get("stage_latencies_ms", {}).items():
        print(f"  {k:<30}: {v}")

    return 0


def _run_compare_cli(args: argparse.Namespace) -> int:
    try:
        config = load_hierarchical_config()
    except ValueError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1

    question = args.question
    print(f"Running Retrieval & Rerank Comparison across 4 Modes for: '{question}'...\n")

    modes = ["single_flat", "multi_flat", "single_parent", "multi_parent"]
    comparison_results = {}

    for m in modes:
        print(f"Running mode: {m}...")
        try:
            # Lệnh compare KHÔNG gọi answer generation (answer_generator_fn returns None)
            res = query_hierarchical_rag(
                question=question,
                mode=m,
                config=config,
                answer_generator_fn=lambda p, c: None,
            )
            comparison_results[m] = res
        except Exception as exc:
            comparison_results[m] = {"status": "error", "error": str(exc)}

    print("\n══ MODE COMPARISON MATRIX ══════════════════════════════════════════════════════════")
    print(f"{'Mode':<16} {'Status':<25} {'Evidence Count':<16} {'Top Rerank Score':<18} {'Latency (ms)'}")
    print("-" * 90)

    for m in modes:
        res = comparison_results.get(m, {})
        st = res.get("status", "unknown")
        acc = res.get("accepted_evidence", [])
        cnt = len(acc)
        top_sc = (acc[0].get("parent_rerank_score", acc[0].get("child_rerank_score", 0.0)) if acc else 0.0)
        total_lat = sum(res.get("stage_latencies_ms", {}).values())
        print(f"{m:<16} {st:<25} {cnt:<16} {top_sc:<18.4f} {total_lat:.1f}")

    print("═" * 90)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Buổi 09 Hierarchical RAG CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    audit_p = sub.add_parser("hierarchy-audit", help="Audit hierarchy (read-only, no file write)")
    audit_p.add_argument("--input-dir", default=None, dest="input_dir",
                         help="Override input directory for chunks")
    audit_p.add_argument("--storage-dir", default=None, dest="storage_dir")

    build_p = sub.add_parser("build-hierarchy", help="Build and persist hierarchy registry")
    build_p.add_argument("--input-dir", default=None, dest="input_dir")
    build_p.add_argument("--storage-dir", default=None, dest="storage_dir")

    status_p = sub.add_parser("hierarchy-status", help="Read-only status of existing registry")
    status_p.add_argument("--storage-dir", default=None, dest="storage_dir")

    expand_p = sub.add_parser("expand-query", help="Generate Multi-Query variants for a question")
    expand_p.add_argument("--question", required=True, help="Original question text")

    multi_child_p = sub.add_parser("multi-child", help="Run multi-query hybrid child retrieval & fusion")
    multi_child_p.add_argument("--question", required=True, help="Original question text")

    parent_ret_p = sub.add_parser("parent-retrieve", help="Retrieve Parent Documents using child hits")
    parent_ret_p.add_argument("--question", required=True, help="Original question text")
    parent_ret_p.add_argument("--mode", choices=["single_parent", "multi_parent"], default="multi_parent", help="Retrieval mode")

    query_p = sub.add_parser("query", help="Run end-to-end RAG query")
    query_p.add_argument("--question", required=True, help="Original question text")
    query_p.add_argument("--mode", choices=["single_flat", "multi_flat", "single_parent", "multi_parent"], default="multi_parent")

    compare_p = sub.add_parser("compare", help="Compare retrieval & rerank across 4 modes without answer generation")
    compare_p.add_argument("--question", required=True, help="Original question text")

    return parser


def main(argv: Optional[List[str]] = None) -> int:  # noqa: F821
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "hierarchy-audit":
        return _run_audit_cli(args)
    if args.command == "build-hierarchy":
        return _run_build_cli(args)
    if args.command == "hierarchy-status":
        return _run_status_cli(args)
    if args.command == "expand-query":
        return _run_expand_query_cli(args)
    if args.command == "multi-child":
        return _run_multi_child_cli(args)
    if args.command == "parent-retrieve":
        return _run_parent_retrieve_cli(args)
    if args.command == "query":
        return _run_query_cli(args)
    if args.command == "compare":
        return _run_compare_cli(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())




