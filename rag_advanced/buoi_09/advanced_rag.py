# ============================================================
# SNAPSHOT — Buổi 09 Baseline Copy
# Source : rag_foundation/buoi_08/advanced_rag.py
# SHA-256: A733CD2615E700CE9599BD4E62BC9C9ECF8C2C175219A1D5C73C75D054B27EED
# Copied : 2026-08-10T16:50:34+07:00
# NOTE   : Đây là bản sao snapshot từ Buổi 08. Không import runtime
#           từ directory buoi_08. Không thay đổi logic ở Bước 02.
# ============================================================
"""Buổi 09 — Advanced RAG module (snapshot từ Buổi 08).

BM25 lexical retrieval (Bước 04): tokenize_vi_legal, build_bm25_index, bm25_search.
Semantic candidate retrieval (Bước 05): get_semantic_candidates, advanced_status.
  - Dùng lại loader, config, collection naming, embedding, Chroma helpers từ rag.py cùng thư mục.
  - Không import runtime từ buoi_07/.
Config loader (Bước 03) nạp `.env` theo `Path(__file__)`. Không phụ thuộc cwd.
"""
import time
import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = BASE_DIR / ".env"


# ──────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────

def _parse_int(var_name: str, default_val: int, min_val: int, max_val: int) -> int:
    raw = os.getenv(var_name, str(default_val)).strip()
    try:
        val = int(raw)
    except ValueError as exc:
        raise ValueError(f"{var_name} phải là số nguyên, nhận: '{raw}'") from exc
    if val < min_val or val > max_val:
        raise ValueError(f"{var_name} phải nằm trong khoảng [{min_val}, {max_val}], nhận: {val}")
    return val


def _parse_float(var_name: str, default_val: float, min_val: float, max_val: float) -> float:
    raw = os.getenv(var_name, str(default_val)).strip()
    try:
        val = float(raw)
    except ValueError as exc:
        raise ValueError(f"{var_name} phải là số thực, nhận: '{raw}'") from exc
    if val < min_val or val > max_val:
        raise ValueError(f"{var_name} phải nằm trong khoảng [{min_val}, {max_val}], nhận: {val}")
    return val


def _require_non_empty(var_name: str, value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{var_name} không được để rỗng")
    return cleaned


def load_advanced_config(env_path: Optional[Path] = None) -> Dict[str, Any]:
    """Nạp và validate cấu hình Advanced RAG từ `.env` (mặc định: cùng thư mục với module)."""


    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    embedding_model = _require_non_empty("GEMINI_EMBEDDING_MODEL", os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"))
    generation_model = _require_non_empty("GEMINI_GENERATION_MODEL", os.getenv("GEMINI_GENERATION_MODEL", "gemini-3.5-flash-lite"))
    reranker_model = _require_non_empty("RERANKER_MODEL", os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"))

    device = os.getenv("RERANK_DEVICE", "auto").strip().lower()
    valid_devices = {"auto", "cpu", "cuda"}
    if device not in valid_devices:
        raise ValueError(f"RERANK_DEVICE phải thuộc {sorted(valid_devices)}, nhận: '{device}'")

    embedding_dim = _parse_int("GEMINI_EMBEDDING_DIM", 768, 128, 3072)
    max_distance = _parse_float("RAG_MAX_DISTANCE", 0.45, 0.0, 2.0)

    bm25_candidates = _parse_int("BM25_CANDIDATES", 20, 1, 100)
    semantic_candidates_count = _parse_int("SEMANTIC_CANDIDATES", 20, 1, 100)
    rerank_candidates = _parse_int("RERANK_CANDIDATES", 20, 1, 100)
    final_top_k = _parse_int("FINAL_TOP_K", 5, 1, 100)

    if final_top_k > rerank_candidates:
        raise ValueError(
            f"FINAL_TOP_K ({final_top_k}) không được lớn hơn RERANK_CANDIDATES ({rerank_candidates})"
        )

    rrf_k = _parse_int("RRF_K", 60, 1, 1000)
    rrf_bm25_weight = _parse_float("RRF_BM25_WEIGHT", 1.0, 0.0, 100.0)
    rrf_semantic_weight = _parse_float("RRF_SEMANTIC_WEIGHT", 1.0, 0.0, 100.0)
    if rrf_bm25_weight == 0.0 and rrf_semantic_weight == 0.0:
        raise ValueError("RRF_BM25_WEIGHT và RRF_SEMANTIC_WEIGHT không được đồng thời bằng 0")

    reranker_max_length = _parse_int("RERANKER_MAX_LENGTH", 512, 64, 4096)
    rerank_batch_size = _parse_int("RERANK_BATCH_SIZE", 4, 1, 64)
    rerank_min_score = _parse_float("RERANK_MIN_SCORE", 0.50, 0.0, 1.0)

    return {
        "gemini_api_key": api_key,
        "gemini_embedding_model": embedding_model,
        "gemini_embedding_dim": embedding_dim,
        "gemini_generation_model": generation_model,
        "rag_max_distance": max_distance,
        "bm25_candidates": bm25_candidates,
        "semantic_candidates": semantic_candidates_count,
        "rerank_candidates": rerank_candidates,
        "final_top_k": final_top_k,
        "rrf_k": rrf_k,
        "rrf_bm25_weight": rrf_bm25_weight,
        "rrf_semantic_weight": rrf_semantic_weight,
        "reranker_model": reranker_model,
        "reranker_max_length": reranker_max_length,
        "rerank_batch_size": rerank_batch_size,
        "rerank_min_score": rerank_min_score,
        "rerank_device": device,
    }


def resolve_rerank_pool_size(union_count: int, config: Dict[str, Any]) -> int:
    """Số ứng viên đưa sang rerank: min(RERANK_CANDIDATES, union_count).

    Union nhỏ hơn RERANK_CANDIDATES là hợp lệ, không phải lỗi cấu hình.
    """
    if union_count < 0:
        raise ValueError("union_count phải >= 0")
    return min(int(config["rerank_candidates"]), union_count)


# ──────────────────────────────────────────────
# Bước 04: Tokenizer & BM25
# ──────────────────────────────────────────────

_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)


def tokenize_vi_legal(text: str) -> List[str]:
    r"""Tokenize văn bản pháp lý tiếng Việt cho BM25.

    Quy tắc: NFC → casefold → regex [\w]+ Unicode.
    Dùng chung cho corpus và query.
    """
    if not isinstance(text, str):
        raise TypeError(f"tokenize_vi_legal: đầu vào phải là str, nhận {type(text).__name__}")
    normalized = unicodedata.normalize("NFC", text)
    casefolded = normalized.casefold()
    return _TOKEN_PATTERN.findall(casefolded)


def build_bm25_index(chunks: List[Dict[str, Any]]) -> Any:
    """Xây BM25Okapi index trong memory. Không pickle, không database."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:  # pragma: no cover
        raise ImportError("rank_bm25 chưa được cài đặt.") from exc
    if not chunks:
        raise ValueError("Không thể xây BM25 index từ danh sách chunk rỗng")
    tokenized_corpus = [tokenize_vi_legal(chunk["text"]) for chunk in chunks]
    return BM25Okapi(tokenized_corpus)


def bm25_search(
    question: str,
    chunks: List[Dict[str, Any]],
    candidate_k: int = 20,
) -> List[Dict[str, Any]]:
    """BM25 lexical retrieval.

    Output mỗi candidate: chunk_id, text, source, page_start, page_end,
    bm25_rank, bm25_score.
    """
    if not isinstance(question, str):
        raise ValueError("question phải là string")
    question_stripped = question.strip()
    if not question_stripped:
        raise ValueError("question không được để rỗng")

    query_tokens = tokenize_vi_legal(question_stripped)
    if not query_tokens:
        raise ValueError(f"question không tạo ra token nào: '{question_stripped}'")
    if not chunks:
        raise ValueError("Danh sách chunks không được rỗng")

    actual_k = min(candidate_k, len(chunks))
    index = build_bm25_index(chunks)
    raw_scores = index.get_scores(query_tokens)

    scored: List[Tuple[float, str, int]] = [
        (float(raw_scores[i]), chunks[i]["chunk_id"], i)
        for i in range(len(chunks))
    ]
    scored.sort(key=lambda x: (-x[0], x[1]))

    return [
        {
            "chunk_id": chunks[orig_idx]["chunk_id"],
            "text": chunks[orig_idx]["text"],
            "source": chunks[orig_idx]["source"],
            "page_start": chunks[orig_idx]["page_start"],
            "page_end": chunks[orig_idx]["page_end"],
            "bm25_rank": rank_zero + 1,
            "bm25_score": score,
        }
        for rank_zero, (score, _cid, orig_idx) in enumerate(scored[:actual_k])
    ]


# ──────────────────────────────────────────────
# Bước 05: Semantic Candidate Retrieval
# ──────────────────────────────────────────────

def _import_rag() -> Any:
    """Import rag module từ cùng thư mục (buoi_08/rag.py)."""
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    import importlib
    return importlib.import_module("rag")


def advanced_status(
    strategy: str = "hierarchical",
    chroma_client: Any = None,
) -> Dict[str, Any]:
    """Trạng thái Advanced RAG — read-only, không tạo collection, không gọi API.

    Trả về:
        strategy, corpus_size, semantic collection name/exists/count,
        embedding model/dimension, BM25 ready flag, reranker model,
        reranker cache exists.
    """
    rag = _import_rag()
    config = rag.load_config()
    adv_config = load_advanced_config()

    # Corpus size (BM25 ready)
    try:
        chunks, stats = rag.load_chunks(strategy=strategy)
        corpus_size = stats.get("valid_chunks", len(chunks))
        bm25_ready = corpus_size > 0
    except Exception:
        corpus_size = 0
        bm25_ready = False

    # Semantic collection — read-only
    collection_name = rag.build_collection_name(
        strategy,
        config["gemini_embedding_model"],
        config["gemini_embedding_dim"],
    )
    client = rag.get_chroma_client(client=chroma_client)
    collection = rag.get_collection_or_none(client, collection_name)
    collection_exists = collection is not None
    collection_count = collection.count() if collection_exists else 0

    # Reranker cache — filesystem check only, no download
    reranker_model = adv_config["reranker_model"]
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    storage_hf = BASE_DIR / "storage" / "huggingface"
    model_slug = reranker_model.replace("/", "--")
    reranker_cache_exists = any(
        root.exists() and any(root.rglob(f"*{model_slug}*"))
        for root in [hf_cache, storage_hf]
    )

    return {
        "strategy": strategy,
        "corpus_size": corpus_size,
        "bm25_ready": bm25_ready,
        "semantic_collection_name": collection_name,
        "collection_exists": collection_exists,
        "collection_count": collection_count,
        "embedding_model": config["gemini_embedding_model"],
        "embedding_dim": config["gemini_embedding_dim"],
        "api_key_present": bool(config["gemini_api_key"]),
        "reranker_model": reranker_model,
        "reranker_cache_exists": reranker_cache_exists,
    }


def get_semantic_candidates(
    question: str,
    candidate_k: int = 20,
    strategy: str = "hierarchical",
    query_embedder: Optional[Any] = None,
    chroma_client: Any = None,
) -> List[Dict[str, Any]]:
    """Truy xuất semantic candidates từ Chroma.

    Output mỗi candidate: chunk_id, text, source, page_start, page_end,
    semantic_rank, semantic_distance.

    Quy tắc:
    - Không tạo collection, không gọi generation.
    - n_results = min(candidate_k, collection.count()).
    - distance thấp hơn xếp trước (giữ thứ tự Chroma trả về).
    - Thiếu API key → raise ValueError, không dùng vector giả.
    - Validate collection metadata/configuration trước khi query.
    """
    rag = _import_rag()
    config = rag.load_config()

    # Validate question
    if not isinstance(question, str):
        raise ValueError("question phải là string")
    question_stripped = question.strip()
    if not question_stripped:
        raise ValueError("question không được để rỗng")

    # API key required — no fallback vector
    if not config["gemini_api_key"] and query_embedder is None:
        raise ValueError(
            "GEMINI_API_KEY bị thiếu. Hãy điền API key vào .env trước khi dùng semantic retrieval."
        )

    # Get collection
    collection_name = rag.build_collection_name(
        strategy,
        config["gemini_embedding_model"],
        config["gemini_embedding_dim"],
    )
    client = rag.get_chroma_client(client=chroma_client)
    collection = rag.get_collection_or_none(client, collection_name)

    # Validate collection existence and metadata
    if collection is None:
        raise ValueError(
            f"Collection '{collection_name}' chưa tồn tại. "
            f"Hãy chạy 'prepare-semantic --strategy {strategy}' trước."
        )
    if collection.count() < 1:
        raise ValueError(f"Collection '{collection_name}' rỗng. Hãy index lại.")

    expected_metadata = rag.build_collection_metadata(strategy, config)
    rag.verify_collection_compatibility(collection, expected_metadata)

    # Query embedding
    query_vector = rag._create_query_embedding(
        question_stripped, config, query_embedder=query_embedder
    )

    # Query Chroma
    n_results = min(candidate_k, collection.count())
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    candidates: List[Dict[str, Any]] = []
    for rank_zero, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        candidates.append({
            "chunk_id": meta.get("chunk_id", ""),
            "text": doc,
            "source": meta.get("source", ""),
            "page_start": meta.get("page_start", 0),
            "page_end": meta.get("page_end", 0),
            "semantic_rank": rank_zero + 1,
            "semantic_distance": float(dist),
        })

    return candidates


def prepare_semantic(
    strategy: str = "hierarchical",
    reset: bool = False,
    embedder: Optional[Any] = None,
    chroma_client: Any = None,
) -> Dict[str, Any]:
    """Index chunks vào Chroma bằng Gemini embedding thật.

    Idempotent — nếu collection đã tồn tại và metadata khớp, chỉ upsert.
    Thiếu API key sẽ fail ngay, không dùng vector giả.
    Ghi vào storage/ của Buổi 08, không đụng Buổi 07.
    """
    rag = _import_rag()
    return rag.index_chunks(
        strategy=strategy,
        reset=reset,
        embedder=embedder,
        client=chroma_client,
    )


# ──────────────────────────────────────────────
# Stubs cho Bước 06+
# ──────────────────────────────────────────────

def rrf_fusion(
    bm25_results: List[Dict[str, Any]],
    semantic_results: List[Dict[str, Any]],
    k: int = 60,
    bm25_weight: float = 1.0,
    semantic_weight: float = 1.0,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """RRF fusion hợp nhất kết quả từ BM25 và Semantic."""
    # 1. Map candidates by chunk_id
    bm25_map = {item["chunk_id"]: item for item in bm25_results}
    semantic_map = {item["chunk_id"]: item for item in semantic_results}

    # 2. Get all chunk_ids
    all_chunk_ids = set(bm25_map.keys()) | set(semantic_map.keys())

    fused_candidates = []

    for cid in all_chunk_ids:
        bm25_item = bm25_map.get(cid)
        sem_item = semantic_map.get(cid)

        # Retrieve metadata
        ref_item = bm25_item if bm25_item is not None else sem_item
        text = ref_item["text"]
        source = ref_item["source"]
        page_start = ref_item["page_start"]
        page_end = ref_item["page_end"]

        # Check consistency if present in both
        if bm25_item is not None and sem_item is not None:
            if bm25_item["source"] != sem_item["source"]:
                raise ValueError(f"Metadata mismatch for chunk {cid}: source doesn't match")
            if bm25_item["page_start"] != sem_item["page_start"] or bm25_item["page_end"] != sem_item["page_end"]:
                raise ValueError(f"Metadata mismatch for chunk {cid}: page range doesn't match")
            if bm25_item["text"].strip() != sem_item["text"].strip():
                raise ValueError(f"Metadata mismatch for chunk {cid}: text doesn't match")

        # Compute RRF score
        score = 0.0
        matched_by = []
        
        bm25_rank = None
        bm25_score = None
        if bm25_item is not None:
            bm25_rank = bm25_item["bm25_rank"]
            bm25_score = bm25_item["bm25_score"]
            score += bm25_weight / (k + bm25_rank)
            matched_by.append("bm25")

        semantic_rank = None
        semantic_distance = None
        if sem_item is not None:
            semantic_rank = sem_item["semantic_rank"]
            semantic_distance = sem_item["semantic_distance"]
            score += semantic_weight / (k + semantic_rank)
            matched_by.append("semantic")

        fused_candidates.append({
            "chunk_id": cid,
            "text": text,
            "source": source,
            "page_start": page_start,
            "page_end": page_end,
            "bm25_rank": bm25_rank,
            "bm25_score": bm25_score,
            "semantic_rank": semantic_rank,
            "semantic_distance": semantic_distance,
            "rrf_score": score,
            "matched_by": matched_by,
        })

    # 3. Sort with tie-breakers
    def sort_key(item):
        bm_r = item["bm25_rank"] if item["bm25_rank"] is not None else float("inf")
        sem_r = item["semantic_rank"] if item["semantic_rank"] is not None else float("inf")
        best_r = min(bm_r, sem_r)
        return (-item["rrf_score"], best_r, sem_r, bm_r, item["chunk_id"])

    fused_candidates.sort(key=sort_key)

    # 4. Assign fused_rank and select top_n
    for rank_idx, item in enumerate(fused_candidates):
        item["fused_rank"] = rank_idx + 1

    return fused_candidates[:top_n]


def hybrid_retrieval(
    question: str,
    strategy: str = "hierarchical",
    query_embedder: Optional[Any] = None,
    chroma_client: Any = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Hybrid retrieval kết hợp BM25 và Semantic."""
    rag = _import_rag()
    adv_config = load_advanced_config()
    
    k = adv_config["rrf_k"]
    bm25_weight = adv_config["rrf_bm25_weight"]
    semantic_weight = adv_config["rrf_semantic_weight"]
    bm25_candidates_count = adv_config["bm25_candidates"]
    semantic_candidates_count = adv_config["semantic_candidates"]
    top_n = adv_config["final_top_k"]
    
    t0 = time.perf_counter()
    chunks, stats = rag.load_chunks(strategy=strategy)
    bm25_results = bm25_search(question=question, chunks=chunks, candidate_k=bm25_candidates_count)
    t1 = time.perf_counter()
    bm25_latency_ms = (t1 - t0) * 1000
    
    t2 = time.perf_counter()
    semantic_results = get_semantic_candidates(
        question=question,
        candidate_k=semantic_candidates_count,
        strategy=strategy,
        query_embedder=query_embedder,
        chroma_client=chroma_client,
    )
    t3 = time.perf_counter()
    semantic_latency_ms = (t3 - t2) * 1000
    
    t4 = time.perf_counter()
    fused_results = rrf_fusion(
        bm25_results=bm25_results,
        semantic_results=semantic_results,
        k=k,
        bm25_weight=bm25_weight,
        semantic_weight=semantic_weight,
        top_n=top_n,
    )
    t5 = time.perf_counter()
    fusion_latency_ms = (t5 - t4) * 1000

    # Calculate overlap count
    bm25_ids = {r["chunk_id"] for r in bm25_results}
    semantic_ids = {r["chunk_id"] for r in semantic_results}
    overlap_count = len(bm25_ids & semantic_ids)
    union_count = len(bm25_ids | semantic_ids)

    trace = {
        "bm25_candidate_count": len(bm25_results),
        "semantic_candidate_count": len(semantic_results),
        "union_count": union_count,
        "overlap_count": overlap_count,
        "fused_count": len(fused_results),
        "rrf_k": k,
        "rrf_bm25_weight": bm25_weight,
        "rrf_semantic_weight": semantic_weight,
        "latency_ms": {
            "bm25": bm25_latency_ms,
            "semantic": semantic_latency_ms,
            "fusion": fusion_latency_ms,
        },
    }
    
    return fused_results, trace


def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Cross‑encoder rerank — triển khai ở Bước 07.

    Returns candidates sorted by ``rerank_score`` descending, each dict includes:
        ``rerank_score`` (float), ``rerank_rank`` (int), ``accepted`` (bool).
    If the reranker model cache is missing, raises a ``ValueError`` with guidance.
    """
    # Verify reranker cache
    status = advanced_status()
    if not status.get("reranker_cache_exists"):
        raise ValueError(
            "Reranker model cache not found. Hãy chạy script tải model trước khi dùng."
        )

    # Lazy import Transformers and torch
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
    except ImportError as exc:
        raise ImportError("Thư viện 'transformers' cần thiết cho rerank.") from exc

    model_name = status["reranker_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # Prepare inputs (query + each candidate text)
    texts = [c["text"] for c in candidates]
    inputs = tokenizer([query] * len(texts), texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = model(**inputs).logits.squeeze(-1).float()
    scores = scores.tolist()

    # Attach scores and acceptance flag
    min_score = threshold if threshold is not None else load_advanced_config().get("rerank_min_score")
    for cand, sc in zip(candidates, scores):
        cand["rerank_score"] = sc
        cand["accepted"] = sc >= min_score

    # Sort by score descending and assign ranks
    sorted_cands = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    for idx, cand in enumerate(sorted_cands, start=1):
        cand["rerank_rank"] = idx

    return sorted_cands[:top_k]


def query_advanced_rag(
    question: str,
    top_k: int = 5,
    mode: str = "hybrid_rerank",
    strategy: str = "hierarchical",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Advanced RAG end‑to‑end pipeline (Bước 08).

    Args:
        question: Câu hỏi người dùng.
        top_k: Số chunk cuối cùng đưa vào generation.
        mode: Một trong ``bm25``, ``semantic``, ``hybrid``, ``hybrid_rerank``.
        strategy: Chunking strategy (e.g. ``hierarchical``).
        config: Optional config dict; nếu None sẽ load từ .env.

    Returns:
        Dict với các key: ``status``, ``mode``, ``answer``, ``citations``,
        ``evidence``, ``warnings``, ``trace``.
    """
    import time as _time

    VALID_MODES = {"bm25", "semantic", "hybrid", "hybrid_rerank"}
    if mode not in VALID_MODES:
        raise ValueError(f"mode phải là một trong {sorted(VALID_MODES)}, nhận: '{mode}'")

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question không được để rỗng")

    adv_cfg = config if config is not None else load_advanced_config()
    warnings_list: List[str] = []
    trace: Dict[str, Any] = {"mode": mode, "strategy": strategy}

    # ── 1. Retrieval ──────────────────────────────────────────────────────
    t_ret_start = _time.perf_counter()

    if mode == "bm25":
        rag = _import_rag()
        chunks, _ = rag.load_chunks(strategy=strategy)
        candidates = bm25_search(
            question=question,
            chunks=chunks,
            candidate_k=adv_cfg["bm25_candidates"],
        )
        candidates = candidates[:top_k]
        trace["bm25_candidate_count"] = len(candidates)
        trace["latency_ms"] = {"bm25": (_time.perf_counter() - t_ret_start) * 1000}

    elif mode == "semantic":
        candidates = get_semantic_candidates(
            question=question,
            candidate_k=adv_cfg["semantic_candidates"],
            strategy=strategy,
        )
        candidates = candidates[:top_k]
        trace["semantic_candidate_count"] = len(candidates)
        trace["latency_ms"] = {"semantic": (_time.perf_counter() - t_ret_start) * 1000}

    elif mode == "hybrid":
        fused, hybrid_trace = hybrid_retrieval(
            question=question,
            strategy=strategy,
        )
        candidates = fused[:top_k]
        trace.update(hybrid_trace)

    else:  # hybrid_rerank
        fused, hybrid_trace = hybrid_retrieval(
            question=question,
            strategy=strategy,
        )
        trace.update(hybrid_trace)

        # Pool size: min(RERANK_CANDIDATES, len(fused))
        pool = fused[: adv_cfg["rerank_candidates"]]
        trace["rerank_pool_size"] = len(pool)

        t_rr = _time.perf_counter()
        try:
            reranked = rerank_candidates(
                query=question,
                candidates=pool,
                top_k=top_k,
                threshold=adv_cfg["rerank_min_score"],
            )
            candidates = reranked
            trace["rerank_latency_ms"] = (_time.perf_counter() - t_rr) * 1000
            trace["reranked_count"] = len(candidates)
            trace["accepted_count"] = sum(1 for c in candidates if c.get("accepted"))

            # Compute rank_change for each candidate
            for cand in candidates:
                fused_rank = cand.get("fused_rank")
                rerank_rank = cand.get("rerank_rank")
                if fused_rank is not None and rerank_rank is not None:
                    cand["rank_change"] = fused_rank - rerank_rank
        except (ValueError, ImportError) as exc:
            warnings_list.append(f"Reranker không khả dụng, dùng kết quả hybrid: {exc}")
            candidates = fused[:top_k]
            trace["rerank_skipped"] = True

    t_ret_end = _time.perf_counter()
    trace["total_retrieval_ms"] = (t_ret_end - t_ret_start) * 1000

    # ── 2. Check candidates ───────────────────────────────────────────────
    if not candidates:
        return {
            "status": "no_candidates",
            "mode": mode,
            "answer": None,
            "citations": [],
            "evidence": [],
            "warnings": warnings_list + ["Không tìm thấy chunk nào phù hợp."],
            "trace": trace,
        }

    # ── 3. Generation ─────────────────────────────────────────────────────
    try:
        import google.generativeai as genai
    except ImportError:
        warnings_list.append("google-generativeai chưa được cài đặt. Bỏ qua generation.")
        return {
            "status": "retrieval_only",
            "mode": mode,
            "answer": None,
            "citations": [],
            "evidence": candidates,
            "warnings": warnings_list,
            "trace": trace,
        }

    api_key = adv_cfg.get("gemini_api_key", "")
    if not api_key:
        warnings_list.append("GEMINI_API_KEY chưa được cấu hình. Bỏ qua generation.")
        return {
            "status": "retrieval_only",
            "mode": mode,
            "answer": None,
            "citations": [],
            "evidence": candidates,
            "warnings": warnings_list,
            "trace": trace,
        }

    genai.configure(api_key=api_key)
    gen_model = genai.GenerativeModel(adv_cfg["gemini_generation_model"])

    # Build context from candidates
    context_parts = []
    citations: List[str] = []
    for i, cand in enumerate(candidates, start=1):
        src = cand.get("source", "?")
        p_start = cand.get("page_start", "?")
        p_end = cand.get("page_end", "?")
        context_parts.append(f"[{i}] ({src}, tr.{p_start}-{p_end})\n{cand['text']}")
        citations.append(f"[{i}] {src}, trang {p_start}–{p_end}")

    context_text = "\n\n".join(context_parts)
    prompt = (
        f"Dựa vào các đoạn văn bản pháp lý sau đây, hãy trả lời câu hỏi một cách chính xác:\n\n"
        f"{context_text}\n\n"
        f"Câu hỏi: {question}\n"
        f"Trả lời:"
    )

    t_gen_start = _time.perf_counter()
    try:
        response = gen_model.generate_content(prompt)
        answer = response.text.strip()
        trace["generation_latency_ms"] = (_time.perf_counter() - t_gen_start) * 1000
        status = "ok"
    except Exception as exc:
        warnings_list.append(f"Generation lỗi: {exc}")
        answer = None
        status = "generation_error"

    return {
        "status": status,
        "mode": mode,
        "answer": answer,
        "citations": citations,
        "evidence": candidates,
        "warnings": warnings_list,
        "trace": trace,
    }


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def _run_bm25_cli(args: argparse.Namespace) -> int:
    rag = _import_rag()
    try:
        chunks, stats = rag.load_chunks(strategy=args.strategy)
    except Exception as exc:
        print(f"ERROR khi load chunks: {exc}")
        return 1
    if not chunks:
        print(f"ERROR: Không có chunk nào cho strategy='{args.strategy}'")
        return 1
    try:
        results = bm25_search(question=args.question, chunks=chunks, candidate_k=args.top_k)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"\nBM25 Retrieval — strategy={args.strategy}  question='{args.question}'")
    print(f"Corpus size: {stats['valid_chunks']}  candidate_k: {args.top_k}")
    print("-" * 90)
    for r in results:
        preview = r["text"][:100].replace("\n", " ")
        print(f"[{r['bm25_rank']:2}] score={r['bm25_score']:.4f}"
              f"  src={r['source']}  p.{r['page_start']}-{r['page_end']}"
              f"  id={r['chunk_id']}")
        print(f"     {preview}…")
    print("-" * 90)
    return 0


def _run_status_cli(args: argparse.Namespace) -> int:
    try:
        info = advanced_status(strategy=args.strategy)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    print("\nAdvanced RAG STATUS")
    print(f"  strategy            : {info['strategy']}")
    print(f"  corpus_size         : {info['corpus_size']}")
    print(f"  bm25_ready          : {'Có' if info['bm25_ready'] else 'Không'}")
    print(f"  semantic_collection : {info['semantic_collection_name']}")
    print(f"  collection_exists   : {'Có' if info['collection_exists'] else 'Không'}")
    print(f"  collection_count    : {info['collection_count']}")
    print(f"  embedding_model     : {info['embedding_model']}")
    print(f"  embedding_dim       : {info['embedding_dim']}")
    print(f"  api_key_present     : {'Có' if info['api_key_present'] else 'Thiếu'}")
    print(f"  reranker_model      : {info['reranker_model']}")
    print(f"  reranker_cache      : {'Có' if info['reranker_cache_exists'] else 'Chưa tải'}")
    return 0


def _run_prepare_semantic_cli(args: argparse.Namespace) -> int:
    print(f"Chuẩn bị semantic index — strategy={args.strategy}"
          + (" [RESET]" if args.reset else ""))
    try:
        result = prepare_semantic(strategy=args.strategy, reset=args.reset)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    print("PREPARE_SEMANTIC_OK")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _run_semantic_cli(args: argparse.Namespace) -> int:
    try:
        results = get_semantic_candidates(
            question=args.question,
            candidate_k=args.top_k,
            strategy=args.strategy,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"\nSemantic Candidates — strategy={args.strategy}  question='{args.question}'")
    print(f"candidate_k: {args.top_k}")
    print("-" * 90)
    for r in results:
        preview = r["text"][:100].replace("\n", " ")
        print(f"[{r['semantic_rank']:2}] dist={r['semantic_distance']:.4f}"
              f"  src={r['source']}  p.{r['page_start']}-{r['page_end']}"
              f"  id={r['chunk_id']}")
        print(f"     {preview}…")
    print("-" * 90)
    return 0


def _run_hybrid_cli(args: argparse.Namespace) -> int:
    try:
        results, trace = hybrid_retrieval(
            question=args.question,
            strategy=args.strategy,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"\nHybrid Retrieval (RRF) — strategy={args.strategy}  question='{args.question}'")
    print(f"BM25 Candidates: {trace['bm25_candidate_count']}  Semantic Candidates: {trace['semantic_candidate_count']}")
    print(f"Union Count: {trace['union_count']}  Overlap Count: {trace['overlap_count']}")
    print(f"RRF Parameters: k={trace['rrf_k']}  bm25_weight={trace['rrf_bm25_weight']}  semantic_weight={trace['rrf_semantic_weight']}")
    print(f"Latencies: BM25={trace['latency_ms']['bm25']:.1f}ms  Semantic={trace['latency_ms']['semantic']:.1f}ms  Fusion={trace['latency_ms']['fusion']:.1f}ms")
    print("-" * 115)
    print(f"{'Rank':<5} | {'RRF Score':<9} | {'BM25 Rank/Score':<16} | {'Semantic Rank/Dist':<19} | {'Matched By':<16} | {'Preview'}")
    print("-" * 115)
    for r in results:
        bm25_info = f"{r['bm25_rank'] or '-'}/{r['bm25_score']:.3f}" if r['bm25_score'] is not None else "-"
        sem_info = f"{r['semantic_rank'] or '-'}/{r['semantic_distance']:.4f}" if r['semantic_distance'] is not None else "-"
        matched_str = ", ".join(r['matched_by'])
        preview = r['text'][:55].replace('\n', ' ')
        print(f"{r['fused_rank']:<5} | {r['rrf_score']:.6f} | {bm25_info:<16} | {sem_info:<19} | {matched_str:<16} | {preview}…")
    print("-" * 115)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Buổi 08 Advanced RAG CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # bm25
    bm25_p = subparsers.add_parser("bm25", help="BM25 lexical retrieval chẩn đoán")
    bm25_p.add_argument("--strategy", default="hierarchical",
                        choices=["fixed-size", "semantic", "hierarchical"])
    bm25_p.add_argument("--question", required=True)
    bm25_p.add_argument("--top-k", type=int, default=5, dest="top_k")

    # status
    status_p = subparsers.add_parser("status", help="Advanced RAG status (read-only)")
    status_p.add_argument("--strategy", default="hierarchical",
                          choices=["fixed-size", "semantic", "hierarchical"])

    # prepare-semantic
    prep_p = subparsers.add_parser("prepare-semantic", help="Index chunks vào Chroma bằng Gemini embedding")
    prep_p.add_argument("--strategy", default="hierarchical",
                        choices=["fixed-size", "semantic", "hierarchical"])
    prep_p.add_argument("--reset", action="store_true", help="Xóa collection cũ trước khi index")

    # semantic
    sem_p = subparsers.add_parser("semantic", help="Semantic candidate retrieval chẩn đoán")
    sem_p.add_argument("--strategy", default="hierarchical",
                       choices=["fixed-size", "semantic", "hierarchical"])
    sem_p.add_argument("--question", required=True)
    sem_p.add_argument("--top-k", type=int, default=5, dest="top_k")

    # hybrid
    hyb_p = subparsers.add_parser("hybrid", help="Hybrid retrieval (BM25 + Semantic + RRF) chẩn đoán")
    hyb_p.add_argument("--strategy", default="hierarchical",
                       choices=["fixed-size", "semantic", "hierarchical"])
    hyb_p.add_argument("--question", required=True)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "bm25":
        return _run_bm25_cli(args)
    if args.command == "status":
        return _run_status_cli(args)
    if args.command == "prepare-semantic":
        return _run_prepare_semantic_cli(args)
    if args.command == "semantic":
        return _run_semantic_cli(args)
    if args.command == "hybrid":
        return _run_hybrid_cli(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
