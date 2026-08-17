"""UI Helpers cho Buổi 09 Streamlit App (Bước 08).

Các hàm helper thuần Python giúp biến đổi dữ liệu kết quả từ `hierarchical_rag.py`
thành các cấu trúc dữ liệu thích hợp cho rendering UI (DataFrames, Trees, Matrices).
100% offline, không đụng Streamlit hay API call.
"""

from typing import Any, Dict, List, Optional, Tuple


def build_query_child_matrix(
    queries: List[Dict[str, Any]],
    merged_child_hits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Tạo bảng ma trận Query–Child.

    Hàng: child_id
    Cột: Q0, Q1, Q2...
    Giá trị: rank của child trong query đó (hoặc None nếu không xuất hiện).
    """
    if not queries or not merged_child_hits:
        return {"qids": [], "rows": []}

    qids = [q["query_id"] for q in queries]
    rows = []

    for hit in merged_child_hits:
        cid = hit["child_id"]
        pq_ranks = hit.get("per_query_ranks", {})
        rank_cells = {qid: pq_ranks.get(qid) for qid in qids}

        rows.append({
            "child_id": cid,
            "source": hit.get("source", ""),
            "support_query_count": hit.get("support_query_count", 0),
            "multi_query_rrf_score": hit.get("multi_query_rrf_score", 0.0),
            "ranks": rank_cells,
        })

    return {"qids": qids, "rows": rows}


def format_parent_tree_node(parent: Dict[str, Any]) -> Dict[str, Any]:
    """Tạo cấu trúc dữ liệu cây Parent–Child cho 1 parent candidate."""
    art = parent.get("structural_path", {}).get("article") or "Document Block"
    chap = parent.get("structural_path", {}).get("chapter") or ""
    rank_orig = parent.get("parent_rank", "?")
    rank_rr = parent.get("parent_rerank_rank", "?")
    rank_change = parent.get("parent_rank_change", 0)

    change_str = f"+{rank_change}" if rank_change > 0 else str(rank_change)

    return {
        "parent_id": parent.get("parent_id"),
        "header_title": f"Parent [{parent.get('parent_id')}] — {art}",
        "source_pages": f"{parent.get('source')} (tr. {parent.get('page_start')}-{parent.get('page_end')})",
        "chapter": chap,
        "article": art,
        "rank_summary": f"Rank ban đầu: {rank_orig} ➔ Rank sau Rerank: {rank_rr} (Thay đổi: {change_str})",
        "scores": f"Parent RRF Score: {parent.get('parent_rrf_score', 0.0):.6f} | Rerank Score: {parent.get('parent_rerank_score', 0.0):.4f}",
        "anchor_child_id": parent.get("anchor_child_id"),
        "scoring_child_ids": parent.get("scoring_child_ids", []),
        "supporting_child_ids": parent.get("supporting_child_ids", []),
        "support_query_ids": parent.get("support_query_ids", []),
        "text": parent.get("text", ""),
        "char_count": len(parent.get("text", "")),
        "ambiguous": parent.get("ambiguous", False),
        "warnings": parent.get("warnings", []),
    }


def format_citation_display(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chuẩn hóa danh sách citation hiển thị cho UI."""
    formatted = []
    for cit in citations:
        art = cit.get("structural_path", {}).get("article") or "Văn bản"
        score = cit.get("parent_rerank_score", cit.get("child_rerank_score", 0.0))
        formatted.append({
            "label": f"[{cit.get('evidence_id')}]",
            "title": f"{art} — {cit.get('source')}",
            "pages": f"Trang {cit.get('page_start')}-{cit.get('page_end')}",
            "score": f"{score:.4f}",
            "parent_id": cit.get("parent_id"),
            "anchor_child_id": cit.get("anchor_child_id"),
            "supporting_child_ids": cit.get("supporting_child_ids", []),
            "ambiguous": cit.get("ambiguous", False),
            "warnings": cit.get("warnings", []),
        })
    return formatted


def build_mode_comparison_row(mode: str, res: Dict[str, Any]) -> Dict[str, Any]:
    """Tạo 1 hàng dữ liệu cho bảng so sánh 4 modes trong Tab 4."""
    status = res.get("status", "unknown")
    acc = res.get("accepted_evidence", [])
    cnt = len(acc)

    unit_type = "parent" if "parent" in mode else "child"
    ev_ids = ", ".join([e.get("evidence_id", "?") for e in acc]) if acc else "—"

    top_sc = (acc[0].get("parent_rerank_score", acc[0].get("child_rerank_score", 0.0)) if acc else 0.0)

    trace = res.get("trace", {})
    child_cnt = trace.get("union_child_count", trace.get("input_child_hit_count", len(res.get("child_hits", []))))
    parent_cnt = trace.get("unique_parent_count", len(res.get("parent_candidates", [])))

    ctx_chars = trace.get("expanded_parent_chars", sum(len(e.get("text", "")) for e in acc))
    exp_factor = trace.get("context_expansion_factor", 1.0)

    latencies = res.get("stage_latencies_ms", {})
    tot_lat = sum(latencies.values()) if latencies else 0.0

    api_counts = res.get("api_call_counts", {})

    return {
        "mode": mode,
        "status": status,
        "unit_type": unit_type,
        "evidence_count": cnt,
        "evidence_ids": ev_ids,
        "top_rerank_score": round(top_sc, 4),
        "retrieved_child_count": child_cnt,
        "expanded_parent_count": parent_cnt,
        "context_chars": ctx_chars,
        "expansion_factor": round(exp_factor, 2),
        "total_latency_ms": round(tot_lat, 1),
        "generation_calls": api_counts.get("generation_calls", 0),
        "embedding_calls": api_counts.get("embedding_calls", 0),
        "warnings": res.get("warnings", []),
    }


def map_status_badge(status: str) -> Tuple[str, str, str]:
    """Trả về (type, title, description) hỗ trợ rendering error UI UX.

    Types: success, warning, error, info
    """
    MAP = {
        "ready": ("success", "Sẵn sàng", "Pipeline hoàn thành thành công."),
        "ready_with_warnings": ("warning", "Sẵn sàng (có cảnh báo)", "Pipeline thành công nhưng có cảnh báo dữ liệu."),
        "hierarchy_not_ready": ("error", "Hierarchy Store Chưa Sẵn Sàng", "Hãy bấm 'Build Hierarchy Registry' ở sidebar trước."),
        "collection_not_ready": ("error", "Vector DB Chưa Sẵn Sàng", "Hãy bấm 'Prepare Semantic Index' ở sidebar trước."),
        "query_generation_unavailable": ("warning", "Multi-Query Không Khả Dụng", "Không sinh được query variants (thiếu API Key hoặc lỗi model). Đã tự động fallback về Q0."),
        "multi_query_partial": ("warning", "Multi-Query Một Phần", "Một số generated queries bị lỗi hoặc không có kết quả."),
        "reranker_unavailable": ("error", "Reranker Model Chưa Tải", "Chưa tải model Cross-Encoder (BAAI/bge-reranker-v2-m3). Hãy chạy script download trước."),
        "insufficient_evidence": ("warning", "Không Đủ Căn Cứ Đạt Gate", "Không có tài liệu nào đạt RERANK_MIN_SCORE. Đã dừng sinh câu trả lời để tránh ảo giác."),
        "generation_error": ("error", "Lỗi Gemini Generation", "Không thể gọi Gemini API để sinh câu trả lời."),
    }
    return MAP.get(status, ("info", f"Trạng thái: {status}", "Chi tiết trong trace log."))
