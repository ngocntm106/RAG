"""RAG Foundation — Buổi 09: Multi-query & Parent–Child Retrieval Dashboard.

Streamlit App giao diện Buổi 09 phân biệt rõ nét với Buổi 08:
1. Nhiều query từ 1 câu hỏi (Cards Q0..Qn)
2. Ma trận Query–Child (Table / Heatmap)
3. Cây Parent–Child (Tree View)
4. Parent rank trước / sau Rerank (Rank movement badges)
5. Context Expansion Factor (Metrics & Charts)

Run command:
    python -m streamlit run rag_advanced/buoi_09/app.py
"""

from pathlib import Path
import sys
import json
import os

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import hierarchical_rag
from ui_helpers import (
    build_mode_comparison_row,
    build_query_child_matrix,
    format_citation_display,
    format_parent_tree_node,
    map_status_badge,
)


def main():
    import streamlit as st
    import pandas as pd

    # Page Configuration
    st.set_page_config(
        page_title="Buổi 09 — Multi-query & Parent–Child RAG",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Styling CSS
    st.markdown("""
        <style>
        .main-header { font-size: 2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
        .sub-header { font-size: 1.05rem; font-weight: 500; color: #475569; margin-bottom: 1.5rem; }
        .card { background-color: #F8FAFC; border-radius: 8px; padding: 1rem; border: 1px solid #E2E8F0; margin-bottom: 1rem; }
        .q0-badge { background-color: #DBEAFE; color: #1E40AF; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
        .qn-badge { background-color: #F1F5F9; color: #475569; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
        .anchor-badge { background-color: #DCFCE7; color: #166534; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
        .scoring-badge { background-color: #FEF3C7; color: #92400E; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
        </style>
    """, unsafe_allow_html=True)

    # Header section
    st.markdown('<div class="main-header">🏦 RAG Foundation — Buổi 09: Multi-query & Parent–Child Retrieval</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">🔄 <b>Pipeline:</b> Query fan-out ➔ Hybrid per query ➔ Cross-query RRF ➔ Parent expansion ➔ Parent rerank</div>', unsafe_allow_html=True)

    # Load baseline config
    config = hierarchical_rag.load_hierarchical_config()

    # Session State Initialization
    if "last_query_result" not in st.session_state:
        st.session_state["last_query_result"] = None
    if "compare_results" not in st.session_state:
        st.session_state["compare_results"] = None

    # ──────────────────────────────────────────────────────────────────────────
    # SIDEBAR
    # ──────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Cấu Hình Pipeline")

        mode = st.selectbox(
            "RAG Mode",
            options=["multi_parent", "single_parent", "multi_flat", "single_flat"],
            index=0,
            help="Chọn pipeline mode xử lý query"
        )

        st.subheader("🎛️ Parameters")
        mq_count = st.slider("MULTI_QUERY_COUNT", min_value=1, max_value=5, value=config.get("multi_query_count", 2))
        per_q_cand = st.slider("PER_QUERY_CANDIDATES", min_value=5, max_value=50, value=config.get("per_query_candidates", 12))
        parent_cand = st.slider("PARENT_CANDIDATES", min_value=3, max_value=30, value=config.get("parent_candidates", 10))
        final_top_k = st.slider("FINAL_PARENT_TOP_K", min_value=1, max_value=10, value=config.get("final_parent_top_k", 3))
        rerank_min = st.slider("RERANK_MIN_SCORE", min_value=0.0, max_value=1.0, value=float(config.get("rerank_min_score", 0.35)), step=0.05)

        # Update runtime config
        runtime_config = dict(config)
        runtime_config["multi_query_count"] = mq_count
        runtime_config["per_query_candidates"] = per_q_cand
        runtime_config["parent_candidates"] = parent_cand
        runtime_config["final_parent_top_k"] = final_top_k
        runtime_config["rerank_min_score"] = rerank_min
        runtime_config["strategy"] = "hierarchical"

        st.divider()

        # System Status Section
        st.header("📌 Trạng Thái Hệ Thống")

        # 1. API Key
        has_api_key = bool(os.getenv("GEMINI_API_KEY", "").strip())
        if has_api_key:
            st.success("🔑 Gemini API Key: Đã cấu hình")
        else:
            st.warning("⚠️ Gemini API Key: Thiếu (Fallback Mode)")

        # 2. Models
        st.caption(f"**Embedding Model:** `{config['gemini_embedding_model']}`")
        st.caption(f"**Generation Model:** `{config['gemini_generation_model']}`")
        st.caption(f"**Reranker Model:** `{config['reranker_model']}`")

        # 3. Hierarchy Store Status
        h_status = hierarchical_rag.hierarchy_status()
        if h_status["registry_exists"]:
            m_info = h_status.get("manifest", {}).get("counts", {})
            st.success(f"📂 Hierarchy Store: Ready ({m_info.get('children', 0)} children, {m_info.get('parents', 0)} parents)")
        else:
            st.error("❌ Hierarchy Store: Chưa có registry")

        st.divider()

        # Action Buttons (Action riêng, có xác nhận)
        st.header("🛠️ Thao Tác Hệ Thống")
        if st.button("🔨 Build Hierarchy Registry", use_container_width=True):
            with st.spinner("Đang xây dựng Hierarchy Registry..."):
                try:
                    chunks, load_stats, fingerprints = hierarchical_rag.load_hierarchical_chunks()
                    children = hierarchical_rag.resolve_hierarchy(chunks)
                    parents, children = hierarchical_rag.build_parents(children, runtime_config)
                    manifest = hierarchical_rag.save_hierarchy(children, parents, load_stats, fingerprints, runtime_config)
                    st.success(f"Xây dựng thành công {manifest['counts']['parents']} parent windows!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Lỗi build hierarchy: {exc}")

        if st.button("⚡ Prepare Semantic Index", use_container_width=True):
            with st.spinner("Đang chuẩn bị Semantic Vector Index..."):
                try:
                    import advanced_rag
                    res = advanced_rag.prepare_semantic(strategy="hierarchical")
                    st.success(f"Chroma Vector DB sẵn sàng! Collection: '{res.get('collection_name')}'")
                except Exception as exc:
                    st.error(f"Lỗi prepare semantic: {exc}")

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN CONTENT TABS
    # ──────────────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "❓ Ask Advanced RAG",
        "🔀 Query Fan-out",
        "🌳 Parent–Child Explorer",
        "📊 Mode Comparison",
        "📈 Evaluation"
    ])

    # ── TAB 1: ASK ADVANCED RAG ───────────────────────────────────────────────
    with tab1:
        st.subheader("Hỏi đáp Văn bản Pháp luật Ngân hàng")

        q_input = st.text_area(
            "Nhập câu hỏi của bạn:",
            value="Điều kiện vay vốn và nhu cầu vốn không được cho vay là gì?",
            height=90,
            key="ask_question_input"
        )

        col_btn, col_mode_info = st.columns([1, 4])
        with col_btn:
            run_query_btn = st.button("🚀 Chạy Pipeline", type="primary", use_container_width=True)
        with col_mode_info:
            st.info(f"Chạy ở Mode: **{mode}** | Candidate Limit: **{parent_cand}** | Min Score: **{rerank_min}**")

        if run_query_btn and q_input.strip():
            with st.spinner(f"Đang xử lý query qua mode '{mode}'..."):
                res = hierarchical_rag.query_hierarchical_rag(
                    question=q_input.strip(),
                    mode=mode,
                    config=runtime_config,
                )
                st.session_state["last_query_result"] = res

        # Render Latest Result if available
        res = st.session_state.get("last_query_result")
        if res:
            st.divider()

            # Status Badge Header
            badge_type, badge_title, badge_desc = map_status_badge(res["status"])
            if badge_type == "success":
                st.success(f"**{badge_title}** — {badge_desc}")
            elif badge_type == "warning":
                st.warning(f"**{badge_title}** — {badge_desc}")
            elif badge_type == "error":
                st.error(f"**{badge_title}** — {badge_desc}")
            else:
                st.info(f"**{badge_title}** — {badge_desc}")

            # Top Metrics Bar
            m1, m2, m3, m4 = st.columns(4)
            lats = res.get("stage_latencies_ms", {})
            tot_lat = sum(lats.values())
            m1.metric("⏱️ Tổng Latency", f"{tot_lat:.1f} ms")
            m2.metric("📄 Accepted Evidence", len(res.get("accepted_evidence", [])))
            m3.metric("🤖 Gen API Calls", res.get("api_call_counts", {}).get("generation_calls", 0))
            m4.metric("🔤 Embed API Calls", res.get("api_call_counts", {}).get("embedding_calls", 0))

            # Answer Section
            if res.get("answer"):
                st.subheader("💡 Câu Trả Lời")
                st.markdown(res["answer"])

            # Citations Section
            cits = res.get("citations", [])
            if cits:
                st.subheader("📌 Danh Sách Dẫn Chứng (Citations)")
                formatted_cits = format_citation_display(cits)
                for item in formatted_cits:
                    with st.expander(f"{item['label']} {item['title']} ({item['pages']}) — Score: {item['score']}"):
                        st.write(f"**Parent ID:** `{item['parent_id']}` | **Anchor Child:** `{item['anchor_child_id']}`")
                        st.write(f"**Supporting Children:** `{', '.join(item['supporting_child_ids'])}`")

            # Warnings Section
            if res.get("warnings"):
                st.subheader("⚠️ Cảnh Báo")
                for w in res["warnings"]:
                    st.warning(w)

    # ── TAB 2: QUERY FAN-OUT ──────────────────────────────────────────────────
    with tab2:
        st.subheader("🔀 Multi-Query Fan-out & Query–Child Matrix")

        res = st.session_state.get("last_query_result")
        if not res:
            st.info("Hãy chạy câu hỏi ở **Tab 1** để xem phân tích Multi-Query Fan-out.")
        else:
            q_set = res.get("query_set", res.get("trace", {}).get("queries", []))
            queries = res.get("queries") or (q_set if isinstance(q_set, list) else [])

            if queries:
                st.markdown("#### 1. Các Câu Hỏi Biến Thể (Query Variants)")
                q_cols = st.columns(len(queries))
                for idx, q_item in enumerate(queries):
                    with q_cols[idx]:
                        is_q0 = q_item["query_id"] == "Q0"
                        badge_cls = "q0-badge" if is_q0 else "qn-badge"
                        st.markdown(
                            f'<div class="card">'
                            f'<span class="{badge_cls}">{q_item["query_id"]} ({q_item["origin"]})</span><br/>'
                            f'<b>Focus:</b> {q_item.get("focus", "N/A")}<br/>'
                            f'<p style="margin-top: 0.5rem;">{q_item["text"]}</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

            # Query-Child Matrix
            child_hits = res.get("child_hits") or res.get("trace", {}).get("merged_child_hits", [])
            # Try fetching merged child hits from retrieval trace if empty
            if not child_hits and res.get("accepted_evidence"):
                # Collect child hits from supporting children
                all_cids = []
                for p in res["accepted_evidence"]:
                    all_cids.extend(p.get("supporting_child_ids", []))
                child_hits = [{"child_id": cid, "support_query_count": 1, "multi_query_rrf_score": 0.01} for cid in set(all_cids)]

            matrix_data = build_query_child_matrix(queries, child_hits)
            if matrix_data["rows"]:
                st.markdown("#### 2. Ma Trận Query–Child Rank")
                matrix_rows = []
                for r in matrix_data["rows"]:
                    row_dict = {
                        "Child ID": r["child_id"],
                        "Source": r["source"],
                        "Support Count": r["support_query_count"],
                        "MQ-RRF Score": f"{r['multi_query_rrf_score']:.6f}",
                    }
                    for qid in matrix_data["qids"]:
                        rank_val = r["ranks"].get(qid)
                        row_dict[qid] = f"Rank {rank_val}" if rank_val else "—"
                    matrix_rows.append(row_dict)

                df_matrix = pd.DataFrame(matrix_rows)
                st.dataframe(df_matrix, use_container_width=True)

    # ── TAB 3: PARENT–CHILD EXPLORER ─────────────────────────────────────────
    with tab3:
        st.subheader("🌳 Parent–Child Resolution Tree Explorer")

        res = st.session_state.get("last_query_result")
        if not res or not res.get("parent_candidates"):
            st.info("Hãy chạy câu hỏi với mode **multi_parent** hoặc **single_parent** ở **Tab 1** để xem Cây Parent–Child.")
        else:
            parents = res.get("parent_candidates", [])
            for p in parents:
                node = format_parent_tree_node(p)

                with st.expander(f"{node['header_title']} | {node['scores']}"):
                    st.markdown(f"**Nguồn:** `{node['source_pages']}`")
                    st.markdown(f"**Chương/Điều:** {node['chapter']} | {node['article']}")
                    st.markdown(f"**Biến động Rank:** `{node['rank_summary']}`")
                    st.markdown(f"**Scoring Children (Top Limit):** `{', '.join(node['scoring_child_ids'])}`")
                    st.markdown(f"**Supporting Queries:** `{', '.join(node['support_query_ids'])}`")

                    if node["ambiguous"]:
                        st.warning("⚠️ Parent chứa child có nhãn ambiguous/unclear!")

                    st.markdown("---")
                    st.caption("Nội dung Parent Document Window đầy đủ:")
                    st.text_area(f"Text Parent [{node['parent_id']}]", value=node["text"], height=150, key=f"p_text_{node['parent_id']}")

    # ── TAB 4: MODE COMPARISON ───────────────────────────────────────────────
    with tab4:
        st.subheader("📊 Mode Comparison Matrix (Retrieval & Rerank Only)")
        st.caption("Chạy cùng 1 câu hỏi qua 4 modes để so sánh hiệu năng. Không sinh câu trả lời để tiết kiệm API.")

        comp_q = st.text_input(
            "Câu hỏi so sánh:",
            value="Điều kiện vay vốn và các trường hợp không được cho vay là gì?",
            key="compare_q_input"
        )

        if st.button("⚡ Chạy 4-Mode Comparison", type="primary"):
            with st.spinner("Đang chạy 4-Mode Comparison..."):
                comp_results = {}
                modes_list = ["single_flat", "multi_flat", "single_parent", "multi_parent"]
                for m in modes_list:
                    res_m = hierarchical_rag.query_hierarchical_rag(
                        question=comp_q.strip(),
                        mode=m,
                        config=runtime_config,
                        answer_generator_fn=lambda p, c: None,  # Bypass answer gen
                    )
                    comp_results[m] = res_m
                st.session_state["compare_results"] = comp_results

        comp_data = st.session_state.get("compare_results")
        if comp_data:
            rows = [build_mode_comparison_row(m, res) for m, res in comp_data.items()]
            df_comp = pd.DataFrame(rows)

            st.markdown("#### Ma Trận So Sánh Các Mode")
            st.dataframe(df_comp, use_container_width=True)

            st.info("ℹ️ **Lưu ý:** Sự khác biệt về Top Score và Context Chars phản ánh hiệu quả mở rộng bối cảnh của Parent Windowing và Multi-Query RRF.")

    # ── TAB 5: EVALUATION ─────────────────────────────────────────────────────
    with tab5:
        st.subheader("📈 Offline Benchmark Evaluation Metrics")

        # Check if evaluation report exists
        eval_report_file = BASE_DIR / "eval_report.json"
        if eval_report_file.exists():
            try:
                report_data = json.loads(eval_report_file.read_text(encoding="utf-8"))
                st.success("Đã tìm thấy báo cáo đánh giá Offline gần nhất!")
                st.json(report_data)
            except Exception as exc:
                st.error(f"Lỗi đọc file báo cáo evaluation: {exc}")
        else:
            st.warning("Chưa có báo cáo đánh giá offline (`eval_report.json`). Hãy chạy script `evaluate.py` để xuất báo cáo.")

        st.markdown("""
        #### Bảng Chỉ Số Đánh Giá Kỳ Vọng (SPEC Buổi 09)
        | Mode | Child Recall@5 | Parent Recall@3 | MRR@5 | nDCG@5 | Latency (ms) | Context Chars |
        |---|---|---|---|---|---|---|
        | `single_flat` | Baseline | N/A | Baseline | Baseline | Fast | ~1,200 |
        | `multi_flat` | High | N/A | High | High | Medium | ~2,500 |
        | `single_parent` | N/A | High | High | High | Medium | ~6,000 |
        | `multi_parent` | **Max** | **Max** | **Max** | **Max** | ~1.2s | **~12,000** |
        """)


if __name__ == "__main__":
    main()
