"""
==============================================================================
STREAMLIT WEB APPLICATION (Buổi 08 - Advanced RAG System Dashboard)
Mục đích:
  Giao diện trực quan so sánh đa tầng giữa Lexical BM25, Dense Semantic Vector,
  Reciprocal Rank Fusion (RRF) và Cross-Encoder Reranker.
==============================================================================
"""
import sys
import os
import json
import time
from pathlib import Path
import streamlit as st

# Thêm thư mục buoi_08 vào python path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import rag
import advanced_rag

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Advanced RAG Dashboard - Buổi 08",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện hiện đại
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .metric-card {
        background-color: #1E222D;
        border: 1px solid #2E3440;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .evidence-card {
        background-color: #1A1D24;
        border-left: 4px solid #4C566A;
        padding: 14px;
        margin-bottom: 12px;
        border-radius: 4px;
    }
    .evidence-card-accepted {
        border-left: 4px solid #A3BE8C;
        background-color: #1E2522;
    }
    .evidence-card-rejected {
        border-left: 4px solid #BF616A;
        background-color: #241E20;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        margin-right: 6px;
    }
    .badge-bm25 { background-color: #5E81AC; color: white; }
    .badge-semantic { background-color: #B48EAD; color: white; }
    .badge-rrf { background-color: #D08770; color: white; }
    .badge-rerank { background-color: #EBCB8B; color: black; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=600)
def load_bm25_cached(strategy: str):
    """Cache tải và validate chunks corpus cho BM25"""
    return rag.load_chunks(rag.DEFAULT_INPUT_DIR, strategy=strategy)


# SIDEBAR CONFIG & STATUS
st.sidebar.title("⚙️ Cấu Hình Advanced RAG")

try:
    config = advanced_rag.get_advanced_config()
    cfg_loaded = True
except Exception as e:
    st.sidebar.error(f"Lỗi nạp .env config: {e}")
    config = {}
    cfg_loaded = False

strategy = st.sidebar.selectbox(
    "Chiến lược Chunking",
    options=["hierarchical", "fixed-size", "semantic"],
    index=0,
    help="Chọn chiến lược phân đoạn văn bản"
)

retrieval_mode = st.sidebar.selectbox(
    "Chế độ Retrieval Mặc định",
    options=["hybrid_rerank", "hybrid", "semantic", "bm25"],
    index=0,
    help="hybrid_rerank là chế độ chuẩn của Advanced RAG"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Thông Số Hệ Thống")

if cfg_loaded:
    st.sidebar.text(f"• Final Top-K: {config['FINAL_TOP_K']}")
    st.sidebar.text(f"• BM25 Candidates: {config['BM25_CANDIDATES']}")
    st.sidebar.text(f"• Semantic Candidates: {config['SEMANTIC_CANDIDATES']}")
    st.sidebar.text(f"• RRF k={config['RRF_K']}, w_bm25={config['RRF_BM25_WEIGHT']}, w_sem={config['RRF_SEMANTIC_WEIGHT']}")
    st.sidebar.text(f"• Reranker Candidates: {config['RERANK_CANDIDATES']}")
    st.sidebar.text(f"• Reranker Min Score: {config['RERANK_MIN_SCORE']}")
    st.sidebar.text(f"• Reranker Model: {config['RERANKER_MODEL']}")
    st.sidebar.text(f"• Device Mode: {config['RERANK_DEVICE']}")

    status_info = advanced_rag.run_advanced_status(strategy=strategy, config=config)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔌 Trạng Thái Tài Nguyên")
    st.sidebar.write(f"• **Corpus Size**: {status_info['corpus_size']} chunks")
    st.sidebar.write(f"• **BM25 Corpus**: {'🟢 Sẵn sàng' if status_info['bm25_ready'] else '🔴 Chưa có'}")
    st.sidebar.write(f"• **Semantic Coll**: `{status_info['semantic_collection_name']}`")
    st.sidebar.write(f"• **Coll Records**: {status_info['collection_count']} chunks ({'🟢 Exists' if status_info['collection_exists'] else '🔴 Missing'})")
    st.sidebar.write(f"• **API Key Status**: {'🟢 Có' if status_info['api_key_status'] == 'Có' else '🔴 Thiếu GEMINI_API_KEY'}")
    st.sidebar.write(f"• **Reranker Cache**: {'🟢 Sẵn sàng' if status_info['reranker_cache_exists'] else '⚪ Chưa tải (Tải khi dùng)'}")

st.title("⚡ Advanced RAG System — Legal Document QA")
st.caption("Buổi 08: Kết hợp BM25 Keyword Search, Dense Vector Search, Reciprocal Rank Fusion (RRF) & Cross-Encoder Reranker")

tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Hỏi đáp Advanced RAG",
    "🔀 So sánh Retrieval",
    "🔍 Pipeline Trace",
    "📈 Đánh giá Performance"
])

# ==========================================
# TAB 1: HỎI ĐÁP ADVANCED RAG
# ==========================================
with tab1:
    st.header("Hỏi Đáp Với Advanced RAG Pipeline")
    user_question = st.text_input(
        "Nhập câu hỏi pháp lý tiếng Việt:",
        value="Điều 7 quy định như thế nào về cơ cấu lại thời hạn trả nợ?",
        key="tab1_q"
    )

    col_btn, col_mode = st.columns([1, 3])
    with col_mode:
        selected_mode = st.radio("Chế độ truy xuất:", ["hybrid_rerank", "hybrid", "semantic", "bm25"], horizontal=True)
    with col_btn:
        st.write("")
        st.write("")
        btn_run = st.button("🚀 Chạy Pipeline RAG", type="primary")

    if btn_run:
        if not user_question.strip():
            st.warning("Vui lòng nhập câu hỏi.")
        else:
            with st.spinner("Đang chạy pipeline Advanced RAG..."):
                try:
                    q_res = advanced_rag.run_advanced_query(
                        question=user_question,
                        mode=selected_mode,
                        strategy=strategy,
                        config=config
                    )
                    st.session_state["last_query_result"] = q_res

                except Exception as ex:
                    st.error(f"Lỗi thực thi Query: {ex}")
                    if "collection" in str(ex).lower():
                        st.info("💡 Hướng dẫn: Mở terminal và chạy lệnh `python rag_foundation/buoi_08/advanced_rag.py prepare-semantic --strategy hierarchical` để index dữ liệu ChromaDB.")

    if "last_query_result" in st.session_state:
        q_res = st.session_state["last_query_result"]

        # 1. Status Indicator
        status_val = q_res["status"]
        if status_val == "answered":
            st.success("✅ **Trạng thái**: Đã trả lời thành công (Answered)")
        elif status_val == "insufficient_evidence":
            st.warning("⚠️ **Trạng thái**: Không đủ thông tin phù hợp (Insufficient Evidence)")
        elif status_val == "retrieval_only":
            st.info("ℹ️ **Trạng thái**: Chỉ truy xuất dữ liệu (Retrieval Only)")
        elif status_val == "reranker_unavailable":
            st.error("❌ **Trạng thái**: Reranker không khả dụng (Reranker Unavailable)")

        # 2. Câu Trả Lời
        st.subheader("💡 Câu Trả Lời Tổng Hợp")
        st.markdown(q_res["answer"])

        # 3. Trích Dẫn Citations
        if q_res["citations"]:
            st.subheader("📌 Danh Sách Trích Dẫn Citations")
            for cit in q_res["citations"]:
                p_str = f"Trang {cit['page_start']}" if cit['page_start'] == cit['page_end'] else f"Trang {cit['page_start']}-{cit['page_end']}"
                st.markdown(f"- **{cit['citation_label']}**: Nguồn `{cit['source']}` ({p_str}) | Chunk ID: `{cit['chunk_id']}`")

        # 4. Warnings
        if q_res["warnings"]:
            with st.expander("⚠️ Danh sách Cảnh báo hệ thống"):
                for w in q_res["warnings"]:
                    st.write(f"- {w}")

        # 5. Evidence Cards (Top Candidates)
        st.subheader("📑 Danh Sách Evidence Candidates & Chi Tiết Thứ Hạng")
        for ev in q_res["evidence"]:
            is_acc = ev["accepted"]
            border_cls = "evidence-card-accepted" if is_acc else "evidence-card-rejected"
            acc_str = "🟢 Accepted" if is_acc else "🔴 Rejected"

            with st.container():
                st.markdown(f"""
                <div class="evidence-card {border_cls}">
                    <div style="display:flex; justify-between; align-items:center;">
                        <span><b>Chunk ID:</b> <code>{ev['chunk_id']}</code> | <b>Nguồn:</b> {ev['source']} (Trang {ev['page_start']})</span>
                        <span style="font-weight:bold;">{acc_str}</span>
                    </div>
                    <p style="margin-top:8px; font-size:14px;">{ev['text']}</p>
                </div>
                """, unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    b_rank = f"Rank {ev['bm25_rank']}" if ev['bm25_rank'] else "None"
                    b_score = f"{ev['bm25_score']:.4f}" if ev['bm25_score'] is not None else "None"
                    st.caption(f"🔵 **BM25**: {b_rank} (Score: {b_score})")
                with c2:
                    s_rank = f"Rank {ev['semantic_rank']}" if ev['semantic_rank'] else "None"
                    s_dist = f"{ev['semantic_distance']:.4f}" if ev['semantic_distance'] is not None else "None"
                    st.caption(f"🟣 **Semantic**: {s_rank} (Dist: {s_dist})")
                with c3:
                    f_rank = f"Rank {ev['fused_rank']}" if ev['fused_rank'] else "None"
                    rrf_s = f"{ev['rrf_score']:.6f}" if ev['rrf_score'] is not None else "None"
                    st.caption(f"🟠 **RRF Fusion**: {f_rank} (Score: {rrf_s})")
                with c4:
                    r_rank = f"Rank {ev['rerank_rank']}" if ev['rerank_rank'] else "None"
                    r_score = f"{ev['rerank_score']:.4f}" if ev['rerank_score'] is not None else "None"
                    r_shift = f"+{ev['rank_change']}" if ev['rank_change'] and ev['rank_change'] > 0 else f"{ev['rank_change']}" if ev['rank_change'] is not None else "0"
                    st.caption(f"🟡 **Reranker**: {r_rank} (Score: {r_score}, Shift: {r_shift})")
                st.markdown("---")

# ==========================================
# TAB 2: SO SÁNH RETRIEVAL (NO GENERATION)
# ==========================================
with tab2:
    st.header("🔀 So Sánh Độc Lập Các Chế Độ Retrieval")
    st.info("Tab này chạy cùng một câu hỏi qua 4 chế độ truy xuất khác nhau mà **TUYỆT ĐỐI KHÔNG gọi LLM Generation**, giúp so sánh trực tiếp vị trí và sự dịch chuyển của các chunk.")

    comp_q = st.text_input(
        "Nhập câu hỏi để so sánh retrieval:",
        value="Điều 7 quy định như thế nào về cơ cấu lại thời hạn trả nợ?",
        key="tab2_q"
    )
    btn_compare = st.button("🔍 Thực Hiện So Sánh 4 Modes", type="secondary")

    if btn_compare:
        if not comp_q.strip():
            st.warning("Vui lòng nhập câu hỏi.")
        else:
            with st.spinner("Đang thực thi retrieval và reranking cho 4 modes..."):
                try:
                    cmp_res = advanced_rag.run_mode_comparison(
                        question=comp_q,
                        strategy=strategy,
                        config=config
                    )
                    st.session_state["last_comparison_result"] = cmp_res
                except Exception as ex:
                    st.error(f"Lỗi so sánh retrieval: {ex}")

    if "last_comparison_result" in st.session_state:
        cmp_res = st.session_state["last_comparison_result"]

        st.subheader("📊 Bảng Tổng Hợp Thứ Hạng Qua Các Modes")
        # Format table Data
        table_rows = []
        for row in cmp_res["comparison_table"]:
            table_rows.append({
                "Chunk ID": row["chunk_id"],
                "BM25 Rank": f"Rank {row['ranks'].get('bm25')}" if 'bm25' in row['ranks'] else "-",
                "Semantic Rank": f"Rank {row['ranks'].get('semantic')}" if 'semantic' in row['ranks'] else "-",
                "RRF Hybrid Rank": f"Rank {row['ranks'].get('hybrid')}" if 'hybrid' in row['ranks'] else "-",
                "Rerank Rank": f"Rank {row['ranks'].get('hybrid_rerank')}" if 'hybrid_rerank' in row['ranks'] else "-",
                "Xuất hiện tại Modes": ", ".join(row["appeared_in_modes"])
            })
        st.dataframe(table_rows, use_container_width=True)

        st.subheader("🖼️ Top-K Candidates Cạnh Nhau Giữa Các Modes")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        modes_list = [("bm25", "🔵 BM25 Only", col_m1),
                      ("semantic", "🟣 Semantic Only", col_m2),
                      ("hybrid", "🟠 Hybrid RRF", col_m3),
                      ("hybrid_rerank", "🟡 Hybrid + Rerank", col_m4)]

        for m_key, m_title, col in modes_list:
            with col:
                st.markdown(f"### {m_title}")
                st.caption(f"Latency: {cmp_res['mode_latencies'].get(m_key, 0)}ms")
                m_items = cmp_res["mode_results"].get(m_key, [])
                for idx, item in enumerate(m_items, start=1):
                    with st.expander(f"#{idx} - {item['chunk_id']}", expanded=True):
                        st.caption(f"Nguồn: {item['source']} (Trang {item['page_start']})")
                        st.write(item['text'][:120] + "...")

# ==========================================
# TAB 3: PIPELINE TRACE
# ==========================================
with tab3:
    st.header("🔍 Pipeline Execution Trace & Metric Flow")
    st.info("Theo dõi luồng xử lý và số lượng ứng viên qua từng tầng của hệ thống Advanced RAG.")

    if "last_query_result" in st.session_state:
        trace = st.session_state["last_query_result"]["trace"]

        st.subheader("📈 Luồng Biến Đổi Số Lượng Candidate (Candidate Flow)")
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("BM25 Candidates", trace["bm25_candidates"])
        with m2:
            st.metric("Semantic Candidates", trace["semantic_candidates"])
        with m3:
            st.metric("Union (Overlap)", f"{trace['union']} ({trace['overlap']})")
        with m4:
            st.metric("Reranked", trace["reranked"])
        with m5:
            st.metric("Accepted Evidence", trace["accepted"])

        st.subheader("⏱️ Thời Gian Thực Thi Chi Tiết (Latency Breakdown)")
        lat = trace["latency_ms"]
        l1, l2, l3, l4, l5, l6 = st.columns(6)
        l1.metric("BM25", f"{lat['bm25']}ms")
        l2.metric("Semantic", f"{lat['semantic']}ms")
        l3.metric("RRF Fusion", f"{lat['fusion']}ms")
        l4.metric("Reranker", f"{lat['rerank']}ms")
        l5.metric("LLM Gen", f"{lat['generation']}ms")
        l6.metric("Tổng Cộng", f"{lat['total']}ms")

        st.markdown("""
        ---
        #### 💡 Chú Thích Thang Đo & Ý Nghĩa Thống Kê
        - **BM25 Score**: Điểm trùng khớp từ khóa (Raw score càng cao càng tốt, không có giới hạn trên).
        - **Cosine Distance**: Khoảng cách vector ngữ nghĩa (Càng gần 0.0 càng tương đồng, max threshold `0.45`).
        - **RRF Score**: Điểm dung hợp theo thứ hạng $1/(k + \text{rank})$ (Càng cao càng đại diện tốt ở cả 2 nhánh).
        - **Rerank Score**: Điểm Sigmoid của Cross-Encoder trong $[0.0, 1.0]$ (Không phải xác suất thống kê tuyệt đối, threshold `0.50`).
        """)
    else:
        st.warning("Vui lòng thực hiện một truy vấn ở Tab 1 để xem Pipeline Trace thực tế.")

# ==========================================
# TAB 4: ĐÁNH GIÁ PERFORMANCE
# ==========================================
with tab4:
    st.header("📈 Báo Cáo Đánh Giá Hiệu Năng Hệ Thống (Evaluation Reports)")
    st.info("Đọc và hiển thị kết quả đánh giá tự động từ các file JSON trong thư mục `reports/`.")

    reports_dir = BASE_DIR / "reports"
    report_files = [f for f in os.listdir(reports_dir) if f.endswith(".json")] if os.path.exists(reports_dir) else []

    if not report_files:
        st.warning("Chưa tìm thấy báo cáo đánh giá nào trong thư mục `reports/`.")
        st.info("💡 Hướng dẫn: Chạy lệnh `python rag_foundation/buoi_08/evaluate.py --strategy hierarchical --k 5` từ terminal để tạo báo cáo đánh giá.")
    else:
        selected_report = st.selectbox("Chọn file Báo cáo Đánh giá:", report_files)
        report_path = reports_dir / selected_report

        try:
            with open(report_path, "r", encoding="utf-8") as rf:
                rep_data = json.load(rf)

            st.write(f"**Báo cáo**: `{selected_report}` | **Thời gian**: `{rep_data.get('timestamp', 'N/A')}`")
            st.write(f"**Strategy**: `{rep_data.get('strategy', 'N/A')}` | **Eval K**: `{rep_data.get('k', 'N/A')}`")

            if rep_data.get("has_human_review_warning"):
                st.warning("⚠️ **CẢNH BÁO GOLD LABELS**: Bộ câu hỏi chứa các mục `needs_human_review=true`. Kết quả đánh giá mang tính chất tham khảo và chưa tuyên bố mode chiến thắng chính thức.")

            st.subheader("📊 Kết Quả Metrics Đánh Giá Theo Retrieval Mode")
            metrics_list = rep_data.get("mode_metrics", [])
            st.dataframe(metrics_list, use_container_width=True)

        except Exception as ex:
            st.error(f"Lỗi đọc file báo cáo đánh giá: {ex}")
