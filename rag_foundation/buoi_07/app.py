import sys
import streamlit as st
from pathlib import Path

# Cấu hình đường dẫn để import rag.py
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import rag

st.set_page_config(
    page_title="RAG Pipeline System - Buổi 07",
    page_icon="📚",
    layout="wide"
)


def main():
    st.title("📚 RAG Pipeline System - Buổi 07")
    st.caption("Giao diện hỏi đáp RAG với Semantic Retrieval, Confidence Gate & Citation Mapping")

    # 1. Đọc cấu hình từ rag.py
    try:
        config = rag.get_config()
    except Exception as e:
        st.error(f"Lỗi đọc cấu hình từ .env: {e}")
        st.stop()

    # 2. Sidebar: Cấu hình & Trạng thái hệ thống
    st.sidebar.header("⚙️ Cấu hình & Trạng thái")

    # Hiển thị thông tin API Key (Có/Thiếu, tuyệt đối không lộ key)
    if config["API_KEY_PRESENT"]:
        st.sidebar.success("🔑 GEMINI_API_KEY: Có")
    else:
        st.sidebar.error("🔑 GEMINI_API_KEY: Thiếu (Vui lòng điền vào file .env)")

    # Selectbox chọn Strategy
    strategy = st.sidebar.selectbox(
        "Chiến lược Chunking (Strategy):",
        options=["hierarchical", "semantic", "fixed-size"],
        index=0
    )

    # Selector chọn Top-K (1 đến 10)
    top_k = st.sidebar.slider(
        "Số lượng evidence (Top-K):",
        min_value=1,
        max_value=10,
        value=int(config["DEFAULT_TOP_K"])
    )

    st.sidebar.divider()
    st.sidebar.subheader("ℹ️ Cấu hình Model & Threshold")
    st.sidebar.text(f"• Embedding Model: {config['GEMINI_EMBEDDING_MODEL']}")
    st.sidebar.text(f"• Embedding Dim  : {config['GEMINI_EMBEDDING_DIM']}")
    st.sidebar.text(f"• Generation Model: {config['GEMINI_GENERATION_MODEL']}")
    st.sidebar.text(f"• Max Distance   : {config['RAG_MAX_DISTANCE']}")

    # Kiểm tra trạng thái collection (Read-only status)
    try:
        status_info = rag.run_status(strategy=strategy)
    except Exception as e:
        status_info = {
            "collection_name": "Lỗi",
            "exists": False,
            "record_count": 0
        }
        st.sidebar.error(f"Lỗi đọc status: {e}")

    st.sidebar.divider()
    st.sidebar.subheader("🗄️ Trạng thái ChromaDB")
    st.sidebar.text(f"Collection: {status_info['collection_name']}")
    if status_info["exists"]:
        st.sidebar.success(f"Trạng thái: Đã tồn tại ({status_info['record_count']} records)")
    else:
        st.sidebar.warning("Trạng thái: Chưa tồn tại (0 records)")

    # 3. Khu vực Indexing (Quản lý dữ liệu vector)
    st.subheader("📥 1. Lập chỉ mục dữ liệu (Indexing)")
    with st.expander("Quản lý Index & Vector Collection", expanded=not status_info["exists"]):
        reset_option = st.checkbox("Reset collection trước khi index (Xóa dữ liệu cũ của strategy này)")

        if st.button("🚀 Index dữ liệu"):
            if not config["API_KEY_PRESENT"]:
                st.error("Không thể index: Thiếu GEMINI_API_KEY trong file .env. Vui lòng điền key và thử lại.")
            else:
                with st.spinner(f"Đang sinh vector embedding và lập chỉ mục cho strategy '{strategy}'..."):
                    try:
                        idx_res = rag.run_index(
                            input_dir=rag.DEFAULT_INPUT_DIR,
                            strategy=strategy,
                            reset=reset_option
                        )
                        st.session_state["last_index_result"] = idx_res
                        st.success("Indexing hoàn tất thành công!")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Lỗi trong quá trình index: {ex}")

        if "last_index_result" in st.session_state:
            res = st.session_state["last_index_result"]
            st.info(
                f"Kết quả index gần nhất:\n"
                f"• Collection: {res['collection_name']}\n"
                f"• Chunks đã index: {res['indexed_chunks']}\n"
                f"• Tổng số records hiện tại: {res['total_records']}\n"
                f"• Reset thực hiện: {res['reset_performed']}"
            )

    st.divider()

    # 4. Khu vực Hỏi đáp (Question & Query)
    st.subheader("💬 2. Hỏi đáp với tài liệu (RAG Query)")

    question_input = st.text_area(
        "Nhập câu hỏi của bạn (tối đa 2000 ký tự):",
        placeholder="Ví dụ: Quy định về quy trình và lãi suất như thế nào?",
        height=100
    )

    if st.button("🔍 Gửi câu hỏi", type="primary"):
        clean_q = question_input.strip()
        if not clean_q:
            st.warning("Vui lòng nhập nội dung câu hỏi trước khi gửi.")
        elif not config["API_KEY_PRESENT"]:
            st.error("Thiếu GEMINI_API_KEY trong file .env. Vui lòng điền API Key để thực hiện truy vấn.")
        elif not status_info["exists"] or status_info["record_count"] == 0:
            st.error(f"Collection cho strategy '{strategy}' chưa được lập chỉ mục hoặc rỗng. Vui lòng bấm 'Index dữ liệu' ở trên.")
        else:
            with st.spinner("Đang tìm kiếm thông tin và tổng hợp câu trả lời..."):
                try:
                    q_res = rag.run_query(
                        question=clean_q,
                        strategy=strategy,
                        top_k=top_k
                    )
                    st.session_state["last_query_result"] = q_res
                except Exception as ex:
                    st.error(f"Lỗi thực hiện truy vấn: {ex}")

    # 5. Hiển thị Kết quả Hỏi đáp (Answer & Evidence)
    if "last_query_result" in st.session_state:
        q_res = st.session_state["last_query_result"]
        st.markdown("### 📋 Kết quả")

        # Display status badge
        status = q_res.get("status", "")
        if status == "answered":
            st.success("✅ Trạng thái: Answered (Đã trả lời dựa trên tài liệu)")
        elif status == "insufficient_evidence":
            st.warning("⚠️ Trạng thái: Insufficient Evidence (Không đủ thông tin liên quan)")
        elif status == "retrieval_only":
            st.info("ℹ️ Trạng thái: Retrieval Only (Đã truy xuất nguồn nhưng chưa tạo được câu trả lời tổng hợp)")
        else:
            st.write(f"Trạng thái: {status}")

        # Display Answer
        st.markdown("#### 💡 Câu trả lời:")
        st.markdown(q_res['answer'])

        # Display Citations (Mapped by rag.py)
        citations = q_res.get("citations", [])
        if citations:
            st.markdown("#### 📌 Trích dẫn nguồn (Citations):")
            for c in citations:
                st.markdown(f"- **{c['evidence_id']}**: `{c['display']}`")

        # Display Warnings
        warnings = q_res.get("warnings", [])
        if warnings:
            st.markdown("#### ⚠️ Cảnh báo:")
            for w in warnings:
                st.warning(w)

        st.divider()

        # Display Evidence Items
        st.markdown("### 📄 Nguồn tham khảo")
        evidences = q_res.get("evidence", [])

        if not evidences:
            st.info("Chưa có nguồn tham khảo nào được truy xuất.")
        else:
            st.caption("ℹ️ Khoảng cách Cosine (Distance) càng nhỏ cho thấy độ tương đồng ngữ nghĩa càng cao.")
            for ev in evidences:
                p_start = ev["page_start"]
                p_end = ev["page_end"]
                page_str = f"tr. {p_start}" if p_start == p_end else f"tr. {p_start}-{p_end}"
                summary_line = f"{ev['source']} – {page_str} – Chunk: {ev['chunk_id']}"

                accepted = ev.get("accepted", False)
                status_icon = "🟢 ĐẠT NGƯỠNG" if accepted else "🔴 BỊ LOẠI (Vượt RAG_MAX_DISTANCE)"

                with st.expander(f"[{ev['evidence_id']}] {summary_line} | {status_icon}"):
                    st.write(f"**Evidence ID**: {ev['evidence_id']}")
                    st.write(f"**File Nguồn**: {ev['source']}")
                    st.write(f"**Trang**: {page_str}")
                    st.write(f"**Chunk ID**: {ev['chunk_id']}")
                    st.write(f"**Distance**: `{ev['distance']}` ({'Chấp nhận cho Prompt' if accepted else 'Bỏ qua cho Prompt'})")
                    st.markdown("**Nội dung Chunk:**")
                    st.code(ev["text"], language="text")


if __name__ == "__main__":
    main()
