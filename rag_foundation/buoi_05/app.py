import json
import streamlit as st
from pathlib import Path
import pandas as pd

# Streamlit Page Config
st.set_page_config(
    page_title="RAG Foundation - Visualizer Buổi 5",
    page_icon="🧩",
    layout="wide"
)

# Paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"

# Custom Styling (CSS)
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .badge-ocr {
        background-color: #E3F2FD;
        color: #0D47A1;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-pymupdf {
        background-color: #E8F5E9;
        color: #1B5E20;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

def load_output_files():
    if not OUTPUT_DIR.exists():
        return {}
    
    chunks_files = list(OUTPUT_DIR.glob("*_chunks.json"))
    data = {}
    for cf in chunks_files:
        doc_name = cf.name.replace("_chunks.json", ".pdf")
        raw_file = OUTPUT_DIR / f"{cf.stem.replace('_chunks', '_raw')}.json"
        
        try:
            chunks = json.loads(cf.read_text(encoding="utf-8"))
            raw = json.loads(raw_file.read_text(encoding="utf-8")) if raw_file.exists() else {}
            data[doc_name] = {
                "chunks": chunks,
                "raw": raw
            }
        except Exception as e:
            st.error(f"Lỗi đọc file {cf.name}: {e}")
    return data

def main():
    st.markdown('<div class="main-title">🧩 Visualizer Chiến Lược Chunking & OCR</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Trực quan hóa sự thay đổi từ trang PDF ➔ OCR/Text Layer ➔ Các chiến lược Chunking (Buổi 5)</div>', unsafe_allow_html=True)

    data = load_output_files()
    
    if not data:
        st.warning("⚠️ Chưa tìm thấy dữ liệu trong thư mục `output/`. Vui lòng chạy lệnh `--write` trước khi mở UI:")
        st.code("python RAG/rag_foundation/buoi_05/src/process_rag.py --write", language="powershell")
        return

    # Sidebar
    st.sidebar.header("⚙️ Cấu Hình & Bộ Lọc")
    selected_doc = st.sidebar.selectbox("📄 Chọn tài liệu PDF:", list(data.keys()))
    
    doc_info = data[selected_doc]
    all_chunks = doc_info["chunks"]
    raw_info = doc_info["raw"]
    ocr_used = raw_info.get("ocr_used", False)

    # Display OCR status badge
    st.sidebar.markdown("---")
    if ocr_used:
        st.sidebar.markdown('Phương thức trích xuất: <span class="badge-ocr">⚡ LlamaParse OCR</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('Phương thức trích xuất: <span class="badge-pymupdf">📄 PyMuPDF Text Layer</span>', unsafe_allow_html=True)

    # Strategy filter
    strategies = ["Tất cả chiến lược", "fixed-size", "semantic", "hierarchical"]
    selected_strategy = st.sidebar.radio("🎯 Chọn chiến lược Chunking:", strategies)

    # Search filter
    search_keyword = st.sidebar.text_input("🔍 Tìm kiếm từ khóa trong Chunk:", "")

    # Filter chunks
    filtered_chunks = all_chunks
    if selected_strategy != "Tất cả chiến lược":
        filtered_chunks = [c for c in filtered_chunks if c["strategy"] == selected_strategy]

    if search_keyword.strip():
        filtered_chunks = [c for c in filtered_chunks if search_keyword.lower() in c["text"].lower()]

    # Main Metrics Dashboard
    col1, col2, col3, col4 = st.columns(4)
    lengths = [len(c["text"]) for c in filtered_chunks] if filtered_chunks else [0]
    
    col1.metric("📦 Tổng số Chunk", len(filtered_chunks))
    col2.metric("📏 Độ dài Min", f"{min(lengths)} ký tự")
    col3.metric("📏 Độ dài Max", f"{max(lengths)} ký tự")
    col4.metric("📊 Độ dài Trung bình", f"{round(sum(lengths)/len(lengths), 1) if lengths else 0} ký tự")

    st.markdown("---")

    # Tabs for Comparison vs Detailed View
    tab1, tab2, tab3 = st.tabs(["📊 So Sánh 3 Chiến Lược", "🔍 Danh Sách Chunks Chi Tiết", "📄 Văn Bản Thô (Raw Text)"])

    with tab1:
        st.subheader("📊 So sánh Thống kê giữa các Chiến lược Chunking")
        
        strat_stats = []
        for strat in ["fixed-size", "semantic", "hierarchical"]:
            strat_c = [c for c in all_chunks if c["strategy"] == strat]
            if strat_c:
                lens = [len(c["text"]) for c in strat_c]
                strat_stats.append({
                    "Chiến lược": strat,
                    "Số lượng Chunk": len(strat_c),
                    "Độ dài Min": min(lens),
                    "Độ dài Max": max(lens),
                    "Độ dài TB": round(sum(lens)/len(lens), 1)
                })
        
        df_stats = pd.DataFrame(strat_stats)
        st.dataframe(df_stats, use_container_width=True)

        if not df_stats.empty:
            st.markdown("#### Biểu đồ phân bổ số lượng Chunk:")
            st.bar_chart(df_stats.set_index("Chiến lược")["Số lượng Chunk"])

    with tab2:
        st.subheader(f"🔍 Danh Sách {len(filtered_chunks)} Chunks")
        
        if not filtered_chunks:
            st.info("Không tìm thấy Chunk nào phù hợp với bộ lọc.")
        else:
            for i, chunk in enumerate(filtered_chunks):
                with st.expander(f"📌 Chunk [{chunk['chunk_id']}] | Chiến lược: {chunk['strategy']} | Trang: {chunk['page_start']}-{chunk['page_end']} | Độ dài: {len(chunk['text'])} ký tự"):
                    
                    meta = chunk.get("structure_metadata", {})
                    if meta:
                        if "warning" in meta:
                            st.warning(f"⚠️ Cảnh báo: {meta['warning']}")
                        else:
                            st.info(f"🏛 Cấu trúc: **{meta.get('chuong', 'N/A')}** ➔ **{meta.get('dieu', 'N/A')}**")
                    
                    st.code(chunk["text"], language="markdown")

    with tab3:
        st.subheader("📄 Dữ liệu Text sau OCR / Trích xuất (Raw)")
        if raw_info.get("full_text"):
            st.text_area("Toàn bộ văn bản (NFC Normalized):", raw_info["full_text"], height=400)
        elif raw_info.get("pages"):
            for p in raw_info["pages"]:
                st.markdown(f"**Trang {p['page_num']}**")
                st.text_area(f"Trang {p['page_num']}:", p["text"], height=200, key=f"raw_page_{p['page_num']}")

if __name__ == "__main__":
    main()
