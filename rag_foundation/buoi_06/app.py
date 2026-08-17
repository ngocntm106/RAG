import os
import importlib
import streamlit as st
import rag

importlib.reload(rag)

# 1. Page Config
st.set_page_config(page_title="RAG Buổi 6", page_icon="⚖️")

# 2. Sidebar - Trạng thái hệ thống
with st.sidebar:
    st.header("⚙️ Trạng thái hệ thống")
    
    # Lấy status từ backend
    stat = rag.status()
    db_type = stat.get("db_type", "")
    
    # PostgreSQL Status
    if db_type == "postgres":
        st.success("🐘 PostgreSQL: Đang kết nối")
    else:
        st.warning("🗄️ PostgreSQL: Disconnected (Đang dùng SQLite fallback)")
        
    # ChromaDB Status
    chroma_chunks = stat.get("chunks_chroma", 0)
    st.info(f"🧬 ChromaDB: {chroma_chunks} vectors")
    
    # Gemini Key Status
    if os.environ.get("GEMINI_API_KEY"):
        st.success("🔑 Gemini API Key: Đã cấu hình")
    else:
        st.error("🔑 Gemini API Key: Còn thiếu (Chỉ hoạt động ở chế độ Retrieval)")

# 3. Main Area
st.title("⚖️ Trợ lý RAG Pháp Luật")

# Nút Index
if st.button("🚀 Index dữ liệu (Buổi 5)"):
    with st.spinner("Đang xử lý dữ liệu và lưu vào Database..."):
        try:
            num = rag.index()
            st.success(f"✅ Hoàn tất! Đã index thành công {num} chunks.")
        except Exception as e:
            st.error(f"❌ Có lỗi khi index: {e}")

st.divider()

# Nhập câu hỏi
question = st.text_input("Nhập câu hỏi của bạn:")
max_k = max(1, chroma_chunks) if chroma_chunks > 0 else 10
k_docs = st.slider("Số lượng tài liệu truy xuất (Top-k):", min_value=1, max_value=max_k, value=min(3, max_k))

if st.button("🔍 Truy vấn") and question:
    with st.spinner("Đang tìm kiếm thông tin..."):
        try:
            # Pipeline: Question ➔ Top-k ➔ Gemini ➔ Answer
            result = rag.ask(question=question, k=k_docs)
            contexts = result.get("contexts", [])
            answer = result.get("answer", "")
            
            # Hiển thị Answer (nếu có LLM hoặc báo lỗi API Key)
            st.subheader("💡 Câu trả lời (Answer)")
            st.info(answer)
            
            # Hiển thị Top-k
            st.subheader(f"📑 Top-{k_docs} Tài liệu liên quan (Retrieval)")
            for i, ctx in enumerate(contexts, 1):
                with st.expander(f"Tài liệu {i}"):
                    st.text(ctx)
                    
        except Exception as e:
            st.error(f"Lỗi truy vấn: {e}")
