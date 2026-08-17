"""
Multi-hop Graph RAG Dashboard - Buổi 11
Interactive QA & Graph Traversal Explorer using Neo4j and Gemini.

Run with:
    python -m streamlit run rag_advanced/buoi_11/app.py
"""

import sys
import os
from pathlib import Path

# Add current directory to path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd

from neo4j_client import Neo4jClient
from embedder import VietnameseEmbedder
from retriever import MultiHopGraphRetriever
from generator import GeminiGenerator
from graph_rag import GraphRAGPipeline
from config import NEO4J_URI, NEO4J_DATABASE, EMBEDDING_MODEL_NAME, GEMINI_MODEL_NAME

# Predefined test questions from Buoi 11
PRESET_QUESTIONS = [
    "Nghị định 15/2026/NĐ-CP thay thế cho nghị định nào, và căn cứ vào những luật nào?",
    "Nghị định 15/2026/NĐ-CP được hợp nhất với văn bản nào, và quy định mô hình vector nhúng gồm những thông số gì?",
    "Thông tư 02/2024/TT-BTTTT quy chuẩn kỹ thuật dữ liệu căn cứ vào nghị định nào, và nghị định đó lại căn cứ vào luật nào?",
    "Nghị định 27/2018/NĐ-CP thay thế hoặc sửa đổi cho văn bản nào về quản lý Internet và thông tin trên mạng?",
    "Luật An ninh mạng số 24/2018/QH14 là căn cứ ban hành cho những nghị định nào trong hệ thống?",
]

@st.cache_resource(show_spinner="Đang kết nối CSDL Neo4j...")
def get_neo4j_client():
    client = Neo4jClient()
    client.connect()
    return client

@st.cache_resource(show_spinner="Đang tải mô hình nhúng tiếng Việt MSMARCO...")
def get_embedder():
    return VietnameseEmbedder.get_instance()

def get_pipeline(api_key: str = "", model_name: str = ""):
    client = get_neo4j_client()
    embedder = get_embedder()
    retriever = MultiHopGraphRetriever(neo4j_client=client, embedder=embedder)
    generator = GeminiGenerator(api_key=api_key or None, model_name=model_name or None)
    return GraphRAGPipeline(retriever=retriever, generator=generator)

def main():
    st.set_page_config(
        page_title="Multi-hop Graph RAG — Buổi 11",
        page_icon="🕸️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Dark-mode and Light-mode compatible CSS Styling
    st.markdown("""
        <style>
        .main-title {
            font-size: 2.1rem;
            font-weight: 700;
            color: #38BDF8;
            margin-bottom: 0.25rem;
        }
        .sub-title {
            font-size: 1.05rem;
            color: #94A3B8;
            margin-bottom: 1.5rem;
        }
        .stat-card {
            background-color: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        }
        .stat-val {
            font-size: 1.8rem;
            font-weight: 700;
            color: #38BDF8;
        }
        .stat-label {
            font-size: 0.82rem;
            color: #94A3B8;
            font-weight: 600;
            text-transform: uppercase;
        }
        .chunk-card {
            background-color: rgba(30, 41, 59, 0.85);
            border: 1px solid rgba(59, 130, 246, 0.4);
            border-left: 5px solid #38BDF8;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.85rem;
            color: #F1F5F9;
        }
        .chunk-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #60A5FA;
        }
        .chunk-content-box {
            background-color: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.15);
            color: #E2E8F0;
            padding: 10px 14px;
            border-radius: 6px;
            font-size: 0.95rem;
            margin-top: 8px;
            line-height: 1.5;
        }
        .rel-card {
            background-color: rgba(20, 83, 45, 0.25);
            border: 1px solid rgba(34, 197, 94, 0.4);
            border-left: 5px solid #22C55E;
            border-radius: 8px;
            padding: 0.9rem;
            margin-bottom: 0.65rem;
            color: #F1F5F9;
        }
        .rel-badge {
            display: inline-block;
            background-color: #166534;
            color: #86EFAC;
            padding: 3px 9px;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 700;
            margin-right: 6px;
        }
        .hop-badge {
            display: inline-block;
            background-color: #3730A3;
            color: #C7D2FE;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
        }
        .score-badge {
            display: inline-block;
            background-color: #78350F;
            color: #FDE68A;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 700;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<div class="main-title">🕸️ Multi-hop Graph RAG — Tra Cứu Đồ Thị Tri Thức Đa Bước</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title"><b>Pipeline:</b> Dense Vector Search (MSMARCO) ➔ Graph Traversal (Neo4j $N$-hops) ➔ Context Expansion ➔ Grounded LLM Generation</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cấu Hình Đồ Thị & RAG")
        
        try:
            client = get_neo4j_client()
            stats = client.get_database_statistics()
            st.success(f"🟢 Neo4j: Đã kết nối (`{stats['database']}`)")
            
            with st.expander("📊 Thống kê Database", expanded=False):
                st.write(f"**Tổng số Node:** {stats['total_nodes']}")
                for k, v in stats['node_counts'].items():
                    st.write(f"- {k}: {v}")
                st.write(f"**Tổng số Quan hệ:** {stats['total_relationships']}")
                for k, v in stats['relationship_counts'].items():
                    st.write(f"- `[:{k}]`: {v}")
        except Exception as e:
            st.error(f"🔴 Lỗi kết nối Neo4j: {e}")

        st.markdown("---")
        st.subheader("🔍 Tham số Truy vấn")
        max_hops = st.slider("Số bước nhảy đồ thị (Max Hops):", min_value=0, max_value=3, value=1, help="0: Vector Search thuần; 1: Quan hệ trực tiếp; 2: Mở rộng đa bước 2 tầng")
        top_k = st.slider("Số phân đoạn khớp trực tiếp (Top-K Chunks):", min_value=1, max_value=8, value=2)
        
        rel_options = ["CAN_CU", "THAY_THE", "HOP_NHAT"]
        selected_rels = st.multiselect(
            "Loại quan hệ duyệt (Relationship Types):",
            options=rel_options,
            default=rel_options,
        )

        st.markdown("---")
        st.subheader("🤖 LLM & Gemini API")
        api_key_input = st.text_input(
            "Gemini API Key:",
            type="password",
            placeholder="Nhập khóa API mới...",
            help="Nếu khóa hiện tại bị lỗi 403 / hết quota, bạn có thể dán API Key mới tại đây để gọi Gemini trực tiếp.",
        )
        model_name_input = st.text_input("Model Name:", value=GEMINI_MODEL_NAME)
        temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    # Metrics Summary Cards
    try:
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{stats["node_counts"].get("Document", 15)}</div><div class="stat-label">Văn bản Luật (Documents)</div></div>', unsafe_allow_html=True)
        with col_s2:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{stats["node_counts"].get("Chunk", 18)}</div><div class="stat-label">Phân đoạn (Chunks)</div></div>', unsafe_allow_html=True)
        with col_s3:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{stats["relationship_counts"].get("CAN_CU", 5) + stats["relationship_counts"].get("THAY_THE", 2) + stats["relationship_counts"].get("HOP_NHAT", 1)}</div><div class="stat-label">Quan hệ liên văn bản</div></div>', unsafe_allow_html=True)
        with col_s4:
            st.markdown(f'<div class="stat-card"><div class="stat-val">384</div><div class="stat-label">Chiều Vector (MiniLM)</div></div>', unsafe_allow_html=True)
    except Exception:
        pass

    st.markdown("<br>", unsafe_allow_html=True)

    # Question Selection & Input
    st.subheader("💬 Đặt Câu Hỏi Tra Cứu")
    
    preset_choice = st.selectbox(
        "Chọn câu hỏi mẫu từ bài thực hành:",
        options=["-- Nhập câu hỏi tùy chỉnh --"] + PRESET_QUESTIONS,
        index=1,
    )

    if preset_choice != "-- Nhập câu hỏi tùy chỉnh --":
        user_question = st.text_area("Câu hỏi tra cứu:", value=preset_choice, height=70)
    else:
        user_question = st.text_area("Câu hỏi tra cứu:", value="Nghị định 15/2026/NĐ-CP thay thế cho nghị định nào, và căn cứ vào những luật nào?", height=70)

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        run_btn = st.button("🚀 Chạy Tra Cứu Graph RAG", type="primary", use_container_width=True)
    with col_btn2:
        compare_btn = st.button("⚖️ So Sánh Đối Chứng 3 Chế Độ (0-Hop vs 1-Hop vs 2-Hops)", use_container_width=False)

    if run_btn:
        if not user_question.strip():
            st.warning("Vui lòng nhập nội dung câu hỏi.")
            return

        with st.spinner(f"Đang thực thi Graph RAG (Vector Search + {max_hops}-Hop Graph Traversal)..."):
            try:
                pipeline = get_pipeline(api_key=api_key_input, model_name=model_name_input)
                result = pipeline.query(
                    question=user_question,
                    top_k=top_k,
                    max_hops=max_hops,
                    temperature=temperature,
                )
            except Exception as e:
                err_str = str(e)
                if "7687" in err_str or "ServiceUnavailable" in err_str or "actively refused" in err_str:
                    st.error("❌ **CSDL Neo4j chưa được bật (Port 7687)**: Vui lòng mở ứng dụng **Neo4j Desktop** và nhấn nút **Start** trên Database của bạn, sau đó bấm nút thử lại bên dưới.")
                    if st.button("🔄 Thử kết nối lại Neo4j"):
                        st.cache_resource.clear()
                        st.rerun()
                else:
                    st.error(f"❌ **Lỗi thực thi**: {e}")
                return

        # Display Result
        st.markdown("---")
        st.subheader("💡 Kết Quả Trả Lời từ Graph RAG")

        # Answer Box
        if result["status"] == "success":
            st.success(result["answer"])
        elif result["status"] == "fallback_synthesized":
            st.markdown(result["answer"])
        else:
            st.info(f"**Phản hồi hệ thống:** {result['answer']}")

        # Tabs for details
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Phân đoạn Khớp Trực tiếp (Direct Chunks)",
            "🕸️ Mở rộng Quan hệ Đồ thị (Multi-hop Paths)",
            "📑 Toàn bộ Ngữ cảnh Truy vấn (Retrieved Context)",
            "🛠️ Prompt & Thông điệp Hệ thống",
        ])

        with tab1:
            candidates = result["retrieval"]["vector_candidates"]
            if not candidates:
                st.info("Không tìm thấy phân đoạn khớp trực tiếp.")
            else:
                for idx, c in enumerate(candidates, 1):
                    st.markdown(f"""
                        <div class="chunk-card">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                                <span class="chunk-title">[{idx}] {c['doc_title']}</span>
                                <span class="score-badge">Cosine Score: {c['score']:.4f}</span>
                            </div>
                            <div style="font-size:0.9rem; color:#94A3B8; margin-bottom:6px;">
                                <b>Phân đoạn:</b> {c['title']} | <b>Cấp:</b> {c['level']} | <b>ID:</b> <code>{c['chunk_id']}</code>
                            </div>
                            <div class="chunk-content-box">
                                {c['content']}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

        with tab2:
            if max_hops == 0:
                st.info("Chế độ 0-Hop: Không thực hiện duyệt đồ thị mở rộng quan hệ.")
            else:
                paths = result["retrieval"]["multi_hop"]["paths"]
                if not paths:
                    st.info("Không tìm thấy liên kết đồ thị bổ sung nào qua các bước nhảy quan hệ.")
                else:
                    st.write(f"**Tìm thấy {len(paths)} liên kết quan hệ đồ thị:**")
                    for idx, p in enumerate(paths, 1):
                        rel_str = " ➔ ".join([f"<span class='rel-badge'>[:{r}]</span>" for r in p["rel_names"]])
                        st.markdown(f"""
                            <div class="rel-card">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <div>
                                        <span class="hop-badge">{p['hops']} HOP(S)</span>
                                        {rel_str}
                                    </div>
                                    <span style="font-size:0.85rem; color:#CBD5E1;">Mục tiêu: <b>{p['target_id']}</b></span>
                                </div>
                                <div style="margin-top:8px; font-size:0.95rem; color:#F8FAFC;">
                                    <b>Gốc:</b> {p['seed_title']}<br>
                                    <b>➔ Đích:</b> {p['target_title']}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

        with tab3:
            st.text_area("Retrieved Context Text (Formatted):", value=result["retrieval"]["formatted_context"], height=250)

        with tab4:
            st.text_area("Prompt gửi đến LLM:", value=result["prompt"], height=250)
            with st.expander("Xem Prompt Hệ thống (System Instruction)"):
                st.markdown(f"```text\n{result['system_prompt']}\n```")

    elif compare_btn:
        st.markdown("---")
        st.subheader("⚖️ Bảng So Sánh Đối Chứng 3 Chế Độ (0-Hop vs 1-Hop vs 2-Hops)")
        
        with st.spinner("Đang chạy truy vấn so sánh trên cả 3 cấu hình..."):
            try:
                pipeline = get_pipeline(api_key=api_key_input, model_name=model_name_input)
                r0 = pipeline.query(question=user_question, top_k=top_k, max_hops=0)
                r1 = pipeline.query(question=user_question, top_k=top_k, max_hops=1)
                r2 = pipeline.query(question=user_question, top_k=top_k, max_hops=2)
            except Exception as e:
                err_str = str(e)
                if "7687" in err_str or "ServiceUnavailable" in err_str or "actively refused" in err_str:
                    st.error("❌ **CSDL Neo4j chưa được bật (Port 7687)**: Vui lòng mở ứng dụng **Neo4j Desktop** và nhấn nút **Start** trên Database của bạn, sau đó bấm nút thử lại.")
                    if st.button("🔄 Thử kết nối lại Neo4j", key="retry_compare"):
                        st.cache_resource.clear()
                        st.rerun()
                else:
                    st.error(f"❌ **Lỗi thực thi**: {e}")
                return

        c0, c1, c2 = st.columns(3)

        with c0:
            st.markdown("### 🔵 0-Hop (Vector thuần)")
            st.write(f"- **Direct Chunks:** {len(r0['retrieval']['vector_candidates'])}")
            st.write(f"- **Quan hệ đồ thị:** 0 paths")
            st.caption("Chỉ trích xuất các phân đoạn đơn lẻ khớp vector cosine.")
            st.text_area("Ngữ cảnh 0-Hop:", value=r0["retrieval"]["formatted_context"], height=220)

        with c1:
            st.markdown("### 🟢 1-Hop (Quan hệ trực tiếp)")
            st.write(f"- **Direct Chunks:** {len(r1['retrieval']['vector_candidates'])}")
            st.write(f"- **Quan hệ đồ thị:** {len(r1['retrieval']['multi_hop']['paths'])} paths")
            st.caption("Mở rộng quan hệ liên văn bản trực tiếp 1 bước.")
            st.text_area("Ngữ cảnh 1-Hop:", value=r1["retrieval"]["formatted_context"], height=220)

        with c2:
            st.markdown("### 🟣 2-Hops (Đa bước gián tiếp)")
            st.write(f"- **Direct Chunks:** {len(r2['retrieval']['vector_candidates'])}")
            st.write(f"- **Quan hệ đồ thị:** {len(r2['retrieval']['multi_hop']['paths'])} paths")
            st.caption("Mở rộng chuỗi bước nhảy đồ thị gián tiếp 2 tầng.")
            st.text_area("Ngữ cảnh 2-Hops:", value=r2["retrieval"]["formatted_context"], height=220)

if __name__ == "__main__":
    main()
