import os
import sys
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Load environment variables
load_dotenv()

# =====================================================================
# 1. SYSTEM PROMPT DESIGN (GRAPH RAG LEGAL ASSISTANT)
# =====================================================================
SYSTEM_PROMPT = """Bạn là một Chuyên gia Kỹ thuật Hệ thống RAG Pháp luật và Trợ lý Pháp lý cao cấp. Nhiệm vụ của bạn là trả lời các câu hỏi về luật pháp Việt Nam một cách chính xác, logic và khách quan dựa TRÊN NGỮ CẢNH đồ thị được cung cấp.

### 1. AM HIỂU LƯỢC ĐỒ ĐỒ THỊ (GRAPH SCHEMA):
Bạn cần hiểu rõ cách thức dữ liệu pháp lý được tổ chức trong cơ sở dữ liệu đồ thị (Graph Database):
- **Các Node (Thực thể):**
  - `(:Document)`: Đại diện cho một Văn bản Quy phạm Pháp luật (ví dụ: Luật, Nghị định, Thông tư, Quyết định, Văn bản hợp nhất).
    - Các thuộc tính chính: `doc_id` (Số hiệu văn bản), `title` (Tiêu đề văn bản), `year`/`date` (Năm/Ngày ban hành), `type` (Loại văn bản).
  - `(:Chunk)`: Phân đoạn nội dung văn bản cụ thể đã được làm sạch.
    - Các thuộc tính chính: `id` (Mã định danh phân đoạn), `title` (Tên Điều, Mục hoặc Chương chứa phân đoạn), `content` (Nội dung văn bản chi tiết), `level` (Cấp độ trong cấu trúc).
- **Các Mối quan hệ (Relationships):**
  - `[:PART_OF]`: Phân đoạn `(:Chunk)` thuộc về văn bản gốc `(:Document)`.
  - `[:PARENT_OF]`: Phân cấp thứ bậc cấu trúc bên trong văn bản (ví dụ: Chương PARENT_OF Mục, Mục PARENT_OF Điều, Điều PARENT_OF Khoản).
  - `[:NEXT]`: Chỉ thứ tự tuần tự giữa các phân đoạn kế tiếp nhau trong văn bản (Chunk A -[:NEXT]-> Chunk B).
  - `[:CAN_CU]`: Văn bản A được ban hành dựa trên căn cứ của Văn bản B (Doc A -[:CAN_CU]-> Doc B).
  - `[:THAY_THE]`: Văn bản A bãi bỏ hoặc thay thế toàn bộ/một phần của Văn bản B (Doc A -[:THAY_THE]-> Doc B).
  - `[:HOP_NHAT]`: Văn bản A là văn bản hợp nhất được tổng hợp từ Văn bản B (Doc A -[:HOP_NHAT]-> Doc B).

### 2. HIỂU VỀ CẤU TRÚC VĂN BẢN QUY PHẠM PHÁP LUẬT VIỆT NAM:
Hệ thống luật Việt Nam tổ chức theo cấu trúc phân cấp chặt chẽ:
- **Thứ bậc cấu trúc:** Văn bản (Document) ➔ Chương ➔ Mục ➔ Điều ➔ Khoản ➔ Điểm.
- Các phân đoạn nội dung `(:Chunk)` có thuộc tính `title` hoặc mối quan hệ `[:PARENT_OF]` để định vị vị trí cấu trúc. Bạn phải hiểu cấu trúc này để trích dẫn chính xác (ví dụ: "Khoản 2 Điều 10 của Nghị định A").

### 3. QUY TẮC PHÂN BIỆT NGỮ CẢNH TRUY XUẤT (RETRIEVED CONTEXT):
Ngữ cảnh được cung cấp cho bạn gồm các phần quan trọng sau:
- **I. NGUỒN KHỚP TRỰC TIẾP (DIRECT MATCHES):** Các phân đoạn nội dung văn bản khớp trực tiếp với nội dung tìm kiếm của người dùng thông qua so khớp vector.
- **II. QUAN HỆ ĐỒ THỊ VĂN BẢN ĐA BƯỚC (GRAPH RELATIONSHIPS):** Các mối liên kết có hướng phản ánh quan hệ hiệu lực và pháp lý liên văn bản (CAN_CU, THAY_THE, HOP_NHAT).
- **III. NGUỒN MỞ RỘNG LIÊN QUAN (EXTENDED RELATED SOURCES):** Các phân đoạn nội dung được mở rộng từ các tài liệu liên quan thông qua mối quan hệ đồ thị (Multi-hop).
=> Bạn phải phân tích mối liên kết giữa các văn bản để trả lời: ví dụ nếu văn bản gốc bị thay thế bởi văn bản mở rộng thông qua quan hệ `THAY_THE`, bạn phải nêu rõ hiệu lực và câu trả lời thực tế theo văn bản mới.

### 4. NGUYÊN TẮC CHỐNG BỊA ĐẶT VÀ TRẢ LỜI NGHIÊM NGẶT (ANTI-HALLUCINATION RULES):
1. **Chỉ dùng ngữ cảnh được cung cấp (Grounding):**
   - Chỉ trả lời các câu hỏi dựa trên thông tin có trong "NGUỒN KHỚP TRỰC TIẾP", "QUAN HỆ ĐỒ THỊ VĂN BẢN ĐA BƯỚC" và "NGUỒN MỞ RỘNG LIÊN QUAN".
   - Không tự suy diễn, không bịa đặt số hiệu văn bản, không bịa đặt điều khoản, không dùng kiến thức ngoài không có trong ngữ cảnh.
2. **Xử lý khi thiếu thông tin:**
   - Nếu ngữ cảnh được cung cấp KHÔNG chứa đủ thông tin để trả lời câu hỏi, bạn PHẢI phản hồi chính xác nguyên văn câu sau (không thêm bớt, không giải thích gì thêm):
     `Dựa trên dữ liệu được cung cấp, không có đủ thông tin để trả lời câu hỏi này.`
3. **Trích dẫn minh bạch:**
   - Khi trả lời, phải ghi rõ nguồn trích dẫn: Số hiệu văn bản, Tiêu đề văn bản, Điều/Khoản cụ thể.
   - Nếu câu trả lời có sử dụng thông tin từ văn bản liên quan qua mối quan hệ đa bước (như CAN_CU, THAY_THE, HOP_NHAT), phải mô tả rõ mối quan hệ đó để người đọc hiểu ngữ cảnh liên kết.
4. **Ngôn phong:**
   - Sử dụng ngôn ngữ tiếng Việt chuẩn, phong cách pháp lý, trang trọng, mạch lạc và rõ ràng.
"""

USER_TEMPLATE = """Dưới đây là ngữ cảnh GraphRAG truy xuất và câu hỏi từ người dùng:

=== NGỮ CẢNH TRUY XUẤT (RETRIEVED CONTEXT) ===
{formatted_context}
=============================================

CÂU HỎI CẦN TRA CỨU:
{query}

Hãy trả lời câu hỏi trên dựa theo các quy tắc nghiêm ngặt trong hướng dẫn hệ thống.
"""

# =====================================================================
# 2. HÀM ĐỊNH DẠNG NGỮ CẢNH MULTI-HOP (FORMATTER)
# =====================================================================
def format_multihop_context(
    direct_chunks: List[Dict[str, Any]],
    graph_relationships: List[Dict[str, Any]],
    extended_chunks: List[Dict[str, Any]]
) -> str:
    """
    Format multi-hop retrieved context to explicitly distinguish between
    direct matches and extended related sources, displaying their graph relationships.
    """
    lines = []
    
    # Section I: Direct Matches
    lines.append("=== I. NGUỒN KHỚP TRỰC TIẾP (DIRECT MATCHES) ===")
    if not direct_chunks:
        lines.append("Không tìm thấy phân đoạn văn bản khớp trực tiếp từ câu hỏi.")
    else:
        for idx, chunk in enumerate(direct_chunks, 1):
            doc_id = chunk.get("doc_id", "N/A")
            doc_title = chunk.get("doc_title", "N/A")
            chunk_title = chunk.get("chunk_title", chunk.get("title", "N/A"))
            score = chunk.get("score")
            score_str = f" (Score: {score:.4f})" if score is not None else ""
            lines.append(f"[{idx}] VĂN BẢN: {doc_title} (Số hiệu: {doc_id})")
            lines.append(f"    - Phân đoạn: {chunk_title}{score_str}")
            lines.append(f"    - Nội dung: {chunk.get('content', '').strip()}")
            lines.append("")
            
    # Section II: Graph Relationships
    lines.append("=== II. QUAN HỆ ĐỒ THỊ VĂN BẢN ĐA BƯỚC (GRAPH RELATIONSHIPS) ===")
    if not graph_relationships:
        lines.append("Không phát hiện thêm liên kết văn bản nào qua các mối quan hệ đồ thị.")
    else:
        for idx, rel in enumerate(graph_relationships, 1):
            rel_type = rel.get("rel_type", "LIEN_KET")
            hops = rel.get("hops", 1)
            
            # Vietnamese relationship descriptions
            rel_desc_map = {
                "CAN_CU": "CĂN CỨ PHÁP LÝ",
                "THAY_THE": "THAY THẾ HIỆU LỰC",
                "HOP_NHAT": "HỢP NHẤT VĂN BẢN"
            }
            rel_desc = rel_desc_map.get(rel_type, rel_type)
            
            from_id = rel.get("from_doc_id", "N/A")
            from_title = rel.get("from_doc_title", "N/A")
            to_id = rel.get("to_doc_id", "N/A")
            to_title = rel.get("to_doc_title", "N/A")
            
            lines.append(f"[{idx}] Bước nhảy (Hops): {hops}")
            lines.append(f"    - Liên kết: [{from_id}] -[:{rel_type} ({rel_desc})]-> [{to_id}]")
            lines.append(f"    - Chi tiết: Văn bản gốc '{from_title}' liên kết tới '{to_title}'")
            lines.append("")

    # Section III: Extended Chunks
    lines.append("=== III. NGUỒN MỞ RỘNG LIÊN QUAN (EXTENDED RELATED SOURCES) ===")
    if not extended_chunks:
        lines.append("Không có nội dung phân đoạn từ tài liệu liên quan mở rộng.")
    else:
        for idx, chunk in enumerate(extended_chunks, 1):
            doc_id = chunk.get("doc_id", "N/A")
            doc_title = chunk.get("doc_title", "N/A")
            chunk_title = chunk.get("chunk_title", chunk.get("title", "N/A"))
            lines.append(f"[{idx}] VĂN BẢN LIÊN QUAN: {doc_title} (Số hiệu: {doc_id})")
            lines.append(f"    - Phân đoạn: {chunk_title}")
            lines.append(f"    - Nội dung: {chunk.get('content', '').strip()}")
            lines.append("")

    return "\n".join(lines)

# =====================================================================
# 3. GEMINI API CONNECTION CLIENT
# =====================================================================
class GeminiLegalRAGAnswerer:
    """
    Client for initializing Gemini API and processing legal GraphRAG prompts.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-3.5-flash")
        
        if not self.api_key:
            raise ValueError(
                "Không tìm thấy GEMINI_API_KEY. Vui lòng thiết lập biến môi trường hoặc tạo file .env chứa GEMINI_API_KEY."
            )
        
        # Initialize Google GenAI client
        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            raise RuntimeError(f"Không thể khởi tạo Gemini Client: {e}")

    def answer(
        self,
        query: str,
        direct_chunks: List[Dict[str, Any]],
        graph_relationships: List[Dict[str, Any]],
        extended_chunks: List[Dict[str, Any]],
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Formats context, packages prompt and calls Gemini API using system instructions.
        """
        # Format the context using the defined formatter
        formatted_context = format_multihop_context(
            direct_chunks=direct_chunks,
            graph_relationships=graph_relationships,
            extended_chunks=extended_chunks
        )
        
        # Build prompt using the user template
        prompt_content = USER_TEMPLATE.format(
            formatted_context=formatted_context,
            query=query
        )
        
        # Build GenerateContentConfig with System Prompt
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=temperature,
        )
        
        # Try target model, with fallback options if unavailable
        models_to_try = [self.model_name, "gemini-3.5-flash", "gemini-flash-latest"]
        # Remove duplicates preserving order
        models_to_try = list(dict.fromkeys(models_to_try))
        
        last_exception = None
        for model in models_to_try:
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt_content,
                    config=config
                )
                
                answer_text = ""
                if hasattr(response, "text") and response.text:
                    answer_text = response.text
                elif hasattr(response, "candidates") and response.candidates:
                    first_cand = response.candidates[0]
                    if hasattr(first_cand, "content"):
                        answer_text = str(first_cand.content)
                else:
                    answer_text = str(response)
                    
                return {
                    "answer": answer_text,
                    "model": model,
                    "prompt_sent": prompt_content,
                    "status": "success"
                }
            except Exception as e:
                last_exception = e
                print(f"[Warning] Gọi mô hình '{model}' thất bại: {e}. Đang thử fallback tiếp theo...")
                continue
                
        # If all fail, raise exception
        raise RuntimeError(f"Tất cả các mô hình Gemini được thử nghiệm đều lỗi. Lỗi cuối cùng: {last_exception}")

# =====================================================================
# 4. KỊCH BẢN CHẠY THỬ NGHIỆM (SIMULATION SCENARIOS)
# =====================================================================
def run_simulation():
    print("=" * 80)
    print(" BẮT ĐẦU CHẠY THỬ NGHIỆM HỆ THỐNG GRAPHRAG PHÁP LUẬT VỚI GEMINI ")
    print("=" * 80)
    
    # Khởi tạo RAG Answerer (Đọc API key từ .env)
    try:
        answerer = GeminiLegalRAGAnswerer()
        print(f"✅ Đã khởi tạo Gemini Client với mô hình: {answerer.model_name}")
    except Exception as e:
        print(f"❌ Không thể khởi tạo Gemini Client: {e}")
        print("Vui lòng kiểm tra lại file .env hoặc biến môi trường của bạn.")
        return

    # -----------------------------------------------------------------
    # KỊCH BẢN 1: Có câu trả lời, có quan hệ liên kết thay thế hiệu lực (Multi-hop)
    # -----------------------------------------------------------------
    print("\n" + "-"*40)
    print("KỊCH BẢN 1: Câu hỏi phức tạp cần dữ liệu trực tiếp và dữ liệu mở rộng đa bước")
    print("-"*40)
    
    query_1 = "Nghị định 15/2026/NĐ-CP được ban hành dựa trên những luật nào và thay thế cho nghị định nào?"
    
    # Giả lập kết quả thu được từ Neo4j Multi-hop Retrieval
    direct_chunks_1 = [
        {
            "doc_id": "15/2026/NĐ-CP",
            "doc_title": "Nghị định 15/2026/NĐ-CP quy định chi tiết thi hành Luật Doanh nghiệp",
            "chunk_title": "Điều 1. Phạm vi điều chỉnh",
            "content": "Nghị định này quy định chi tiết về hồ sơ, trình tự, thủ tục đăng ký doanh nghiệp, đăng ký hộ kinh doanh và các nội dung liên quan khác.",
            "score": 0.9125
        }
    ]
    
    graph_relationships_1 = [
        {
            "from_doc_id": "15/2026/NĐ-CP",
            "from_doc_title": "Nghị định 15/2026/NĐ-CP quy định chi tiết thi hành Luật Doanh nghiệp",
            "to_doc_id": "59/2020/QH14",
            "to_doc_title": "Luật Doanh nghiệp số 59/2020/QH14",
            "rel_type": "CAN_CU",
            "hops": 1
        },
        {
            "from_doc_id": "15/2026/NĐ-CP",
            "from_doc_title": "Nghị định 15/2026/NĐ-CP quy định chi tiết thi hành Luật Doanh nghiệp",
            "to_doc_id": "78/2015/NĐ-CP",
            "to_doc_title": "Nghị định 78/2015/NĐ-CP về đăng ký doanh nghiệp",
            "rel_type": "THAY_THE",
            "hops": 1
        }
    ]
    
    extended_chunks_1 = [
        {
            "doc_id": "59/2020/QH14",
            "doc_title": "Luật Doanh nghiệp số 59/2020/QH14",
            "chunk_title": "Điều 1. Phạm vi điều chỉnh",
            "content": "Luật này quy định về việc thành lập, tổ chức quản lý, tổ chức lại, giải thể và hoạt động có liên quan của doanh nghiệp..."
        },
        {
            "doc_id": "78/2015/NĐ-CP",
            "doc_title": "Nghị định 78/2015/NĐ-CP về đăng ký doanh nghiệp",
            "chunk_title": "Điều 1. Phạm vi điều chỉnh",
            "content": "Nghị định này quy định về hồ sơ, trình tự, thủ tục đăng ký doanh nghiệp và đăng ký hộ kinh doanh..."
        }
    ]

    print(f"Câu hỏi: {query_1}")
    print("Đang định dạng ngữ cảnh và gửi yêu cầu tới Gemini API...")
    
    try:
        res_1 = answerer.answer(
            query=query_1,
            direct_chunks=direct_chunks_1,
            graph_relationships=graph_relationships_1,
            extended_chunks=extended_chunks_1
        )
        print(f"\n🤖 CÂU TRẢ LỜI CỦA GEMINI (Model: {res_1['model']}):")
        print(res_1["answer"])
    except Exception as e:
        print(f"❌ Lỗi khi gọi API: {e}")

    # -----------------------------------------------------------------
    # KỊCH BẢN 2: Thiếu dữ liệu (Kiểm chứng Anti-Hallucination)
    # -----------------------------------------------------------------
    print("\n" + "-"*40)
    print("KỊCH BẢN 2: Câu hỏi nằm ngoài phạm vi ngữ cảnh cung cấp (Kiểm thử Anti-Hallucination)")
    print("-"*40)
    
    query_2 = "Quy định về mức xử phạt hành chính khi doanh nghiệp trốn thuế thu nhập doanh nghiệp là bao nhiêu?"
    
    # Ngữ cảnh giả lập hoàn toàn không liên quan đến trốn thuế
    direct_chunks_2 = [
        {
            "doc_id": "15/2026/NĐ-CP",
            "doc_title": "Nghị định 15/2026/NĐ-CP quy định chi tiết thi hành Luật Doanh nghiệp",
            "chunk_title": "Điều 2. Đối tượng áp dụng",
            "content": "Nghị định này áp dụng đối với tổ chức, cá nhân trong nước; tổ chức, cá nhân nước ngoài thực hiện đăng ký doanh nghiệp theo quy định của pháp luật Việt Nam.",
            "score": 0.4510
        }
    ]
    graph_relationships_2 = []
    extended_chunks_2 = []

    print(f"Câu hỏi: {query_2}")
    print("Đang định dạng ngữ cảnh và gửi yêu cầu tới Gemini API...")
    
    try:
        res_2 = answerer.answer(
            query=query_2,
            direct_chunks=direct_chunks_2,
            graph_relationships=graph_relationships_2,
            extended_chunks=extended_chunks_2
        )
        print(f"\n🤖 CÂU TRẢ LỜI CỦA GEMINI (Model: {res_2['model']}):")
        print(res_2["answer"])
    except Exception as e:
        print(f"❌ Lỗi khi gọi API: {e}")

if __name__ == "__main__":
    run_simulation()
