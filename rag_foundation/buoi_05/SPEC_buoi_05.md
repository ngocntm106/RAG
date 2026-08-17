# SPECIFICATION: BUỔI 5 — RAG FOUNDATION (OCR & CHUNKING STRATEGIES)

## 1. Đầu vào (Inputs)
- Các file PDF tiếng Việt trong thư mục `RAG/rag_foundation/buoi_05/datademo/`.
- File cấu hình môi trường `RAG/rag_foundation/buoi_05/src/.env` chứa `LLAMA_CLOUD_API_KEY`.

## 2. Yêu cầu xử lý (Processing Requirements)

2. **Fallback LlamaParse OCR (Llama Cloud):**
   - Khi PyMuPDF thất bại hoặc phát hiện lỗi chất lượng text layer, tự động fallback gọi API `AsyncLlamaCloud` (`tier="agentic"`, `version="latest"`, `expand=["markdown_full"]`).
3. **Chuẩn hóa Unicode:**
   - Chuẩn hóa 100% văn bản đầu ra về chuẩn Unicode **NFC** (`unicodedata.normalize('NFC', text)`).
4. **Lưu trữ dữ liệu thô (Raw Output):**
   - Lưu kết quả trích xuất vào `RAG/rag_foundation/buoi_05/output/`.

## 3. Ba chiến lược Chunking (Chunking Strategies)
- **Fixed-size (`fixed-size`):** Chia nhỏ văn bản theo độ dài ký tự cố định (ví dụ: chunk_size=500, overlap=100).
- **Semantic (`semantic`):** Ngắt chunk theo ranh giới đoạn văn (`\n\n`, cách dòng), ưu tiên không ngắt giữa câu.
- **Hierarchical (`hierarchical`):** Chia theo cấu trúc văn bản pháp lý tiếng Việt (Chương → Mục → Điều → Khoản). 
  - *Cảnh báo bắt buộc:* Nếu tài liệu không có cấu trúc Chương/Điều, KHÔNG được tự bịa cấu trúc, phải ghi log cảnh báo và xử lý an toàn.

## 4. Metadata chuẩn của từng Chunk
Mỗi chunk phải chứa đầy đủ các trường:
- `chunk_id`: Mã định danh duy nhất (ví dụ: `fixed_001`, `sem_002`, `hier_003`)
- `strategy`: Tên chiến lược (`fixed-size`, `semantic`, `hierarchical`)
- `source`: Tên file PDF gốc
- `page_start`: Trang bắt đầu (1-indexed)
- `page_end`: Trang kết thúc (1-indexed)
- `text`: Nội dung văn bản tiếng Việt chuẩn Unicode NFC
- `structure_metadata`: Thông tin cấu trúc (Chương, Mục, Điều, Khoản) nếu có

## 5. Ràng buộc bảo mật & Giới hạn
- KHÔNG tạo vector embedding, KHÔNG lưu Vector Database, KHÔNG gọi LLM.
- KHÔNG in secret / API key ra console hay file log.
- KHÔNG ghi đè hay chỉnh sửa các file PDF gốc trong `datademo/`.
