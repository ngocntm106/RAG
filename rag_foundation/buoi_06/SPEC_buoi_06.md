# SPECIFICATION: BUỔI 06 — RAG FOUNDATION (INDEXING, RETRIEVAL & ANSWER GENERATION)

Tài liệu này quy định phạm vi, môi trường, quy chuẩn và ràng buộc dành cho AI Agent khi thực hiện công việc tại **Buổi 06**.

---

## 1. Phạm Vi Quyền Đọc Workspace (Workspace Access Control)

### Chỉ được phép đọc (Allowed):
- `RAG/rag_foundation/buoi_05/output/` (chứa các file dữ liệu chunk đã xử lý)
- `RAG/rag_foundation/buoi_05/.venv/` (môi trường Python chung)
- `RAG/rag_foundation/buoi_06/` (thư mục làm việc chính của Buổi 06)

### Nghiêm cấm đọc (Forbidden):
- Source code của Buổi 5 (`RAG/rag_foundation/buoi_05/src/`, `app.py`, ...)
- README hoặc SPEC các buổi trước
- Các file Notebook (`.ipynb`)
- Git history (`.git`)
- Bất kỳ thư mục nào khác ngoài phạm vi cho phép.

> **Quy tắc Black Box:** Buổi 5 được xem là Black Box hoàn toàn. Agent KHÔNG được phép reverse engineering hay phân tích cách Buổi 5 hoạt động.

---

## 2. Cấu Hình Môi Trường Python (Python Environment)
- **Interpreter:** Sử dụng đúng Python Virtual Environment có sẵn tại `RAG/rag_foundation/buoi_05/.venv/`.
- **Tuyệt đối KHÔNG** tạo thêm virtual environment mới (`.venv`).

---

## 3. Quản Lý Thư Viện (Package Management)
Chỉ cho phép sử dụng và cài đặt các thư viện sau:
- `streamlit`
- `google-genai`
- `chromadb`
- `psycopg`
- `python-dotenv`

> **Ràng buộc:** KHÔNG cài đặt thêm bất kỳ framework hay thư viện phức tạp nào khác (vd: LangChain, LlamaIndex,...).

---

## 4. Phong Cách Lập Trình (Coding Style)
- **Đơn giản hóa tối đa:** Ưu tiên ít file, ít class, ít function, viết code phẳng và dễ đọc.
- **Nghiêm cấm kiến trúc phức tạp:** KHÔNG áp dụng các thiết kế như Repository Pattern, Service Layer, Dependency Injection, Factory Pattern, Plugin Architecture,...

---

## 5. Phạm Vi Chức Năng (Scope of Work)
Chỉ thực hiện 4 thành phần cốt lõi:
1. **Index:** Đưa chunks từ Buổi 5 vào Vector Database (ChromaDB / PostgreSQL).
2. **Retrieval:** Truy vấn các chunk liên quan dựa trên câu hỏi.
3. **Answer:** Sử dụng LLM (`google-genai`) để tạo câu trả lời dựa trên context đã truy vấn.
4. **Streamlit:** Giao diện người dùng đơn giản (`app.py`).

> **Ràng buộc:** KHÔNG phát triển các chức năng nằm ngoài 4 yêu cầu trên.

---

## 6. Xử Lý Lỗi & Báo Cáo (Error Handling)
- Chỉ áp dụng `try/except` ở mức độ tối thiểu phục vụ luồng chạy chính.
- KHÔNG xây dựng hệ thống retry tự động, logging phức tạp hoặc monitoring server.

---

## 7. Bảo Mật (Security Constraints)
- KHÔNG in (print/log) các thông tin nhạy cảm: `API Key`, `password`, `secret` ra console, log file hay giao diện UI.

---

## 8. Giới Hạn Kích Thước Code (Code Size Limit)
- **Mục tiêu:** Tổng kích thước code Python dự kiến trong khoảng **300 – 500 dòng**.
- **Giới hạn tối đa:** Nếu tổng số dòng code vượt quá **700 dòng**, Agent phải chủ động tái cấu trúc và đơn giản hóa thiết kế.
