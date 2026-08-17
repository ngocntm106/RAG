# Bài thực hành 2: Tìm kiếm Đồ thị RAG Đa bước (Multi-hop Graph RAG) và Ứng dụng Hỏi đáp (QA)

## Mục tiêu
Xây dựng một hệ thống Graph RAG (Truy vấn tăng cường bằng đồ thị) bằng cách truy vấn các phân đoạn văn bản và các mối quan hệ được lưu trữ trong cơ sở dữ liệu Neo4j `lab1` từ Bài thực hành 1, thực hiện tìm kiếm đa bước (multi-hop) giữa các văn bản liên quan, và tạo câu trả lời tự động bằng Gemini API.

---

## Các bước thực hiện

### **Bước 1: Kết nối Cơ sở dữ liệu Neo4j**
- Kết nối tới thực thể Neo4j cục bộ bằng các thông tin đã được thiết lập ở Bài thực hành 1:
  - **Connection URL**: `neo4j://localhost:7687` hoặc `bolt://localhost:7687`
  - **Database Name**: `kb-hops`
  - **Credentials**: `neo4j / abcd1234` (hoặc mật khẩu của bạn)

### **Bước 2: Truy vấn Vector và Mối quan hệ Đa bước (Multi-hop)**
- Xây dựng một hàm tìm kiếm ngữ cảnh:
  - Chuyển đổi câu hỏi của người dùng thành vector nhúng bằng mô hình tiếng Việt MSMARCO.
  - Thực hiện tìm kiếm vector trong Neo4j để tìm ra $k$ phân đoạn phù hợp nhất.
  - **Mở rộng Đa bước (Multi-hop)**: Cho phép duyệt qua các mối quan hệ liên kết giữa các tài liệu (ví dụ: `CAN_CU`, `THAY_THE`, `HOP_NHAT`).
  - **Tính linh hoạt**: Cho phép người dùng cấu hình số lượng bước nhảy (ví dụ: $N$ bước nhảy từ tài liệu khớp gốc) để thu thập thêm các đoạn văn bản ngữ cảnh từ các tài liệu luật có liên quan.

### **Bước 3: Tích hợp Ngữ cảnh và Gọi LLM (Gemini API)**
- Kết nối ngữ cảnh đã truy vấn (các đoạn văn bản khớp trực tiếp + các đoạn văn bản từ tài liệu liên quan đa bước) vào Gemini API (`gemini-flash-latest`).
Nhiệm vụ của người học:
- Thiết kế và tinh chỉnh cấu trúc Prompt hệ thống cho LLM:
  - Cung cấp thông tin chi tiết về lược đồ dữ liệu đồ thị (schema) và cấu trúc của văn bản luật tiếng Việt.
  - Hướng dẫn mô hình trả lời chính xác dựa trên ngữ cảnh được cung cấp, nêu rõ nếu ngữ cảnh không có thông tin thay vì tự suy đoán.

### **Bước 4: Kiểm thử và Đánh giá Đường ống(pipeline)**
- Tạo **5 câu hỏi kiểm thử** đại diện cho các tình huống tra cứu luật phức tạp cần thông tin từ nhiều tài liệu liên quan:
  1. *Câu hỏi 1*: Nghị định 46/2023/NĐ-CP thay thế cho nghị định nào, và nghị định bị thay thế đó có nội dung gì nổi bật về kinh doanh bảo hiểm?
  2. *Câu hỏi 2*: Văn bản hợp nhất số 52/VBHN-NHNN được hợp nhất từ văn bản nào, và quy định về hồ sơ, thủ tục cấp giấy phép lần đầu của ngân hàng thương mại gồm những tài liệu gì?
  3. *Câu hỏi 3*: Thông tư số 01/2025/TT-NHNN quy định về cấp giấy phép quỹ tín dụng nhân dân được sửa đổi, bổ sung bởi văn bản nào, và những nội dung sửa đổi bổ sung chính là gì?
  4. *Câu hỏi 4*: Thông tư số 41/2016/TT-NHNN về tỷ lệ an toàn vốn của ngân hàng căn cứ vào luật nào, và luật đó quy định chức năng nhiệm vụ của cơ quan nào?
  5. *Câu hỏi 5*: Hoạt động giao nhận, vận chuyển tiền mặt và tài sản quý của Ngân hàng Nhà nước được điều chỉnh bởi Thông tư nào, và Thông tư đó có được sửa đổi bổ sung bởi văn bản nào không?

Nhiệm vụ người học:
- Chạy thử nghiệm trên hệ thống Hỏi đáp của bạn, so sánh câu trả lời thu được khi thay đổi số bước nhảy (ví dụ: so sánh giữa 0 bước, 1 bước và 2 bước nhảy) và ghi nhận kết quả đánh giá so sánh vào một tệp tin mới (ví dụ: `qa_comparison.md`) để chứng minh hiệu quả của ngữ cảnh đa bước.