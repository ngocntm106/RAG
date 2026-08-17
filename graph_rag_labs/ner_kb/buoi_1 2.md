# Bài thực hành 3: Dự đoán và Tự động hoàn thiện Quan hệ giữa các Văn bản bằng LLM

## Mục tiêu
Sử dụng mô hình ngôn ngữ lớn (LLM) để phân tích các văn bản luật trong tập dữ liệu `medium` (có mối quan hệ bị ẩn trong `lab/relationships.csv`), tự động phát hiện và dự đoán các mối quan hệ (như `CAN_CU`, `THAY_THE`, `SUA_DOI_BO_SUNG`), xác minh kết quả và chạy lại quy trình nạp dữ liệu.

---

## Các bước thực hiện

### **Bước 1: Phân tích Dữ liệu và Dự đoán Mối quan hệ bằng LLM**
- Kiểm tra tập dữ liệu trong thư mục `lab/` (bao gồm `metadata.csv` và `content.csv` chứa 30 tài liệu, nhưng `relationships.csv` hoàn toàn trống).
- Viết một script để gửi thông tin tiêu đề và nội dung của các cặp tài liệu tiềm năng cho LLM (sử dụng Gemini API).
- Yêu cầu LLM dự đoán xem giữa hai tài liệu có mối quan hệ pháp lý nào hay không và xác định loại quan hệ tương ứng (`CAN_CU`, `THAY_THE`, `SUA_DOI_BO_SUNG`, `HOP_NHAT`, v.v.).

### **Bước 2: Đối sánh và Xác minh Kết quả**
- So sánh các mối quan hệ do LLM dự đoán với kết quả chuẩn nằm trong thư mục `medium/relationships.csv` (được coi là bộ nhãn kiểm thử chuẩn).
- Tính toán các chỉ số đánh giá độ chính xác (Precision, Recall, F1-Score) để xem mô hình ngôn ngữ nhận diện các liên kết pháp lý tốt đến mức nào.

### **Bước 3: Tái nạp dữ liệu Đồ thị mở rộng**
- Lưu các mối quan hệ do LLM dự đoán (hoặc sau khi đã được hiệu chỉnh) vào tệp `lab/relationships.csv`.
- Thực hiện chạy lại quy trình nạp dữ liệu của Bài thực hành 1 với tập dữ liệu đầy đủ 30 tài liệu mới để xây dựng một đồ thị tri thức hoàn chỉnh.