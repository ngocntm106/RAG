# Agent Specification & System Contract — Buổi 08: Advanced RAG System

## 1. Workspace và Security
- Vùng làm việc tuyệt đối: `rag_foundation/buoi_08/`.
- Không chỉnh sửa bất kỳ tài nguyên nào trong `buoi_05/`, `buoi_06/`, `buoi_07/`.
- Bảo mật tuyệt đối: Không commit hay lưu trữ `.env` thật hoặc API key trong repository. `.env` được khai báo trong `.gitignore`.

---

## 2. Quan hệ với Buổi 05 và Buổi 07
- **Buổi 05**: Nguồn dữ liệu chunks chuẩn hóa (`rag_foundation/buoi_05/output/chunks/`) và môi trường ảo Python (`rag_foundation/buoi_05/.venv/`).
- **Buổi 07**: Cung cấp baseline module (`rag_foundation/buoi_08/rag.py`), đã được sao chép độc lập và không phụ thuộc runtime vào Buổi 07.
- **Buổi 08**: Phát triển hệ thống Advanced RAG hoàn chỉnh (BM25 Lexical + Dense Semantic + RRF Fusion + Cross-Encoder Reranker + Pipeline Tracing + Evaluation Framework).

---

## 3. Data Contract
mỗi record chunk tuân thủ nghiêm ngặt schema JSON:
- `chunk_id`: String (duy nhất, không rỗng).
- `strategy`: String (`fixed-size`, `semantic`, `hierarchical`).
- `source`: String (tên file nguồn).
- `page_start`: Integer (>= 1).
- `page_end`: Integer (>= page_start).
- `text`: String (không rỗng sau khi strip).

---

## 4. BM25 Tokenizer & Retrieval Contract
- **Tokenizer**: Tách từ cho tiếng Việt bằng cách chuyển chữ thường, loại bỏ ký tự đặc biệt, giữ lại từ đơn và từ ghép cơ bản.
- **Index**: Xây dựng chỉ mục BM25 qua `rank_bm25.BM25Okapi` trên tập chunks đã validate.
- **Candidate Output**: Trả về danh sách `BM25_CANDIDATES` top records kèm điểm số BM25 và rank.

---

## 5. Semantic Candidate Contract
- Tái sử dụng `generate_query_embedding` và `ChromaDB PersistentClient` từ baseline `rag.py`.
- Truy xuất danh sách `SEMANTIC_CANDIDATES` top records dựa trên khoảng cách Cosine.

---

## 6. RRF Fusion Contract
- Điểm RRF cho từng tài liệu $d$:
  $$RRF(d) = w_{bm25} \cdot \frac{1}{k + r_{bm25}(d)} + w_{sem} \cdot \frac{1}{k + r_{sem}(d)}$$
- Tham số cấu hình: $k = 60$, $w_{bm25} = 1.0$, $w_{sem} = 1.0$.
- Union candidates được sắp xếp theo điểm $RRF(d)$ giảm dần, chọn ra danh sách `RERANK_CANDIDATES`.

---

## 7. Cross-Encoder Reranker Contract
- Model: `BAAI/bge-reranker-v2-m3` (hoặc Cross-Encoder tương thích).
- Đầu vào: Cặp `(query, chunk_text)`.
- Chấm điểm số tương đồng ngữ nghĩa thực tế.
- Lọc theo `RERANK_MIN_SCORE` (mặc định `0.50`).

---

## 8. Final Evidence và Citation Contract
- Chọn `FINAL_TOP_K` candidates có điểm số rerank cao nhất vượt ngưỡng.
- Ánh xạ nhãn `[E1]`, `[E2]` sang thông tin citation thực tế dạng `[Nguồn: ..., tr. N-M, chunk: ...]`.
- Xóa bỏ các nhãn giả lập không tồn tại (ví dụ `[E99]`) và đưa vào danh sách `warnings`.

---

## 9. Pipeline Trace Contract
Trả về dictionary có cấu trúc đầy đủ minh bạch từng bước:
```json
{
  "status": "answered | insufficient_evidence | retrieval_only",
  "question": "...",
  "answer": "...",
  "pipeline_trace": {
    "bm25_candidates": [...],
    "semantic_candidates": [...],
    "rrf_candidates": [...],
    "reranked_candidates": [...]
  },
  "evidence": [...],
  "citations": [...],
  "warnings": [...]
}
```

---

## 10. Evaluation Metrics Contract
Bộ chỉ số đo lường hiệu năng dựa trên `eval/questions.json`:
- **Hit Rate @ K (Hit@K)**: Tỷ lệ câu hỏi có ít nhất 1 relevant chunk nằm trong top-K.
- **MRR @ K**: Mean Reciprocal Rank vị trí xuất hiện đầu tiên của relevant chunk.
- **NDCG @ K**: Normalized Discounted Cumulative Gain đánh giá độ chính xác xếp hạng.

---

## 11. Offline Testing Contract
- Tất cả unit test trong `tests/` phải chạy 100% offline, dùng temporary directories, mock API calls và fake embeddings.
- Không kết nối Internet hay Hugging Face Hub trong quá trình test tự động.

---

## 12. UI Comparison Contract
Giao diện Streamlit (`app.py`) cung cấp chế độ so sánh:
- Cột bên trái: Kết quả Baseline (Buổi 07 Semantic Search).
- Cột bên phải: Kết quả Advanced RAG (BM25 + RRF + Reranker).
- Bảng hiển thị Pipeline Trace chi tiết minh bạch.
