# RAG Foundation — Buổi 09: Multi-query Expansion & Parent–Child Retrieval Engine

Hệ thống RAG nâng cao dành cho Văn bản Pháp luật Ngân hàng Việt Nam kết hợp kỹ thuật **Multi-Query Expansion (Fan-out)** và **Parent–Child Hierarchical Retrieval**.

---

## 1. Mục Tiêu & Khác Biệt Giữa Buổi 08 & Buổi 09

| Tiêu chí | Buổi 08 (Advanced RAG) | Buổi 09 (Hierarchical & Multi-Query RAG) |
|---|---|---|
| **Query Processing** | Single Query ($Q_0$) duy nhất | **Multi-Query Expansion**: Sinh $Q_1..Q_n$ đa góc nhìn từ $Q_0$ |
| **Retrieval Unit** | Flat Chunk (200-500 từ) đơn lẻ | **Child Chunk Retrieval ➔ Parent Window Return** |
| **Fusion Layer** | Single Hybrid RRF (BM25 + Semantic) | **Two-Stage RRF**: Inner RRF per query + Cross-Query RRF Fusion |
| **Context Window** | Đoạn văn cắt vụn, mất ngữ cảnh | **Parent Window (1000-6000 ký tự)** bao trọn Điều/Khoản |
| **Reranker Input** | Cặp $(Q_0, \text{Child Text})$ | Cặp **$(Q_0, \text{Parent Window Text})$** chuẩn hóa Sigmoid |

---

## 2. Sơ Đồ Kiến Trúc Pipeline (Two-Stage Fusion & Parent Expansion)

```
                       ┌────────────────────────────────┐
                       │  User Original Question (Q0)   │
                       └───────────────┬────────────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         │  Multi-Query Generator    │ (Gemini 3.5 Flash Lite)
                         └─────────────┬─────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
      Query Q0 (Original)        Query Q1 (Legal Term)      Query Q2 (Paraphrase)
            │                          │                          │
   [Hybrid Retrieval]         [Hybrid Retrieval]         [Hybrid Retrieval]
   (BM25 + Semantic)          (BM25 + Semantic)          (BM25 + Semantic)
            │                          │                          │
      Inner RRF Q0               Inner RRF Q1               Inner RRF Q2
            │                          │                          │
            └──────────────────────────┼──────────────────────────┘
                                       │
                                       ▼
                       ┌────────────────────────────────┐
                       │  Cross-Query RRF Fusion (MQ)   │ w(Q0)=1.5, w(Qi)=1.0
                       └───────────────┬────────────────┘
                                       │ Top Fused Child Hits
                                       ▼
                       ┌────────────────────────────────┐
                       │    Child-to-Parent Lookup      │ (Hierarchy Registry)
                       └───────────────┬────────────────┘
                                       │
                                       ▼
                       ┌────────────────────────────────┐
                       │   Parent Score Aggregation     │ Top-3 Scoring Children
                       └───────────────┬────────────────┘
                                       │ Top Parent Candidates
                                       ▼
                       ┌────────────────────────────────┐
                       │ Cross-Encoder Parent Rerank    │ (bge-reranker-v2-m3)
                       │ Pair: (Q0, Parent Window Text) │ Logit ➔ Sigmoid [0,1]
                       └───────────────┬────────────────┘
                                       │ Evidence Gate (score >= 0.35)
                                       ▼
                       ┌────────────────────────────────┐
                       │   Gemini Answer Generation     │ Prompt: Q0 + Evidence [P1]
                       └────────────────────────────────┘ (Max 2 Gen API Calls)
```

---

## 3. Bốn Pipeline Modes Comparison

1. **`single_flat`**: Single query ($Q_0$) ➔ Hybrid retrieval ➔ Rerank child chunk.
2. **`multi_flat`**: Multi-query ($Q_0 + Q_i$) ➔ Per-query hybrid ➔ MQ-RRF fusion ➔ Rerank child chunk bằng $Q_0$.
3. **`single_parent`**: Single query ($Q_0$) ➔ Hybrid retrieval ➔ Child-to-parent lookup & aggregation ➔ Rerank parent bằng $Q_0$.
4. **`multi_parent`**: Multi-query ($Q_0 + Q_i$) ➔ Per-query hybrid ➔ MQ-RRF ➔ Child-to-parent lookup & aggregation ➔ Rerank parent bằng $Q_0$ (Mode chính Buổi 09).

---

## 4. Cấu Trúc Thư Mục & Thiết Lập `.env`

### Thư mục Project (`rag_advanced/buoi_09/`):
```text
buoi_09/
├── .env.example
├── README.md
├── SPEC_buoi_09.md
├── hierarchical_rag.py     # Main Core Engine (Hierarchy, Expansion, RRF, Rerank, Pipeline)
├── advanced_rag.py         # Snapshot Buổi 08
├── rag.py                  # Snapshot Baseline
├── ui_helpers.py           # UI Data Formatter Helpers (100% offline)
├── app.py                  # Streamlit Dashboard App (5 Tabs)
├── evaluate.py             # Evaluation Benchmark Engine
├── eval/
│   └── questions.json      # Benchmark Dataset
├── storage/
│   └── hierarchy/          # Children, Parents, Manifest Registries
└── tests/                  # 73 Offline Unit Tests (100% PASS)
```

### Thiết lập File `.env`:
Tạo file `.env` tại thư mục làm việc theo mẫu `.env.example`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_GENERATION_MODEL=gemini-3.5-flash-lite
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

---

## 5. Build Hierarchy & Giải Thích Ambiguous / Warnings

### Quy Tắc Ưu Tiên Phân Loại Cấu Trúc (Precedence):
1. **`metadata`**: Nếu chunk có chứa sẵn thuộc tính `chapter` hoặc `article` trong metadata.
2. **`heading_inferred`**: Suy luận từ dòng đầu tiên nếu khớp regex tiêu chuẩn (`Điều \d+`, `Chương [I-X]+`). Các tham chiếu chéo nằm giữa đoạn (inline legal reference) bị bỏ qua.
3. **`carried_forward`**: Kế thừa cấu trúc từ chunk liền trước nếu cùng nguồn tài liệu (`source`).
4. **`document_fallback`**: Trái lại, gán về khối tài liệu mặc định của file nguồn.

### Warning & Ambiguous:
- **Ambiguous Child**: Đoạn văn bản nghi ngờ hoặc chứa mâu thuẫn tiêu đề.
- **Parent Warning**: Parent window chứa từ 2 trẻ ambiguous trở lên hoặc vượt quá kích thước khuyến nghị.

---

## 6. Query Expansion Contract & API Call Budget

- **Query Set Contract**: Luôn luôn bao gồm $Q_0$ nguyên văn (`origin="original"`, `focus="original_intent"`). Các câu hỏi biến thể $Q_1..Q_n$ mang `origin="generated"`.
- **Invented Article Guard**: Tự động loại bỏ các query sinh ra có chứa `Điều X` mà $Q_0$ ban đầu không nhắc tới.
- **API Call Budget Limit**: Tối đa **2 Gemini Generation API calls** trong một query `multi_parent` (1 call expansion + 1 call answer generation).

---

## 7. Công Thức Hai Tầng RRF & Parent Score Aggregation

### 1. Inner RRF (Tầng 1 - Trong từng Query):
$$\text{inner\_rrf}(d) = \frac{1}{60 + \text{rank}_{\text{BM25}}(d)} + \frac{1}{60 + \text{rank}_{\text{Semantic}}(d)}$$

### 2. Cross-Query RRF (Tầng 2 - Giữa các Query):
$$\text{multi\_query\_rrf\_score}(d) = \sum_{q \in Q} \frac{w(q)}{60 + \text{rank}_q(d)}$$
Trong đó: $w(Q_0) = 1.5$, $w(Q_i) = 1.0$.

### 3. Parent Score Aggregation:
$$\text{parent\_rrf\_score}(P) = \sum_{c \in \text{top-3 children of } P} \frac{1}{60 + \text{multi\_query\_rank}(c)}$$

---

## 8. Child Retrieval ➔ Parent Return ➔ Parent Rerank

- **Child Retrieval**: Tìm kiếm theo vector & từ khóa trên các child chunks nhỏ (200-500 từ) để đạt độ chính xác khớp thẻ cao.
- **Parent Return**: Tra cứu chính xác parent_id tương ứng từ Hierarchy Registry và trả về Parent Window (1000-6000 ký tự).
- **Parent Rerank**: Đưa cặp $(Q_0, \text{Parent Text})$ qua Cross-Encoder model (`BAAI/bge-reranker-v2-m3`). Tính `parent_rerank_score = sigmoid(logit)`.

---

## 9. Danh Sách Lệnh CLI

```powershell
# 1. Audit cấu trúc hierarchy (Read-only)
python rag_advanced/buoi_09/hierarchical_rag.py hierarchy-audit

# 2. Xây dựng và lưu trữ Hierarchy Registry
python rag_advanced/buoi_09/hierarchical_rag.py build-hierarchy

# 3. Kiểm tra trạng thái Registry (Read-only)
python rag_advanced/buoi_09/hierarchical_rag.py hierarchy-status

# 4. Sinh các biến thể Multi-Query
python rag_advanced/buoi_09/hierarchical_rag.py expand-query --question "Điều kiện vay vốn ngân hàng?"

# 5. Retrieval child chunk đa câu hỏi & Cross-query RRF
python rag_advanced/buoi_09/hierarchical_rag.py multi-child --question "Điều kiện vay vốn ngân hàng?"

# 6. Tra cứu Parent Documents từ child hits
python rag_advanced/buoi_09/hierarchical_rag.py parent-retrieve --question "Điều kiện vay vốn ngân hàng?" --mode multi_parent

# 7. Chạy End-to-End Query (Full Pipeline)
python rag_advanced/buoi_09/hierarchical_rag.py query --mode multi_parent --question "Điều kiện vay vốn ngân hàng?"

# 8. So sánh 4 modes (Retrieval & Rerank Only)
python rag_advanced/buoi_09/hierarchical_rag.py compare --question "Điều kiện vay vốn ngân hàng?"

# 9. Chạy Benchmark Evaluation Engine
python evaluate.py --k 3

# 10. Khởi chạy Streamlit Dashboard UI
python -m streamlit run rag_advanced/buoi_09/app.py
```

---

## 10. Giải Thích Parameter Controls & Context Budget

- `PER_QUERY_CANDIDATES` (mặc định 12): Số lượng child hits tối đa lấy ra ở từng query lẻ.
- `PARENT_CANDIDATES` (mặc định 10): Số lượng parent candidates tối đa đưa vào Cross-Encoder Reranker.
- `FINAL_PARENT_TOP_K` (mặc định 3): Số lượng parent evidence tối đa giữ lại sau khi rerank và qua Evidence Gate.
- `TOTAL_CONTEXT_MAX_CHARS` (mặc định 16,000): Hạn mức bối cảnh tối đa đưa vào prompt để tránh tràn cửa sổ Gemini LLM.

---

## 11. Evaluation Metrics & Giới Hạn Gold Labels

- **Metrics**: Child Recall@K, Parent Recall@K, MRR@K, nDCG@K (binary relevance), Latency (mean/p50).
- **Giới Hạn Gold Labels**: Các nhãn trong `eval/questions.json` có cờ `needs_human_review: true`. Hệ thống **không tự tuyên bố mode thắng tuyệt đối** nếu chưa có đối chiếu chính thức từ chuyên gia pháp lý.

---

## 12. Troubleshooting & Xử Lý Lỗi Thường Gặp

- **`hierarchy_not_ready`**: Thư mục `storage/hierarchy/` bị thiếu hoặc stale fingerprint. Chạy `python hierarchical_rag.py build-hierarchy`.
- **`reranker_unavailable`**: Thiếu cache model Cross-Encoder local. Chạy script download model `bge-reranker-v2-m3`.
- **`query_generation_unavailable`**: Chưa điền `GEMINI_API_KEY` trong file `.env`. Pipeline sẽ tự động chuyển về fallback dùng $Q_0$.
- **`insufficient_evidence`**: Không có parent nào đạt `RERANK_MIN_SCORE` (0.35). Thử hạ nhẹ ngưỡng `rerank_min_score`.

---

## 13. Tuyên Bố Trách Nhiệm (Disclaimer)

> ⚠️ **TUYÊN BỐ TRÁCH NHIỆM PHÁP LÝ**:
> Sản phẩm này là bài thực hành thử nghiệm kỹ thuật RAG (Retrieval-Augmented Generation) trên dữ liệu văn bản pháp luật ngân hàng Việt Nam. Câu trả lời sinh ra từ mô hình chỉ mang tính chất tham khảo kỹ thuật, **TUYỆT ĐỐI KHÔNG DÙNG LÀM TƯ VẤN PHÁP LÝ CHÍNH THỨC**.
