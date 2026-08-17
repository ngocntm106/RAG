# Specification — Buổi 09: Hierarchical + Multi-Query RAG

## 1. Mục tiêu và khác biệt Buổi 08 / Buổi 09

| Đặc điểm | Buổi 08 (Advanced RAG) | Buổi 09 (Hierarchical + Multi-Query RAG) |
|---|---|---|
| **Retrieval đơn vị** | Flat child chunk | Flat chunk hoặc Parent document |
| **Số lượng query** | 1 query (Q0) | Q0 + N variant queries (Multi-Query) |
| **Fusion** | BM25 + Semantic → RRF (single query) | Cross-query RRF trên kết quả từ nhiều Qi |
| **Context đưa vào LLM** | Top-K child chunks rời rạc | Top-K parent documents (đầy đủ ngữ cảnh Điều/Khoản) |
| **Reranker** | Cross-Encoder trên child | Cross-Encoder hoặc score-based trên parent |
| **Trích dẫn** | chunk_id + page child | parent_id + article heading + page range đầy đủ |
| **Rủi ro chính** | Chunk ngắn mất ngữ cảnh | Parent quá dài, variant query làm mất số Điều |

**Mục tiêu Buổi 09:**
1. Xây dựng Hierarchy Registry ánh xạ `chunk_id → parent_id`.
2. Triển khai Multi-Query Generation: LLM sinh N variant queries bảo toàn số Điều/Khoản.
3. Cross-query RRF fusion: gom kết quả từ Q0 + tất cả Qi, áp dụng weight khác nhau.
4. Child-to-Parent Resolution: từ child hits tra ngược parent document đầy đủ.
5. Parent Aggregation & Rerank: tổng hợp điểm children → xếp hạng parent, áp context budget.
6. Bốn mode có thể so sánh trực tiếp với Buổi 08 baseline.

---

## 2. Sơ đồ Pipeline Tổng thể

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT: Q0 (câu hỏi gốc)                                                    │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Multi-Query Generation      │
                    │  LLM → Q1, Q2, …, QN        │
                    │  (bảo toàn số Điều/Khoản)   │
                    └──────────────┬──────────────┘
                                   │  Q0 + Q1..QN
           ┌───────────────────────▼───────────────────────┐
           │          Per-Query Hybrid Retrieval            │
           │  Với mỗi Qi:                                   │
           │   BM25 Search (PER_QUERY_CANDIDATES)           │
           │    +                                           │
           │   Semantic Search (PER_QUERY_CANDIDATES)       │
           │    ↓                                           │
           │   RRF Fusion → child hit list [Ri]             │
           └───────────────────────┬───────────────────────┘
                                   │ R0, R1, …, RN
                    ┌──────────────▼──────────────┐
                    │   Cross-Query RRF Fusion     │
                    │   w0×R0 ⊕ w1×R1 ⊕ … ⊕ wN×RN│
                    │   → unified child hit list   │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Child-to-Parent Resolution  │
                    │  Hierarchy Registry lookup   │
                    │  chunk_id → parent_id        │
                    │  (warning nếu ambiguous)     │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │    Parent Aggregation        │
                    │  Σ child scores → parent     │
                    │  score (top PARENT_SCORE_    │
                    │  CHILD_LIMIT children/parent)│
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │    Parent Rerank             │
                    │  Cross-Encoder hoặc          │
                    │  aggregation-score order     │
                    │  → Top FINAL_PARENT_TOP_K   │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Context Budget Truncation  │
                    │   TOTAL_CONTEXT_MAX_CHARS    │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Generation + Citation      │
                    │   Gemini LLM                 │
                    │   Trích dẫn: parent_id,      │
                    │   article heading, page range│
                    └─────────────────────────────┘
```

---

## 3. Bốn Mode Retrieval

| Mode | Multi-Query | Parent Resolution | Mô tả |
|---|---|---|---|
| `single_flat` | ✗ (Q0 only) | ✗ (flat chunks) | Giống Buổi 08 — dùng làm baseline |
| `multi_flat` | ✓ (Q0+Qi) | ✗ (flat chunks) | Cross-query RRF nhưng kết quả vẫn là child chunk |
| `single_parent` | ✗ (Q0 only) | ✓ | Single-query → child → parent lookup |
| `multi_parent` | ✓ (Q0+Qi) | ✓ | **Main mode Buổi 09**: full pipeline |

**Quy tắc chung:**
- Tất cả mode đều dùng cùng corpus, cùng BM25 index và cùng Chroma collection.
- Tất cả mode đều cho phép so sánh Recall@K, MRR@K, nDCG@K trong evaluate.py.
- `single_flat` phải cho ra kết quả tương đương Buổi 08 `hybrid` mode (xác nhận regression).

---

## 4. QueryVariant Schema và Validation

```python
# Mỗi variant query được sinh ra phải tuân theo schema này:
QueryVariant = {
    "query_index": int,           # 0 = Q0 (original), 1..N = variants
    "text": str,                  # Nội dung câu hỏi
    "rrf_weight": float,          # MULTI_QUERY_ORIGINAL_WEIGHT cho Q0,
                                  # MULTI_QUERY_VARIANT_WEIGHT cho Qi
    "generation_method": str,     # "original" | "llm_generated"
}
```

**Validation rules:**
- `text` phải là string không rỗng sau khi strip.
- `len(text) <= MULTI_QUERY_MAX_CHARS`.
- `query_index` bắt đầu từ 0, tăng tuần tự.
- `rrf_weight > 0.0`.
- Variant query **không được** xóa hoặc thay thế số Điều/Khoản có trong Q0 gốc (kiểm tra bằng regex `r"Điều\s+\d+|khoản\s+\d+"` trước/sau sinh).
- Nếu LLM trả về variant vi phạm validation → bỏ qua variant đó và ghi warning vào trace, không raise exception.
- Nếu tất cả variants đều bị loại → fallback về single-query mode.

---

## 5. Hierarchy Registry Schema

```python
# Lưu tại storage/hierarchy/<strategy>_registry.json
HierarchyRegistry = {
    "version": str,              # "1.0"
    "strategy": str,             # "hierarchical"
    "source_files": List[str],   # danh sách file JSON nguồn
    "built_at": str,             # ISO8601 timestamp
    "parents": {
        "<parent_id>": {
            "parent_id": str,    # ví dụ: "TT_39_2016_NHNN:hierarchical:parent:Điều 8"
            "source": str,       # "TT_39_2016_NHNN.pdf"
            "article": str,      # heading Điều, ví dụ: "Điều 8. Những nhu cầu vốn không được cho vay"
            "chapter": str,      # heading Chương (nếu có), có thể null
            "page_start": int,
            "page_end": int,
            "text": str,         # văn bản đầy đủ của parent (toàn bộ Điều)
            "children": List[str]  # danh sách chunk_id thuộc parent này
        }
    },
    "child_to_parent": {
        "<chunk_id>": "<parent_id>"  # mapping ngược để lookup O(1)
    },
    "orphan_chunks": List[str],   # chunk_id không có parent (structure=null hoặc thiếu article)
    "ambiguous_warnings": List[{  # chunk thuộc nhiều article khác nhau
        "chunk_id": str,
        "candidate_parents": List[str],
        "chosen_parent": str,
        "reason": str
    }]
}
```

**Lưu ý quan trọng:**
- Registry chỉ được build từ hierarchical chunks thực tế — không hard-code số liệu.
- `orphan_chunks` tương ứng với các record có `structure=null` hoặc thiếu `structure.article`.
- Registry là read-only sau khi build — không tự update khi query.

---

## 6. ParentDocument Schema

```python
ParentDocument = {
    "parent_id": str,            # khóa trong Registry
    "source": str,
    "article": str,              # heading Điều đầy đủ
    "chapter": str | None,
    "page_start": int,
    "page_end": int,
    "text": str,                 # văn bản đầy đủ parent (đã truncate nếu > PARENT_MAX_CHARS)
    "text_truncated": bool,      # True nếu text bị cắt
    "child_ids": List[str],      # chunk_id của các children đã khớp retrieval
    "child_count": int,          # số children đã match (dùng scoring)
}
```

---

## 7. MultiQueryChildHit và ParentCandidate Schema

```python
# Kết quả sau Cross-Query RRF (trước parent resolution)
MultiQueryChildHit = {
    "chunk_id": str,
    "text": str,
    "source": str,
    "page_start": int,
    "page_end": int,
    "cross_query_rrf_score": float,
    "cross_query_rank": int,
    "per_query_ranks": {         # rank của chunk này trong từng Qi (nếu xuất hiện)
        "q0": int | None,
        "q1": int | None,
        # ...
    },
    "matched_by_queries": List[int],  # danh sách query_index đã tìm thấy chunk này
}

# Kết quả sau Parent Aggregation
ParentCandidate = {
    "parent_id": str,
    "source": str,
    "article": str,
    "chapter": str | None,
    "page_start": int,
    "page_end": int,
    "text": str,
    "text_truncated": bool,
    "aggregated_score": float,   # tổng cross_query_rrf_score của top-PARENT_SCORE_CHILD_LIMIT children
    "child_hit_count": int,      # số child chunks khớp
    "top_children": List[str],   # chunk_id của các children đóng góp vào score
    "rerank_score": float | None, # None nếu chưa rerank hoặc reranker không khả dụng
    "rerank_rank": int | None,
    "accepted": bool | None,
}
```

---

## 8. Quy tắc Hierarchy Resolution và Ambiguous Warning

### 8.1 Quy tắc xây dựng Registry

1. **Group by article**: Dùng `structure.article` làm key group. **KHÔNG** dùng riêng mình `structure.article` nếu thiếu — phải kết hợp thêm `source` để tránh nhầm lẫn giữa các văn bản.
2. **Parent ID format**: `"<source_stem>:hierarchical:parent:<article_heading>"` (ví dụ: `"TT_39_2016_NHNN:hierarchical:parent:Điều 8. Những nhu cầu vốn không được cho vay"`).
3. **Orphan handling**: Chunk thiếu `structure.article` → thêm vào `orphan_chunks`, không tạo parent giả.
4. **Parent text construction**: Ghép text tất cả children theo thứ tự `chunk_id` số. Nếu vượt `PARENT_MAX_CHARS`, cắt và đặt `text_truncated=True`.
5. **Page range**: `page_start = min(children.page_start)`, `page_end = max(children.page_end)`.

### 8.2 Ambiguous Warning

Xảy ra khi cùng một `chunk_id` xuất hiện với hai `structure.article` khác nhau (không bình thường, nhưng có thể xảy ra ở văn bản sửa đổi kiểu TT 06/2023 cite Điều từ TT 39/2016):

- Ưu tiên `structure.article` có trong `structure` của chính chunk đó.
- Nếu vẫn ambiguous → chọn parent có nhiều siblings hơn.
- Luôn ghi vào `ambiguous_warnings` với lý do rõ ràng.
- **KHÔNG** raise exception — pipeline vẫn tiếp tục.

### 8.3 Regex Heading Cảnh báo

Khi dùng regex để nhận diện Điều trong `text`, phải phân biệt:
- **Heading thực sự**: `^Điều\s+\d+\.` (đứng đầu dòng, theo sau bởi dấu chấm và tiêu đề).
- **Trích dẫn nội văn**: `theo quy định tại Điều\s+\d+`, `Sửa đổi.*Điều\s+\d+`, `khoản.*Điều\s+\d+`.

Chỉ dùng heading pattern cho việc xây dựng Registry — không dùng trích dẫn pattern.

---

## 9. Công thức Cross-Query RRF và Parent Aggregation

### 9.1 Per-Query RRF (trong mỗi Qi)

Giống Buổi 08 — kết hợp BM25 và Semantic:
$$\text{RRF\_Score}_{q_i}(d) = \frac{1}{k + \text{rank}_{\text{bm25}}(d)} + \frac{1}{k + \text{rank}_{\text{sem}}(d)}$$

với $k = $ `RRF_K` (mặc định 60).

### 9.2 Cross-Query RRF Fusion

Hợp nhất kết quả từ tất cả queries (Q0 có trọng số cao hơn):

$$\text{CrossRRF}(d) = \frac{w_0}{k_{cq} + \text{rank}_{R_0}(d)} + \sum_{i=1}^{N} \frac{w_i}{k_{cq} + \text{rank}_{R_i}(d)}$$

Trong đó:
- $w_0 = $ `MULTI_QUERY_ORIGINAL_WEIGHT` (mặc định 1.5)
- $w_i = $ `MULTI_QUERY_VARIANT_WEIGHT` (mặc định 1.0) cho $i \geq 1$
- $k_{cq} = $ `MULTI_QUERY_RRF_K` (mặc định 60)
- $\text{rank}_{R_i}(d)$ = vị trí của chunk $d$ trong danh sách kết quả của query $Q_i$ (nếu $d$ không xuất hiện trong $R_i$ → không cộng thêm, không phạt)

### 9.3 Parent Aggregation

Sau khi có `CrossRRF(d)` cho từng child chunk $d$:

$$\text{ParentScore}(p) = \sum_{d \in \text{top-}L\text{ children of }p} \text{CrossRRF}(d)$$

Trong đó $L = $ `PARENT_SCORE_CHILD_LIMIT` (mặc định 3). Chỉ tính `L` child có `CrossRRF` cao nhất của mỗi parent.

---

## 10. Context Budget và Citation Contract

### 10.1 Context Budget

- Tổng ký tự context đưa vào LLM không vượt `TOTAL_CONTEXT_MAX_CHARS` (mặc định 16,000).
- Nếu `FINAL_PARENT_TOP_K` parents vượt budget: cắt bớt parent cuối cùng (không cắt giữa câu — tìm ranh giới câu gần nhất).
- Mỗi parent text tối đa `PARENT_MAX_CHARS` ký tự (mặc định 6,000).
- Thứ tự ưu tiên: parent rank cao nhất → thêm vào context trước.

### 10.2 Citation Contract

- Trích dẫn được lấy **100% từ metadata thực tế** của `ParentDocument` — không tin LLM tự điền.
- Định dạng citation chuẩn:
  ```
  [N] Nguồn: <source>, <article_heading>, tr. <page_start>–<page_end>, parent: <parent_id>
  ```
- Ví dụ:
  ```
  [1] Nguồn: TT_39_2016_NHNN.pdf, Điều 8. Những nhu cầu vốn không được cho vay, tr. 5–6, parent: TT_39_2016_NHNN:hierarchical:parent:Điều 8. Những nhu cầu vốn không được cho vay
  ```
- Orphan chunks (không có parent) dùng citation format của Buổi 08 (chunk_id level).

---

## 11. Status / Failure Contract

```python
# Trả về từ hierarchical_status() và query_hierarchical_rag()
StatusResponse = {
    "status": str,         # "ok" | "no_candidates" | "retrieval_only" |
                           # "generation_error" | "registry_missing" |
                           # "reranker_unavailable" | "api_key_missing"
    "mode": str,           # mode đang chạy
    "warnings": List[str], # mọi warning không fatal
    "trace": {
        "query_variants": List[QueryVariant],
        "per_query_child_hits": {...},
        "cross_query_hits": List[MultiQueryChildHit],
        "parent_candidates": List[ParentCandidate],
        "context_chars_used": int,
        "context_truncated": bool,
        "latency_ms": {
            "multi_query_generation": float,
            "per_query_retrieval": float,
            "cross_query_rrf": float,
            "parent_resolution": float,
            "parent_aggregation": float,
            "rerank": float,
            "generation": float,
            "total": float,
        }
    },
    "answer": str | None,
    "citations": List[str],
    "evidence": List[ParentCandidate],
}
```

**Quy tắc failure:**
- `registry_missing`: Registry chưa build → báo lỗi rõ, không tạo registry on-the-fly.
- `api_key_missing`: Thiếu GEMINI_API_KEY → fallback `retrieval_only` (vẫn trả về evidence).
- `reranker_unavailable`: Cache reranker chưa có → dùng aggregation-score order, ghi warning.
- Mọi exception trong single stage → ghi warning, cố gắng tiếp tục (graceful degradation).

---

## 12. Testability và Dependency Injection

Tất cả hàm chính trong `hierarchical_rag.py` phải nhận dependency injection parameters:

```python
def query_hierarchical_rag(
    question: str,
    mode: str = "multi_parent",
    strategy: str = "hierarchical",
    config: Optional[Dict] = None,
    query_embedder: Optional[Callable] = None,   # DI: mock embedding
    llm_caller: Optional[Callable] = None,        # DI: mock LLM generation
    chroma_client: Optional[Any] = None,          # DI: mock Chroma client
    reranker_fn: Optional[Callable] = None,       # DI: mock reranker
    registry: Optional[Dict] = None,              # DI: pre-loaded registry
) -> Dict[str, Any]: ...
```

**Contract tests phải chạy 100% offline (không cần API key, không cần Chroma thật):**
- Dùng `query_embedder=lambda text: [0.0] * 768` (mock embedding).
- Dùng `llm_caller=lambda prompt: "Mock answer"` (mock LLM).
- Dùng `chroma_client=MockChromaClient(...)` (in-memory mock).
- Dùng `registry=load_json("tests/fixtures/hierarchical_sample.json")` (fixture).

---

## 13. Evaluation Metrics và Acceptance Criteria

### 13.1 Metrics

| Metric | Mô tả | Đơn vị |
|---|---|---|
| `recall_at_k` | Tỷ lệ relevant parent docs trong top-K | [0, 1] |
| `mrr_at_k` | Vị trí trung bình relevant parent đầu tiên | [0, 1] |
| `ndcg_at_k` | Quality xếp hạng với binary relevance | [0, 1] |
| `latency_ms` | Thời gian end-to-end mỗi query | ms |
| `latency_p50` | Median latency | ms |
| `latency_p95` | P95 latency | ms |
| `context_chars_mean` | Trung bình ký tự context dùng | chars |

### 13.2 Gold Labels

- Tất cả mục trong `eval/questions.json` có `needs_human_review=true` cho đến khi chuyên gia pháp lý xác nhận.
- Gold labels hiện tại ở cấp độ `relevant_chunk_ids` — evaluation phải map sang `relevant_parent_ids` theo Registry.
- **Không tuyên bố mode thắng chính thức** cho đến khi có xác nhận gold labels.

### 13.3 Acceptance Criteria (sơ bộ, chờ xác nhận gold)

| Mode | Recall@3 tối thiểu | MRR@3 tối thiểu | Latency P95 tối đa |
|---|---|---|---|
| `single_flat` | ≥ Buổi 08 hybrid baseline | ≥ Buổi 08 | ≤ Buổi 08 × 1.2 |
| `multi_flat` | ≥ `single_flat` | ≥ `single_flat` | ≤ `single_flat` × (N+1) |
| `single_parent` | ≥ `single_flat` | ≥ `single_flat` | ≤ `single_flat` × 1.5 |
| `multi_parent` | ≥ `single_parent` | ≥ `single_parent` | ≤ `single_parent` × (N+1) |

---

## 14. Xác nhận Phạm vi Ghi

> [!IMPORTANT]
> Tất cả code, dữ liệu trung gian, index, registry và báo cáo đánh giá của Buổi 09
> **chỉ được lưu trữ trong** `rag_advanced/buoi_09/`.
>
> Tuyệt đối **không chỉnh sửa** bất kỳ tài nguyên nào thuộc:
> - `rag_foundation/buoi_05/` — nguồn dữ liệu hierarchical chunks (read-only)
> - `rag_foundation/buoi_06/`, `buoi_07/`, `buoi_08/` — baseline code (read-only)
>
> `rag.py` và `advanced_rag.py` trong Buổi 09 là **bản snapshot** của Buổi 08,
> không import runtime từ directory `buoi_08`, không dùng storage của Buổi 08.
>
> Không commit file `.env` chứa API key thật.
> Toàn bộ unit test phải chạy **offline hoàn toàn** (không cần API key, không cần mạng).
