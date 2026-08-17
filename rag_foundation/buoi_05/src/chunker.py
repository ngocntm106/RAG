import re
from typing import List, Dict, Any

def fixed_size_chunking(
    doc_data: Dict[str, Any], 
    chunk_size: int = 500, 
    overlap: int = 100
) -> List[Dict[str, Any]]:
    """Chiến lược Fixed-size chunking cắt theo số ký tự với gối đầu (overlap)."""
    chunks = []
    source = doc_data["source"]
    chunk_counter = 1

    full_text = ""
    page_map = []
    
    for page in doc_data["pages"]:
        p_num = page["page_num"]
        txt = page["text"]
        start_idx = len(full_text)
        full_text += txt + "\n"
        end_idx = len(full_text)
        page_map.append((start_idx, end_idx, p_num))

    if not full_text.strip():
        return []

    start = 0
    text_len = len(full_text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_str = full_text[start:end].strip()

        if chunk_str:
            p_start = 1
            p_end = 1
            for p_s, p_e, p_n in page_map:
                if start >= p_s and start < p_e:
                    p_start = p_n
                if end > p_s and end <= p_e:
                    p_end = p_n

            chunks.append({
                "chunk_id": f"fixed_{chunk_counter:03d}",
                "strategy": "fixed-size",
                "source": source,
                "page_start": p_start,
                "page_end": max(p_start, p_end),
                "text": chunk_str,
                "structure_metadata": {}
            })
            chunk_counter += 1

        if end == text_len:
            break
        start += (chunk_size - overlap)

    return chunks

def semantic_chunking(
    doc_data: Dict[str, Any], 
    target_chunk_size: int = 600
) -> List[Dict[str, Any]]:
    """
    Chiến lược Semantic chunking: Ngắt theo đoạn văn (\n\n, cách dòng), 
    nếu đoạn dài thì tách theo câu để ưu tiên không cắt giữa câu.
    """
    chunks = []
    source = doc_data["source"]
    chunk_counter = 1

    for page in doc_data["pages"]:
        p_num = page["page_num"]
        text = page["text"]
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

        units = []
        for para in paragraphs:
            if len(para) > target_chunk_size:
                sentences = [s.strip() for s in re.split(r'(?<=[.!?;\n])\s+', para) if s.strip()]
                units.extend(sentences)
            else:
                units.append(para)

        current_units = []
        current_len = 0

        for unit in units:
            unit_len = len(unit)
            if current_len + unit_len > target_chunk_size and current_units:
                chunk_str = "\n\n".join(current_units)
                chunks.append({
                    "chunk_id": f"sem_{chunk_counter:03d}",
                    "strategy": "semantic",
                    "source": source,
                    "page_start": p_num,
                    "page_end": p_num,
                    "text": chunk_str,
                    "structure_metadata": {}
                })
                chunk_counter += 1
                current_units = [unit]
                current_len = unit_len
            else:
                current_units.append(unit)
                current_len += unit_len + 2

        if current_units:
            chunk_str = "\n\n".join(current_units)
            chunks.append({
                "chunk_id": f"sem_{chunk_counter:03d}",
                "strategy": "semantic",
                "source": source,
                "page_start": p_num,
                "page_end": p_num,
                "text": chunk_str,
                "structure_metadata": {}
            })
            chunk_counter += 1

    return chunks

def hierarchical_chunking(doc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Chiến lược Hierarchical chunking dành cho văn bản có cấu trúc tiếng Việt (Chương → Mục → Điều → Khoản).
    RÀNG BUỘC: Nếu không có cấu trúc, KHÔNG được tự bịa heading, phải in WARNING và fallback ngắt theo đoạn.
    """
    chunks = []
    source = doc_data["source"]
    chunk_counter = 1

    full_text = ""
    for page in doc_data["pages"]:
        full_text += f"\n--- Trang {page['page_num']} ---\n" + page["text"]

    chuong_pattern = r'(Chương\s+[IVXLCDM\d]+\.?:?\s*[^\n]*)'
    dieu_pattern = r'(Điều\s+\d+\.\s*[^\n]*)'

    has_chuong = bool(re.search(chuong_pattern, full_text, re.IGNORECASE))
    has_dieu = bool(re.search(dieu_pattern, full_text, re.IGNORECASE))

    if not (has_chuong or has_dieu):
        print(f" ⚠️  [WARNING] File '{source}': Không tìm thấy cấu trúc Chương/Điều trong văn bản. Không bịa cấu trúc giả.")
        for page in doc_data["pages"]:
            p_num = page["page_num"]
            chunks.append({
                "chunk_id": f"hier_{chunk_counter:03d}",
                "strategy": "hierarchical",
                "source": source,
                "page_start": p_num,
                "page_end": p_num,
                "text": page["text"].strip(),
                "structure_metadata": {
                    "warning": "Văn bản không có cấu trúc Chương/Điều chuẩn"
                }
            })
            chunk_counter += 1
        return chunks

    dieu_splits = re.split(r'(?=\bĐiều\s+\d+\.)', full_text)
    current_chuong = "Chủ thể chung"
    current_dieu = ""

    for block in dieu_splits:
        block_str = block.strip()
        if not block_str:
            continue

        chuong_match = re.search(r'Chương\s+[IVXLCDM\d]+\.?:?\s*([^\n]*)', block_str, re.IGNORECASE)
        if chuong_match:
            current_chuong = chuong_match.group(0).strip()

        dieu_match = re.search(r'Điều\s+(\d+)\.\s*([^\n]*)', block_str)
        if dieu_match:
            dieu_num = dieu_match.group(1)
            dieu_title = dieu_match.group(2).strip()
            current_dieu = f"Điều {dieu_num}. {dieu_title}"

        page_match = re.search(r'--- Trang (\d+) ---', block_str)
        p_num = int(page_match.group(1)) if page_match else 1

        chunks.append({
            "chunk_id": f"hier_{chunk_counter:03d}",
            "strategy": "hierarchical",
            "source": source,
            "page_start": p_num,
            "page_end": p_num,
            "text": block_str,
            "structure_metadata": {
                "chuong": current_chuong,
                "dieu": current_dieu
            }
        })
        chunk_counter += 1

    return chunks

def calculate_chunk_stats(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tính thống kê số lượng chunk, min-len, max-len, avg-len."""
    if not chunks:
        return {"count": 0, "min_len": 0, "max_len": 0, "avg_len": 0}
    
    lengths = [len(c["text"]) for c in chunks]
    return {
        "count": len(chunks),
        "min_len": min(lengths),
        "max_len": max(lengths),
        "avg_len": round(sum(lengths) / len(lengths), 1)
    }
