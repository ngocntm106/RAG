import fitz  # PyMuPDF
import unicodedata
import re
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Tuple
from llama_cloud import AsyncLlamaCloud

def normalize_nfc(text: str) -> str:
    """Chuẩn hóa Unicode NFC cho văn bản tiếng Việt."""
    if not text:
        return ""
    return unicodedata.normalize('NFC', text)

def validate_text_quality(text: str) -> Tuple[bool, str]:
    """
    Kiểm tra chất lượng text layer trích xuất từ PyMuPDF.
    Phát hiện các tình huống lỗi:
    - Text rỗng (scan image không có OCR layer)
    - Ký tự replacement \ufffd hoặc null byte
    - Lỗi mã hóa font tiếng Việt (tỷ lệ glyph lỗi cao)
    """
    stripped = text.strip()
    if not stripped:
        return False, "Văn bản rỗng (trang scan không có text layer)"
    
    if "\ufffd" in stripped or "\x00" in stripped:
        return False, "Phát hiện ký tự lỗi font/encoding (replacement char \\ufffd)"
    
    # Kiểm tra tỷ lệ ký tự lạ / không in được
    printable_chars = sum(1 for c in stripped if c.isprintable() or c in "\n\r\t")
    if len(stripped) > 0 and (printable_chars / len(stripped)) < 0.85:
        return False, "Tỷ lệ ký tự lạ cao (nghi ngờ lỗi mã hóa font tiếng Việt)"
    
    return True, "Chất lượng text layer tốt"

async def extract_pdf_with_fallback(pdf_path: Path, api_key: str, force_ocr: bool = False) -> Dict[str, Any]:
    """
    Luồng trích xuất PDF độc lập:
    (1) Nếu force_ocr=True hoặc PyMuPDF bị lỗi, chạy LlamaParse OCR.
    (2) Đọc PDF bằng PyMuPDF nếu không force OCR.
    (3) Chuẩn hóa Unicode NFC.
    """
    doc_name = pdf_path.name
    pages_data = []
    need_ocr_fallback = False
    fallback_reasons = []

    if not force_ocr:
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    raw_text = page.get_text("text")
                    nfc_text = normalize_nfc(raw_text)
                    
                    is_valid, reason = validate_text_quality(nfc_text)
                    if not is_valid:
                        need_ocr_fallback = True
                        fallback_reasons.append(f"Trang {page_num + 1}: {reason}")
                    
                    pages_data.append({
                        "page_num": page_num + 1,
                        "text": nfc_text,
                        "ocr_used": False
                    })
                except Exception as page_err:
                    need_ocr_fallback = True
                    fallback_reasons.append(f"Lỗi đọc trang {page_num + 1}: {page_err}")
                    pages_data.append({
                        "page_num": page_num + 1,
                        "text": "",
                        "ocr_used": False
                    })
            doc.close()
        except Exception as e:
            need_ocr_fallback = True
            fallback_reasons.append(f"Lỗi mở PDF bằng PyMuPDF: {e}")

    # Nếu phát hiện trang bị lỗi hoặc yêu cầu force_ocr, fallback gọi API LlamaParse OCR
    if need_ocr_fallback or force_ocr:
        if force_ocr:
            print(f"\n 🚀 [FORCE OCR] Bắt buộc chạy LlamaParse OCR cho file '{doc_name}'...")
        else:
            print(f"\n ⚠️  [FALLBACK OCR] File '{doc_name}' kích hoạt LlamaParse OCR do:")
            for r in fallback_reasons[:3]:
                print(f"     - {r}")

        if not api_key or api_key == "KEY CỦA BẠN":
            print(f" ⚠️  [WARNING] API Key chưa khả dụng hoặc chưa hợp lệ. Sử dụng PyMuPDF layer hiện có.")
            return {
                "source": doc_name,
                "ocr_used": False,
                "pages": pages_data,
                "warning": "LlamaParse API Key chưa sẵn sàng, giữ text PyMuPDF"
            }

        try:
            print(f" 🚀 Đang gửi file lên LlamaParse OCR (tier='agentic')...")
            client = AsyncLlamaCloud(api_key=api_key)
            file_obj = await client.files.create(file=str(pdf_path), purpose="parse")
            
            result = await client.parsing.parse(
                file_id=file_obj.id,
                tier="agentic",
                version="latest",
                expand=["markdown_full"],
            )
            
            ocr_text = normalize_nfc(result.markdown_full)
            print(f" ✅  LlamaParse OCR thành công cho '{doc_name}'.")
            
            return {
                "source": doc_name,
                "ocr_used": True,
                "full_text": ocr_text,
                "pages": [{"page_num": 1, "text": ocr_text, "ocr_used": True}],
                "warning": None
            }
        except Exception as e:
            print(f" ❌ [ERROR] Lỗi gọi LlamaParse API: {e}. Giữ kết quả từ PyMuPDF.")
            return {
                "source": doc_name,
                "ocr_used": False,
                "pages": pages_data,
                "warning": f"Lỗi LlamaParse API: {e}"
            }

    return {
        "source": doc_name,
        "ocr_used": False,
        "pages": pages_data,
        "warning": None
    }
