import sys
import json
import argparse
import asyncio
from pathlib import Path

# Fix UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from config import DATADEMO_DIR, OUTPUT_DIR, load_api_key
from ocr_engine import extract_pdf_with_fallback
from chunker import (
    fixed_size_chunking, 
    semantic_chunking, 
    hierarchical_chunking, 
    calculate_chunk_stats
)

async def process_all(write_to_disk: bool = False, force_ocr: bool = False):
    api_key = load_api_key()
    
    if not DATADEMO_DIR.exists():
        print(f" ❌ Thư mục dữ liệu '{DATADEMO_DIR}' không tồn tại.")
        return

    pdf_files = list(DATADEMO_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f" ⚠️  Không tìm thấy file PDF nào trong '{DATADEMO_DIR}'.")
        return

    print("=" * 80)
    print(" LUỒNG XỬ LÝ OCR & CHUNKING ĐỘC LẬP (BUỔI 5)")
    print(f" Chế độ: {'--write (LƯU KẾT QUẢ VÀO OUTPUT/)' if write_to_disk else '--dry-run (CHỈ HIỂN THỊ THỐNG KÊ)'}")
    if force_ocr:
        print(" Tùy chọn: --force-ocr (ÉP CHẠY LLAMAPARSE OCR TOÀN BỘ FILE)")
    print("=" * 80)

    if write_to_disk:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for pdf_path in pdf_files:
        print(f"\n📄 Đang xử lý file: {pdf_path.name}")
        
        # 1. OCR / Extraction with PyMuPDF & LlamaParse Fallback
        doc_data = await extract_pdf_with_fallback(pdf_path, api_key, force_ocr=force_ocr)
        
        # 2. Áp dụng 3 chiến lược Chunking
        fixed_chunks = fixed_size_chunking(doc_data)
        sem_chunks = semantic_chunking(doc_data)
        hier_chunks = hierarchical_chunking(doc_data)
        
        # 3. Tính toán thống kê cho 3 chiến lược
        stats = {
            "fixed-size": calculate_chunk_stats(fixed_chunks),
            "semantic": calculate_chunk_stats(sem_chunks),
            "hierarchical": calculate_chunk_stats(hier_chunks)
        }
        
        # 4. Hiển thị thống kê dạng bảng
        print("-" * 75)
        print(f" BẢNG THỐNG KÊ CHUNKING CHO FILE: {pdf_path.name}")
        print("-" * 75)
        print(f"{'Chiến lược Chunking':<20} | {'Số lượng':<10} | {'Độ dài Min':<12} | {'Độ dài Max':<12} | {'Độ dài Trung bình':<18}")
        print("-" * 75)
        for strat_name, s in stats.items():
            print(f"{strat_name:<20} | {s['count']:<10} | {s['min_len']:<12} | {s['max_len']:<12} | {s['avg_len']:<18}")
        print("-" * 75)

        # 5. Hiển thị 1 ví dụ metadata mẫu
        sample_chunk = (hier_chunks or sem_chunks or fixed_chunks)[0] if (hier_chunks or sem_chunks or fixed_chunks) else None
        if sample_chunk:
            print("\n 📌 VÍ DỤ METADATA MẪU CỦA 1 CHUNK:")
            print(json.dumps(sample_chunk, ensure_ascii=False, indent=2))

        # 6. Nếu bật --write, lưu file vào folder output/
        if write_to_disk:
            raw_out = OUTPUT_DIR / f"{pdf_path.stem}_raw.json"
            chunks_out = OUTPUT_DIR / f"{pdf_path.stem}_chunks.json"
            
            raw_out.write_text(json.dumps(doc_data, ensure_ascii=False, indent=2), encoding="utf-8")
            all_chunks = fixed_chunks + sem_chunks + hier_chunks
            chunks_out.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
            
            print(f" 💾 Đã lưu file raw vào: {raw_out.relative_to(OUTPUT_DIR.parent.parent.parent)}")
            print(f" 💾 Đã lưu file chunks vào: {chunks_out.relative_to(OUTPUT_DIR.parent.parent.parent)}")

    print("\n" + "=" * 80)
    print(" HOÀN THÀNH LUỒNG XỬ LÝ!")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Luồng xử lý OCR và Chunking Buổi 5")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Chạy thử nghiệm không lưu đĩa")
    group.add_argument("--write", action="store_true", help="Chạy thực tế và ghi kết quả vào folder output/")
    parser.add_argument("--force-ocr", action="store_true", help="Ép chạy LlamaParse OCR cho tất cả các file PDF")

    args = parser.parse_args()
    asyncio.run(process_all(write_to_disk=args.write, force_ocr=args.force-ocr if hasattr(args, "force-ocr") else args.force_ocr))

if __name__ == "__main__":
    main()
