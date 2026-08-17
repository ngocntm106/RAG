import sys
import importlib
from pathlib import Path

# Fix UTF-8 encoding for Windows terminal output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

REQUIRED_TOOLS = [
    {
        "name": "Python",
        "module": None,
        "min_version": "3.10",
        "install_cmd": "Tải Python 3.10+ từ python.org",
        "description": "Môi trường thực thi Python"
    },
    {
        "name": "PyMuPDF (fitz)",
        "module": "fitz",
        "install_cmd": "pip install pymupdf",
        "description": "Đọc và trích xuất text layer từ PDF"
    },
    {
        "name": "Pillow (PIL)",
        "module": "PIL",
        "install_cmd": "pip install pillow",
        "description": "Xử lý ảnh khi OCR render trang PDF"
    },
    {
        "name": "Llama Cloud (llama-cloud)",
        "module": "llama_cloud",
        "install_cmd": "pip install llama-cloud",
        "description": "Thư viện client gọi LlamaParse OCR"
    },
    {
        "name": "Pydantic",
        "module": "pydantic",
        "install_cmd": "pip install pydantic",
        "description": "Kiểm tra cấu trúc và kiểu dữ liệu Metadata"
    },
    {
        "name": "Streamlit",
        "module": "streamlit",
        "install_cmd": "pip install streamlit",
        "description": "Giao diện trực quan hoá chunking"
    },
    {
        "name": "python-dotenv",
        "module": "dotenv",
        "install_cmd": "pip install python-dotenv",
        "description": "Đọc cấu hình môi trường từ file .env"
    }
]

def check_env():
    results = []
    all_pass = True

    # 1. Check Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_pass = sys.version_info >= (3, 10)
    if not py_pass:
        all_pass = False
    results.append({
        "name": "Python",
        "status": "PASS" if py_pass else "FAIL",
        "version": py_ver,
        "install_cmd": "Tải Python 3.10+ từ python.org",
        "description": "Môi trường thực thi Python"
    })

    # 2. Check Python modules
    for tool in REQUIRED_TOOLS[1:]:
        name = tool["name"]
        mod_name = tool["module"]
        try:
            mod = importlib.import_module(mod_name)
            version = getattr(mod, "__version__", "Đã cài đặt")
            results.append({
                "name": name,
                "status": "PASS",
                "version": version,
                "install_cmd": tool["install_cmd"],
                "description": tool["description"]
            })
        except ImportError:
            all_pass = False
            results.append({
                "name": name,
                "status": "FAIL",
                "version": "Chưa cài đặt",
                "install_cmd": tool["install_cmd"],
                "description": tool["description"]
            })

    # 3. Check .env file (Without revealing secrets)
    env_file = Path(__file__).parent / ".env"
    env_status = "FAIL"
    env_detail = "Không tìm thấy file src/.env"
    if env_file.exists():
        try:
            # Simple line parsing to avoid dependency on dotenv if python-dotenv is not yet installed
            content = env_file.read_text(encoding="utf-8")
            has_key = any(line.strip().startswith("LLAMA_CLOUD_API_KEY") for line in content.splitlines())
            if has_key:
                env_status = "PASS"
                env_detail = "Tìm thấy src/.env (Đã cấu hình key)"
            else:
                env_detail = "File src/.env tồn tại nhưng thiếu LLAMA_CLOUD_API_KEY"
        except Exception as e:
            env_detail = f"Lỗi đọc file .env: {e}"

    # Print Table
    print("=" * 80)
    print(" BẢNG KIỂM TRA MÔI TRƯỜNG OCR & CHUNKING (BUỔI 5)")
    print("=" * 80)
    print(f"{'Công cụ / Thư viện':<28} | {'Trạng thái':<10} | {'Phiên bản / Ghi chú':<34}")
    print("-" * 80)

    for item in results:
        status_str = "[PASS]" if item["status"] == "PASS" else "[FAIL]"
        print(f"{item['name']:<28} | {status_str:<10} | {item['version']:<34}")

    status_env = "[PASS]" if env_status == "PASS" else "[FAIL]"
    print(f"{'File src/.env':<28} | {status_env:<10} | {env_detail:<34}")
    print("-" * 80)

    # Print Fix instructions if any FAIL
    fails = [r for r in results if r["status"] == "FAIL"]
    if fails or env_status == "FAIL":
        print("\n HƯỚNG DẪN KHẮC PHỤC CÁC TRẠNG THÁI [FAIL]:")
        for f in fails:
            print(f"  - {f['name']}: Chạy lệnh: `{f['install_cmd']}`")
        if env_status == "FAIL":
            print("  - File .env: Tạo file `RAG/rag_foundation/buoi_05/src/.env` với nội dung:\n    LLAMA_CLOUD_API_KEY='KEY_CỦA_BẠN'")
        print("\nLưu ý: Đảm bảo kích hoạt virtual environment trước khi cài đặt:")
        print("  Windows PowerShell: .venv\\Scripts\\Activate.ps1")
        print("  Hoặc cài trực tiếp: RAG\\rag_foundation\\buoi_05\\.venv\\Scripts\\python.exe -m pip install <tên-gói>")
    else:
        print("\n TẤT CẢ CÁC KIỂM TRA ĐỀU ĐẠT! Môi trường đã sẵn sàng cho Buổi 5.")

    print("=" * 80)
    return 0 if (all_pass and env_status == "PASS") else 1

if __name__ == "__main__":
    sys.exit(check_env())
