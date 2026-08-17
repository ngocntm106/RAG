import os
import sys
import subprocess
from pathlib import Path

def main():
    print("==================================================")
    print("CHƯƠNG TRÌNH SETUP MÔI TRƯỜNG RAG - BUỔI 06")
    print("==================================================")
    
    # 1. Python Interpreter
    interpreter = sys.executable
    print(f"Python interpreter đang sử dụng:\n{interpreter}")
    
    # Paths
    base_dir = Path(__file__).parent
    env_file = base_dir / ".env"
    env_example = base_dir / ".env.example"
    
    # 2. Setup .ENV
    if not env_file.exists():
        if env_example.exists():
            with open(env_example, 'r', encoding='utf-8') as f:
                content = f.read()
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            env_file.touch()
            
    # Read existing env to not overwrite
    existing_keys = set()
    with open(env_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key = line.split('=')[0].strip()
                existing_keys.add(key)
                
    # Add missing variables
    default_vars = {
        "GEMINI_API_KEY": "",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "rag_db",
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": ""
    }
    
    with open(env_file, 'a', encoding='utf-8') as f:
        for k, v in default_vars.items():
            if k not in existing_keys:
                f.write(f"\n{k}={v}")

    # Ensure dotenv is available before using it
    try:
        import dotenv
    except ImportError:
        subprocess.check_call([interpreter, "-m", "pip", "install", "python-dotenv", "-q"])
        import dotenv
        
    dotenv.load_dotenv(env_file)
    
    # 3. Packages
    packages = ["streamlit", "google-genai", "chromadb", "psycopg[binary]", "python-dotenv"]
    print("\n--- Cài đặt Packages ---")
    try:
        subprocess.check_call([interpreter, "-m", "pip", "install"] + packages + ["-q"])
        print("Đã cài đặt xong các packages.")
    except Exception as e:
        print(f"Lỗi khi cài đặt packages: {e}")
        
    print("\n--- Kết quả Import ---")
    for pkg_name, module_name in [("streamlit", "streamlit"), ("google-genai", "google.genai"), ("chromadb", "chromadb"), ("psycopg", "psycopg"), ("python-dotenv", "dotenv")]:
        try:
            if module_name == "google.genai":
                import google.genai
            else:
                __import__(module_name)
            print(f"[{pkg_name}] -> PASS")
        except ImportError as e:
            print(f"[{pkg_name}] -> FAIL ({e})")

    # 4. ChromaDB
    print("\n--- Trạng thái ChromaDB ---")
    import chromadb
    chroma_status = "Unknown"
    try:
        # Try HTTP Client first
        client = chromadb.HttpClient(host="localhost", port=8000)
        client.heartbeat()
        chroma_status = "Server (http://localhost:8000)"
    except Exception:
        # Use PersistentClient
        chroma_dir = base_dir / "storage" / "chroma"
        chroma_dir.mkdir(parents=True, exist_ok=True)
        chroma_status = f"Embedded Local ({chroma_dir.resolve()})"
    print(chroma_status)
    
    # 5. PostgreSQL
    print("\n--- Trạng thái PostgreSQL ---")
    import psycopg
    from psycopg.errors import OperationalError
    
    pg_host = os.environ.get("POSTGRES_HOST", "localhost")
    pg_port = os.environ.get("POSTGRES_PORT", "5432")
    pg_user = os.environ.get("POSTGRES_USER", "postgres")
    pg_password = os.environ.get("POSTGRES_PASSWORD", "")
    pg_db = os.environ.get("POSTGRES_DB", "rag_db")
    
    pg_status = "Chưa rõ"
    db_status = "Chưa rõ"
    user_actions = []
    
    if not pg_password:
        pg_status = "Thiếu mật khẩu"
        db_status = "Không thể kiểm tra do thiếu mật khẩu"
        user_actions.append("Thêm POSTGRES_PASSWORD vào file .env.")
        try:
            # Check if postgres is accessible without password
            conn = psycopg.connect(f"host={pg_host} port={pg_port} user={pg_user} dbname=postgres", connect_timeout=3)
            conn.close()
            pg_status = "Hoạt động (không yêu cầu mật khẩu)"
            user_actions.remove("Thêm POSTGRES_PASSWORD vào file .env.")
        except Exception:
            user_actions.append("Nếu chưa cài PostgreSQL, tải tại https://www.postgresql.org/download/ và cài đặt, ghi nhớ mật khẩu.")
    else:
        try:
            # Connect to default postgres DB to check/create rag_db
            conn = psycopg.connect(f"host={pg_host} port={pg_port} user={pg_user} password={pg_password} dbname=postgres", autocommit=True, connect_timeout=3)
            pg_status = "Hoạt động"
            
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (pg_db,))
            exists = cur.fetchone()
            
            if not exists:
                cur.execute(f"CREATE DATABASE {pg_db};")
                db_status = f"Vừa được tự động tạo mới"
            else:
                db_status = f"Đã tồn tại"
            conn.close()
            
            # Test connection to rag_db
            conn_db = psycopg.connect(f"host={pg_host} port={pg_port} user={pg_user} password={pg_password} dbname={pg_db}", connect_timeout=3)
            conn_db.close()
            db_status += " (Kết nối thành công)"
            
        except OperationalError as e:
            pg_status = "Không thể kết nối"
            db_status = "Không thể kết nối"
            user_actions.append("PostgreSQL chưa được cài đặt, hoặc chưa khởi động, hoặc sai mật khẩu.")
            user_actions.append("Tải PostgreSQL tại https://www.postgresql.org/download/, cài đặt và điền POSTGRES_PASSWORD vào .env.")

    if not os.environ.get("GEMINI_API_KEY"):
        user_actions.append("Thêm GEMINI_API_KEY vào file .env.")

    print(f"- PostgreSQL Server: {pg_status}")
    print(f"- Database '{pg_db}': {db_status}")
    
    print("\n--- Yêu cầu hành động từ người dùng ---")
    if user_actions:
        for i, action in enumerate(user_actions, 1):
            print(f"{i}. {action}")
    else:
        print("Tất cả môi trường đã sẵn sàng! Không cần thao tác thêm.")

if __name__ == '__main__':
    main()
