import os
import json
import sqlite3
from pathlib import Path
from dotenv import load_dotenv

import chromadb
import psycopg
from google import genai
from google.genai import types

# Load env variables
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path, override=True)

def _get_gemini_client():
    """Lấy Gemini client với API key mới nhất từ môi trường."""
    load_dotenv(env_path, override=True)
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key or key.startswith("your_"):
        return None
    try:
        return genai.Client(api_key=key)
    except Exception:
        return None

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR.parent / "buoi_05" / "output"
CHROMA_DIR = BASE_DIR / "storage" / "chroma"
SQLITE_DB_PATH = BASE_DIR / "storage" / "local_text.db"

def _get_db_connection():
    """Lấy connection tới DB lưu Text (PostgreSQL fallback SQLite .db)."""
    pg_host = os.environ.get("POSTGRES_HOST", "localhost")
    pg_port = os.environ.get("POSTGRES_PORT", "5432")
    pg_user = os.environ.get("POSTGRES_USER", "postgres")
    pg_password = os.environ.get("POSTGRES_PASSWORD", "")
    pg_db = os.environ.get("POSTGRES_DB", "rag_db")
    
    conn = None
    db_type = "postgres"
    try:
        if pg_password:
            conn = psycopg.connect(f"host={pg_host} port={pg_port} user={pg_user} password={pg_password} dbname={pg_db}", connect_timeout=2)
        else:
            raise ValueError("No password")
    except Exception:
        db_type = "sqlite"
        SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(SQLITE_DB_PATH))
    
    # Đảm bảo bảng tồn tại
    if db_type == "postgres":
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    text TEXT,
                    source TEXT
                )
            """)
        conn.commit()
    else:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                text TEXT,
                source TEXT
            )
        """)
        conn.commit()
        
    return conn, db_type

def _get_chroma_collection():
    """Khởi tạo collection Vector với ChromaDB."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name="rag_buoi_06_docs",
        metadata={"hnsw:space": "cosine"}
    )

def _embed_text(text: str) -> list[float]:
    """Tạo embedding với Gemini model gemini-embedding-2 (384 chiều)."""
    client = _get_gemini_client()
    if not client or not text.strip():
        return [0.0] * 384
        
    try:
        response = client.models.embed_content(
            model='gemini-embedding-2',
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=384)
        )
        return response.embeddings[0].values
    except Exception:
        return [0.0] * 384

def index():
    """Đọc JSON từ output/chunks, sinh embedding 384d, lưu text vào SQL và vector vào ChromaDB."""
    collection = _get_chroma_collection()
    conn, db_type = _get_db_connection()
    
    # Đọc từ output/chunks/ hoặc fallback về output/
    chunks_dir = OUTPUT_DIR / "chunks"
    if chunks_dir.exists():
        chunks_files = list(chunks_dir.glob("*.json"))
    else:
        chunks_files = list(OUTPUT_DIR.glob("*chunks*.json"))
    
    count = 0
    for file_path in chunks_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            for item in data:
                chunk_id = item.get("chunk_id", "")
                text_content = item.get("text", "")
                source = item.get("source", "")
                
                if not text_content.strip() or not chunk_id:
                    continue
                    
                # 1. Lưu text vào PostgreSQL hoặc SQLite (.db)
                if db_type == "postgres":
                    with conn.cursor() as cur:
                        cur.execute("INSERT INTO documents (id, text, source) VALUES (%s, %s, %s) ON CONFLICT (id) DO NOTHING", (chunk_id, text_content, source))
                else:
                    cursor = conn.cursor()
                    cursor.execute("INSERT OR IGNORE INTO documents (id, text, source) VALUES (?, ?, ?)", (chunk_id, text_content, source))
                
                # 2. Lưu embedding vào ChromaDB
                embedding = _embed_text(text_content)
                collection.upsert(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    metadatas=[{"source": source}]
                )
                
                count += 1
        except Exception:
            pass
            
    conn.commit()
    conn.close()
    return count

def ask(question: str, k: int = 3):
    """Embedding câu hỏi 384d, tìm top-k, lấy text tương ứng từ DB, gửi Gemini trả lời."""
    collection = _get_chroma_collection()
    conn, db_type = _get_db_connection()
    
    total_vectors = collection.count()
    if total_vectors == 0:
        conn.close()
        return {
            "answer": "⚠️ Cơ sở dữ liệu trống. Vui lòng bấm 'Index dữ liệu' trước khi thực hiện truy vấn.",
            "contexts": []
        }
        
    effective_k = min(k, total_vectors)
    query_embedding = _embed_text(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=effective_k
    )
    
    retrieved_texts = []
    chunk_ids = results["ids"][0] if results["ids"] else []
    
    for cid in chunk_ids:
        if db_type == "postgres":
            with conn.cursor() as cur:
                cur.execute("SELECT text, source FROM documents WHERE id = %s", (cid,))
                row = cur.fetchone()
        else:
            cursor = conn.cursor()
            cursor.execute("SELECT text, source FROM documents WHERE id = ?", (cid,))
            row = cursor.fetchone()
            
        if row:
            retrieved_texts.append(f"[Nguồn: {row[1]}] {row[0]}")
            
    conn.close()
    
    context_str = "\n\n".join(retrieved_texts)
    
    # Ràng buộc: Nếu thiếu GEMINI_API_KEY hoặc key không hợp lệ: Vẫn cho phép retrieval, không gọi LLM.
    gemini_client = _get_gemini_client()
    if not gemini_client:
        return {
            "answer": "⚠️ Thiếu GEMINI_API_KEY. Hệ thống chỉ thực hiện Retrieval (truy xuất dữ liệu), không gọi LLM.",
            "contexts": retrieved_texts
        }
        
    if not context_str:
        return {"answer": "Không tìm thấy dữ liệu liên quan trong cơ sở dữ liệu.", "contexts": []}
        
    prompt = f"Dựa vào các đoạn văn bản sau để trả lời câu hỏi:\n\n{context_str}\n\nCâu hỏi: {question}"
    
    # Gọi Gemini (Ưu tiên model tương thích API KEY)
    for model_name in ['gemini-3.6-flash', 'gemini-3.5-flash', 'gemini-flash-latest']:
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            answer = response.text
            break
        except Exception as e_item:
            answer = f"Lỗi gọi LLM: {e_item}"
            
    return {"answer": answer, "contexts": retrieved_texts}

def status():
    """Thống kê số lượng document và số lượng chunk."""
    collection = _get_chroma_collection()
    conn, db_type = _get_db_connection()
    
    try:
        chroma_count = collection.count()
    except Exception:
        chroma_count = 0
        
    try:
        if db_type == "postgres":
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*), COUNT(DISTINCT source) FROM documents")
                res = cur.fetchone()
                sql_count = res[0]
                doc_count = res[1]
        else:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), COUNT(DISTINCT source) FROM documents")
            res = cursor.fetchone()
            sql_count = res[0]
            doc_count = res[1]
    except Exception:
        sql_count = 0
        doc_count = 0
        
    conn.close()
    
    return {
        "db_type": db_type,
        "documents": doc_count,
        "chunks": sql_count,
        "chroma_vectors": chroma_count
    }
