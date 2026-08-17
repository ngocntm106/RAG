import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
ENV_PATH = Path(__file__).parent / ".env"
DATADEMO_DIR = BASE_DIR / "datademo"
OUTPUT_DIR = BASE_DIR / "output"

def load_api_key() -> str:
    """
    Nạp LLAMA_CLOUD_API_KEY từ file .env thuộc folder src một cách an toàn.
    Tuyệt đối không log hoặc in giá trị key ra console.
    """
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    api_key = os.getenv("LLAMA_CLOUD_API_KEY", "")
    return api_key
