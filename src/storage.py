# src/storage.py
import os
from dotenv import load_dotenv
from supabase import create_client

# ==================================================
# 환경변수 로드 (src/.env 기준)
# ==================================================
load_dotenv(dotenv_path="src/.env")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is not set")

# ==================================================
# Supabase Client (Singleton)
# ==================================================
_supabase = None

def get_supabase():
    global _supabase
    if _supabase is None:
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase

# ==================================================
# Storage 설정
# ==================================================
BUCKET_NAME = "parking"

def upload_image(contents: bytes, path: str) -> str:
    supabase = get_supabase()

    res = supabase.storage.from_(BUCKET_NAME).upload(
        path,
        contents,
        {
            "content-type": "image/jpeg",
        }
    )

    if isinstance(res, dict) and res.get("error"):
        raise RuntimeError(f"Supabase upload failed: {res['error']}")

    return supabase.storage.from_(BUCKET_NAME).get_public_url(path)