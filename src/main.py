from dotenv import load_dotenv
from pathlib import Path

# ==================================================
# .env 명시적 로드 (중요)
# ==================================================
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

import src.app_state as app_state
from src.engine.app_engine import AppEngine
from src.api.voice import router as voice_router
from src.api.voice_ws import router as voice_ws_router  # ✅ WebSocket 추가


# ==================================================
# FastAPI App
# ==================================================
app = FastAPI(title="ParkAssist Voice API")


# ==================================================
# CORS 설정
# ==================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# Startup: 모델 / 엔진 메모리 상주
# ==================================================
@app.on_event("startup")
def startup():
    print("[Startup] Loading Whisper model...")

    # 🔥 전역 상태에 직접 할당 (HTTP / WS 공용)
    app_state.whisper_model = WhisperModel(
        "large-v3",
        device="cpu",
        compute_type="int8_float32",
    )

    print("[Startup] Initializing AppEngine...")
    app_state.app_engine = AppEngine()

    print("[Startup] ✅ Service ready")


# ==================================================
# Routers
# ==================================================
# 기존 HTTP API
app.include_router(voice_router)

# WebSocket API (상시 마이크)
app.include_router(voice_ws_router)
