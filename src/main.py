from dotenv import load_dotenv
from pathlib import Path

# ==================================================
# .env 명시적 로드
# ==================================================
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from faster_whisper import WhisperModel

import src.app_state as app_state
from src.engine.app_engine import AppEngine
from src.parking.session_engine import ParkingSessionEngine

# ✅ Routers
from src.api.voice import router as voice_router
from src.api.voice_session_ws import router as voice_ws_router   # ⭐ 변경
from src.api.plate import router as plate_router
from src.api.payment import router as payment_router

# ==================================================
# FastAPI App
# ==================================================
app = FastAPI(title="ParkAssist Voice API")

# ==================================================
# CORS
# ==================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================================
# Static (TTS mp3 서빙)
# ==================================================
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)

# ==================================================
# Startup: 모델 / 엔진 메모리 상주
# ==================================================
@app.on_event("startup")
def startup():
    print("[Startup] Loading Whisper model...")

    # 🔥 Whisper (HTTP + WebSocket 공용)
    app_state.whisper_model = WhisperModel(
        "medium",
        device="cpu",
        compute_type="int8_float32",
    )

    print("[Startup] Initializing AppEngine...")
    app_state.app_engine = AppEngine()

    print("[Startup] Initializing ParkingSessionEngine...")
    app_state.session_engine = ParkingSessionEngine()

    print("[Startup] ✅ Service ready")

# ==================================================
# Routers
# ==================================================
app.include_router(voice_router)       # HTTP voice API
app.include_router(voice_ws_router)    # WebSocket voice
app.include_router(plate_router)       # OCR
app.include_router(payment_router)     # Payment
