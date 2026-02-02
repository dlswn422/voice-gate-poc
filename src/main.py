from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv


# ==================================================
# 🔧 환경 변수 / 스레드 제한 (중요)
# - 다른 노트북에서 detect 멈춤 방지 핵심
# ==================================================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ==================================================
# 환경 변수 로드
# - src/.env, project root/.env 모두 시도
# ==================================================
def _load_env():
    here = Path(__file__).resolve().parent          # .../src
    root = here.parent                              # .../voice-gate-poc

    load_dotenv(here / ".env")
    load_dotenv(root / ".env")


_load_env()

print("=" * 60)
print("[ENV] Python:", sys.version)
print("[ENV] CWD:", os.getcwd())
print("[ENV] DATABASE_URL loaded =", bool(os.getenv("DATABASE_URL")))
print("[ENV] OMP_NUM_THREADS =", os.getenv("OMP_NUM_THREADS"))
print("[ENV] MKL_NUM_THREADS =", os.getenv("MKL_NUM_THREADS"))
print("=" * 60)


# ==================================================
# Import (env 설정 이후!)
# ==================================================
from speech.faster_whisper_stt import FasterWhisperSTT  # noqa: E402
from engine.app_engine import AppEngine                  # noqa: E402
from nlu.llm_client import detect_intent_llm             # noqa: E402


# --------------------------------------------------
# 🎤 마이크 디바이스 인덱스
# --------------------------------------------------
MIC_DEVICE_INDEX = 1


def main():
    """
    ParkAssist 메인 진입점 (관측 강화 최종본)
    """

    print("\n[MAIN] 🚀 Starting ParkAssist voice pipeline")

    # ==================================================
    # 1️⃣ App Engine
    # ==================================================
    print("[MAIN] Initializing AppEngine...")
    engine = AppEngine()
    print("[MAIN] AppEngine initialized")

    # ==================================================
    # 2️⃣ STT Engine
    # ==================================================
    print("[MAIN] Initializing STT engine...")
    stt = FasterWhisperSTT(
        model_size="large-v3",   # ⚠️ 다른 노트북 느리면 medium 권장
        device_index=MIC_DEVICE_INDEX,
    )
    print("[MAIN] STT engine initialized")

    # ==================================================
    # 3️⃣ detect LLM 사전 warm-up (⭐ 중요 ⭐)
    # - 다른 노트북에서 "detect에서 멈춤" 방지
    # ==================================================
    print("[MAIN] Warming up intent LLM...")
    t0 = time.time()
    try:
        detect_intent_llm("테스트입니다", debug=True)
    except Exception as e:
        print("[MAIN] ❌ detect warm-up failed:", repr(e))
    print(f"[MAIN] detect warm-up done ({time.time() - t0:.2f}s)")

    # ==================================================
    # 4️⃣ STT → AppEngine 콜백 연결
    # ==================================================
    stt.on_text = engine.handle_text
    print("[MAIN] STT callback connected to AppEngine")

    # ==================================================
    # 5️⃣ Listening
    # ==================================================
    print("[MAIN] 🎧 Listening for microphone input...")
    print("[MAIN] (Ctrl+C to stop)\n")

    try:
        stt.start_listening()
    except KeyboardInterrupt:
        print("\n[MAIN] KeyboardInterrupt received")
    except Exception as e:
        print("[MAIN] ❌ Fatal error:", repr(e))
    finally:
        print("[MAIN] Shutting down...")
        stt.stop()


if __name__ == "__main__":
    main()
