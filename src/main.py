from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# ==================================================
# 환경 변수 로드
# - 어디서 실행하든(.env 위치가 root/src 둘 다 가능) 안정적으로 로드
# ==================================================
def _load_env():
    here = Path(__file__).resolve().parent          # .../src
    root = here.parent                              # .../voice-gate-poc

    # 1) src/.env
    load_dotenv(here / ".env")
    # 2) project root/.env
    load_dotenv(root / ".env")


_load_env()
print("[ENV] DATABASE_URL loaded =", bool(os.getenv("DATABASE_URL")))

from speech.faster_whisper_stt import FasterWhisperSTT  # noqa: E402
from engine.app_engine import AppEngine                  # noqa: E402


# --------------------------------------------------
# 마이크 디바이스 인덱스
# - sd.query_devices()로 확인한 Input 장치 번호
# --------------------------------------------------
MIC_DEVICE_INDEX = 1


def main():
    """
    ParkAssist 메인 진입점
    """
    print("[MAIN] Starting ParkAssist voice pipeline")

    # 1) App Engine 생성
    engine = AppEngine()
    print("[MAIN] AppEngine initialized")

    # 2) STT 엔진 생성
    stt = FasterWhisperSTT(
        model_size="large-v3",
        device_index=MIC_DEVICE_INDEX,
    )
    print("[MAIN] STT engine initialized")

    # 3) STT 결과 콜백 연결
    stt.on_text = engine.handle_text
    print("[MAIN] STT callback connected to AppEngine")

    # 4) 음성 입력 대기 시작 (blocking)
    print("[MAIN] Listening for microphone input...")
    stt.start_listening()


if __name__ == "__main__":
    main()
