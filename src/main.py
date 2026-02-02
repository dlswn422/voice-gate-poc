from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv


# ==================================================
# 🔧 스레드 / 병렬 처리 제한
# - 일부 환경에서 detect 멈춤 현상 방지
# ==================================================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ==================================================
# 🌱 환경 변수 로드
# - src/.env → 프로젝트 루트 .env 순서로 시도
# ==================================================
def _load_env():
    here = Path(__file__).resolve().parent
    root = here.parent

    load_dotenv(here / ".env")
    load_dotenv(root / ".env")


_load_env()


# ==================================================
# Import (⚠️ env 설정 이후에 import!)
# ==================================================
from src.speech.faster_whisper_stt import FasterWhisperSTT  # noqa: E402
from src.engine.app_engine import AppEngine                  # noqa: E402
from src.nlu.llm_client import detect_intent_llm             # noqa: E402


# ==================================================
# 🎤 마이크 디바이스 인덱스
# ==================================================
MIC_DEVICE_INDEX = 1


def main():
    """
    ParkAssist 음성 파이프라인 메인 엔트리 포인트
    - STT → Intent Detect → AppEngine 처리
    """

    print("[ParkAssist] 🚀 Starting voice pipeline")

    # ==================================================
    # 1️⃣ App Engine 초기화
    # ==================================================
    engine = AppEngine()

    # ==================================================
    # 2️⃣ STT 엔진 초기화
    # ==================================================
    stt = FasterWhisperSTT(
        model_size="large-v3",   # 성능 이슈 시 medium 권장
        device_index=MIC_DEVICE_INDEX,
    )

    # ==================================================
    # 3️⃣ Intent LLM warm-up
    # - 첫 호출 지연 / 멈춤 현상 방지 목적
    # ==================================================
    try:
        detect_intent_llm("테스트입니다", debug=False)
    except Exception:
        # warm-up 실패해도 서비스는 계속 진행
        pass

    # ==================================================
    # 4️⃣ STT → AppEngine 콜백 연결
    # ==================================================
    stt.on_text = engine.handle_text

    # ==================================================
    # 5️⃣ 마이크 입력 대기
    # ==================================================
    print("[ParkAssist] 🎧 Listening... (Ctrl+C to stop)")

    try:
        stt.start_listening()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("[ParkAssist] ❌ Fatal error:", repr(e))
    finally:
        stt.stop()
        print("[ParkAssist] 👋 Shutdown complete")


if __name__ == "__main__":
    main()
