from dotenv import load_dotenv

# ==================================================
# 환경 변수 로드
# ==================================================
load_dotenv()

from speech.faster_whisper_stt import FasterWhisperSTT
from engine.app_engine import AppEngine

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

    # 1️⃣ App Engine 생성
    engine = AppEngine()
    print("[MAIN] AppEngine initialized")

    # 2️⃣ STT 엔진 생성
    stt = FasterWhisperSTT(
        model_size="large-v3",
        device_index=MIC_DEVICE_INDEX,
    )
    print("[MAIN] STT engine initialized")

    # 3️⃣ STT 결과 콜백 연결
    stt.on_text = engine.handle_text
    print("[MAIN] STT callback connected to AppEngine")

    # 4️⃣ 음성 입력 대기 시작 (blocking)
    print("[MAIN] Listening for microphone input...")
    stt.start_listening()


if __name__ == "__main__":
    main()