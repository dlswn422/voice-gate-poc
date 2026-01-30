from pathlib import Path
from dotenv import load_dotenv

# ??main.py ?꾩튂 湲곗??쇰줈 src/.env瑜???긽 濡쒕뱶
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)

import os
print("[ENV] DATABASE_URL loaded =", bool(os.getenv("DATABASE_URL")))

# ==================================================
# ?섍꼍 蹂??濡쒕뱶
# ==================================================
from speech.faster_whisper_stt import FasterWhisperSTT
from engine.app_engine import AppEngine

# --------------------------------------------------
# 留덉씠???붾컮?댁뒪 ?몃뜳??# - sd.query_devices()濡??뺤씤??Input ?μ튂 踰덊샇
# --------------------------------------------------
MIC_DEVICE_INDEX = 1


def main():
    """
    ParkAssist 硫붿씤 吏꾩엯??    """

    print("[MAIN] Starting ParkAssist voice pipeline")

    # 1截뤴깵 App Engine ?앹꽦
    engine = AppEngine()
    print("[MAIN] AppEngine initialized")

    # 2截뤴깵 STT ?붿쭊 ?앹꽦
    stt = FasterWhisperSTT(
        model_size="large-v3",
        device_index=MIC_DEVICE_INDEX,
    )
    print("[MAIN] STT engine initialized")

    # 3截뤴깵 STT 寃곌낵 肄쒕갚 ?곌껐
    stt.on_text = engine.handle_text
    print("[MAIN] STT callback connected to AppEngine")

    # 4截뤴깵 ?뚯꽦 ?낅젰 ?湲??쒖옉 (blocking)
    print("[MAIN] Listening for microphone input...")
    stt.start_listening()


if __name__ == "__main__":
    main()
