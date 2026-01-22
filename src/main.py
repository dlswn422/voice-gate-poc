from speech.faster_whisper_stt import FasterWhisperSTT
from engine.app_engine import AppEngine

MIC_DEVICE_INDEX = 1  # 노트북 내장 마이크


def main():
    engine = AppEngine()

    stt = FasterWhisperSTT(
        model_size="large-v3",
        device_index=MIC_DEVICE_INDEX,
    )

    stt.on_text = engine.handle_text
    stt.start_listening()


if __name__ == "__main__":
    main()