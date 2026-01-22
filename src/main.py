print("✅ main.py 실행됨")

from speech.whisper_stt import WhisperSTT

# 🎤 노트북 내장 마이크 (네가 확인한 인덱스)
MIC_DEVICE_INDEX = 9


def main():
    print("✅ main() 진입")

    stt = WhisperSTT(
        model_size="base",
        device=MIC_DEVICE_INDEX,
        listen_seconds=1.0,
    )

    def on_intent(intent: str, text: str):
        print("📥 TEXT :", text)

        if intent == "OPEN_GATE":
            print("🟢 차단기 열기 명령 실행")
            # TODO: 실제 차단기 제어 (GPIO / API / 릴레이)

        elif intent == "CLOSE_GATE":
            print("🔴 차단기 닫기 명령 실행")

    stt.on_intent = on_intent
    stt.start_listening()


if __name__ == "__main__":
    main()