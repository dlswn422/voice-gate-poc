# 마이크 → STT → 전처리 → GPT → TTS

import sys
from dotenv import load_dotenv

load_dotenv()

from app.config import JSON_KEY, LANG, RATE, CHUNK, OPENAI_MODEL, SYSTEM_PROMPT
from app.audio.mic import MicrophoneStream
from app.stt.google_stt import run_google_stt_streaming
from app.preprocess.text_preprocess import preprocess_user_text, should_commit_final
from app.llm.openai_llm import OpenAIStreamer
from app.tts.eleven_tts import ElevenTTS


def main():
    print("🎤 마이크 실시간 STT 시작! (종료: Ctrl+C)")
    print("말해봐. 예: '결제가 안 돼요'\n")

    llm = OpenAIStreamer(model=OPENAI_MODEL, system_prompt=SYSTEM_PROMPT)
    tts = ElevenTTS()

    try:
        with MicrophoneStream(RATE, CHUNK) as mic:
            events = run_google_stt_streaming(
                json_key_path=JSON_KEY,
                rate=RATE,
                lang=LANG,
                audio_generator=mic.generator(),
            )

            for kind, text in events:
                if kind == "partial":
                    print(f"… {text}", end="\r")
                    continue

                print(f"\n✅ FINAL(raw): {text}")

                cleaned = preprocess_user_text(text)
                print(f"🧹 CLEANED: {cleaned!r}")

                if not cleaned or not should_commit_final(cleaned):
                    print("(너무 짧거나 의미 없는 발화라 GPT 호출 생략)\n")
                    continue

                # ✅ GPT 호출
                gpt_text = llm.ask_streaming_and_collect(cleaned)

                # 🔊 TTS 실행 (반이중: TTS 중에는 마이크 입력을 무음으로 전송)
                if gpt_text.strip():
                    mic.muted = True
                    tts.speak(gpt_text)
                    mic.muted = False

                print("\n(다음 발화를 말해줘)\n")

    except KeyboardInterrupt:
        print("\n종료합니다.")
        sys.exit(0)


if __name__ == "__main__":
    main()