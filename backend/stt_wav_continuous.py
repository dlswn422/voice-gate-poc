import os
import sys
import time
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

def main():
    if len(sys.argv) < 2:
        print("사용법: python stt_wav_continuous.py path/to/audio.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)

    load_dotenv()
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    lang = os.getenv("AZURE_SPEECH_LANGUAGE", "ko-KR")

    if not key or not region:
        raise RuntimeError("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION 가 .env에 없습니다.")

    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.speech_recognition_language = lang

    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    done = False
    finals = []

    print(f"🎧 연속 인식 테스트: {audio_path}")

    def on_recognized(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech and evt.result.text:
            finals.append(evt.result.text)
            print(f"[FINAL] {evt.result.text}")
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            print("[NO MATCH]")

    def on_canceled(evt):
        nonlocal done
        print("[CANCELED]", evt.reason)
        print("details:", evt.error_details)
        done = True

    def on_session_stopped(evt):
        nonlocal done
        done = True

    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)
    recognizer.session_stopped.connect(on_session_stopped)

    recognizer.start_continuous_recognition()
    while not done:
        time.sleep(0.2)
    recognizer.stop_continuous_recognition()

    print("\n=== 결과 합치기 ===")
    print(" ".join(finals) if finals else "(인식된 문장 없음)")

if __name__ == "__main__":
    main()
