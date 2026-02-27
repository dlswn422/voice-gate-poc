import os
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

def main():
    load_dotenv()

    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    lang = os.getenv("AZURE_SPEECH_LANGUAGE", "ko-KR")

    if not key or not region:
        raise RuntimeError("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION 가 .env에 없습니다.")

    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.speech_recognition_language = lang

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("🎤 마이크 STT 시작. 말해보세요. (Enter 누르면 종료)\n")

    def on_recognizing(evt):
        t = getattr(evt.result, "text", "")
        if t:
            print(f"[PARTIAL] {t}")

    def on_recognized(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech and evt.result.text:
            print(f"[FINAL]   {evt.result.text}")
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            print("[NO MATCH] 인식 실패")

    def on_canceled(evt):
        print("[CANCELED]", evt.reason)
        print("details:", evt.error_details)

    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)

    recognizer.start_continuous_recognition()
    input()
    recognizer.stop_continuous_recognition()
    print("✅ 종료")

if __name__ == "__main__":
    main()
