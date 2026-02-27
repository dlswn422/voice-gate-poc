import os
import sys
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

def main():
    if len(sys.argv) < 2:
        print("사용법: python stt_wav.py path/to/audio.wav")
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

    print(f"🎧 테스트 파일: {audio_path}")
    result = recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("[FINAL]", result.text)
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("[NO MATCH] 인식 실패")
    elif result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        print("[CANCELED]", details.reason)
        print("details:", details.error_details)

if __name__ == "__main__":
    main()
