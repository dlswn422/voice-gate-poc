import os
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

def main():
    load_dotenv()

    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    voice = os.getenv("AZURE_SPEECH_VOICE", "ko-KR-SunHiNeural")

    if not key or not region:
        raise RuntimeError("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION 가 .env에 없습니다.")

    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.speech_synthesis_voice_name = voice

    # ✅ 파일 저장 없이, 기본 스피커로 바로 출력
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    text = "안녕하세요. Azure 텍스트 음성 변환 테스트입니다. 지금 제 목소리가 들리나요?"
    print(f"🔊 TTS 시작 (voice={voice})")

    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("✅ TTS 완료")
    elif result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        print("❌ TTS 취소:", details.reason)
        print("details:", details.error_details)

if __name__ == "__main__":
    main()
