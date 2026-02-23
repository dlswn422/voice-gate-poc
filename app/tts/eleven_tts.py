import os
from elevenlabs.client import ElevenLabs
from elevenlabs import stream as eleven_stream

VOICE_ID = os.getenv("ELEVEN_VOICE_ID")
MODEL_ID = "eleven_multilingual_v2"
print("🎙 ELEVEN_VOICE_ID =", VOICE_ID)
class ElevenTTS:
    def __init__(self):
        api_key = os.getenv("XI_API_KEY") or os.getenv("ELEVEN_API_KEY")
        if not api_key:
            raise RuntimeError("XI_API_KEY 또는 ELEVEN_API_KEY 환경변수가 없습니다.")
        if not VOICE_ID:
            raise RuntimeError("ELEVEN_VOICE_ID 환경변수가 없습니다.")

        # ✅ 핵심: 키를 생성자에 명시적으로 전달
        self.client = ElevenLabs(api_key=api_key)

    def speak(self, text: str):
        audio_stream = self.client.text_to_speech.stream(
            voice_id=VOICE_ID,
            model_id=MODEL_ID,
            text=text,
        )
        eleven_stream(audio_stream)