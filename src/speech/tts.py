from gtts import gTTS
from pathlib import Path
import uuid

TTS_DIR = Path("static/tts")
TTS_DIR.mkdir(parents=True, exist_ok=True)


def synthesize(text: str) -> str:
    filename = f"{uuid.uuid4().hex}.mp3"
    filepath = TTS_DIR / filename

    tts = gTTS(text=text, lang="ko")
    tts.save(filepath)

    # 프론트에서 접근 가능한 URL
    return f"/static/tts/{filename}"
