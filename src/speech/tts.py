from gtts import gTTS
from pathlib import Path
import uuid

# 🔥 main.py 기준으로 프로젝트 루트 계산
BASE_DIR = Path(__file__).resolve().parents[2]
# parents[2] 설명:
# tts.py → speech → src → project_root

STATIC_DIR = BASE_DIR / "static"
TTS_DIR = STATIC_DIR / "tts"
TTS_DIR.mkdir(parents=True, exist_ok=True)

def synthesize(text: str) -> str:
    filename = f"{uuid.uuid4().hex}.mp3"
    filepath = TTS_DIR / filename

    tts = gTTS(text=text, lang="ko")
    tts.save(filepath)

    print(f"[TTS FILE SAVED] {filepath}")  # 🔍 디버그

    return f"/static/tts/{filename}"