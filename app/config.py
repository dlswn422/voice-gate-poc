import os

# =========================
# Google STT 설정
# =========================
JSON_KEY = os.getenv("GOOGLE_JSON_KEY", "speech_demo.json")
LANG = os.getenv("STT_LANG", "ko-KR")

RATE = int(os.getenv("AUDIO_RATE", "16000"))
CHUNK = int(RATE / 10)  # 100ms 프레임 (마이크에서 0.1초 단위로 오디오를 읽음)

# =========================
# OpenAI 설정
# =========================
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "너는 주차장 키오스크 음성 안내 도우미다. 짧고 명확하게 한국어로 답해라."
)