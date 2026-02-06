import src.app_state as app_state

from fastapi import APIRouter, UploadFile, File
from pathlib import Path
import uuid
from src.speech.whisper_service import transcribe_audio

router = APIRouter()

@router.post("/voice")
async def voice(audio: UploadFile = File(...)):
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)

    audio_path = tmp_dir / f"{uuid.uuid4()}.webm"
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    # STT
    user_text = transcribe_audio(str(audio_path))

    # AppEngine
    bot_text = app_state.app_engine.handle_text(user_text)

    return {
        "user_text": user_text,
        "bot_text": bot_text,
        "tts_url": None
    }
