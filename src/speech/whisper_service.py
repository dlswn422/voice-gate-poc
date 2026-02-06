import numpy as np
import soundfile as sf
import tempfile
import os

import src.app_state as app_state


# ==================================================
# 1️⃣ 기존 파일 기반 STT (호환 유지)
# ==================================================
def transcribe_audio(audio_path: str) -> str:
    """
    파일 경로 기반 Whisper STT
    (기존 HTTP /voice API용)
    """
    segments, _ = app_state.whisper_model.transcribe(audio_path)
    return " ".join(seg.text.strip() for seg in segments if seg.text)


# ==================================================
# 2️⃣ PCM 스트리밍 기반 STT (WebSocket용)
# ==================================================
def transcribe_pcm_chunks(
    pcm_chunks: list[np.ndarray],
    whisper_model=None,
    sample_rate: int = 16000,
) -> str:
    """
    PCM(Float32) chunk 리스트를 Whisper로 변환

    Args:
        pcm_chunks: List[np.ndarray] (float32, mono)
        whisper_model: faster-whisper WhisperModel
        sample_rate: 기본 16kHz

    Returns:
        str: 인식된 텍스트
    """

    if not pcm_chunks:
        return ""

    if whisper_model is None:
        whisper_model = app_state.whisper_model

    # ==================================================
    # PCM 병합
    # ==================================================
    audio = np.concatenate(pcm_chunks).astype(np.float32)

    if audio.size == 0:
        return ""

    # ==================================================
    # 임시 WAV 파일 생성
    # (faster-whisper는 파일 입력이 가장 안정적)
    # ==================================================
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sf.write(tmp_path, audio, samplerate=sample_rate)

        segments, _ = whisper_model.transcribe(
            tmp_path,
            language="ko",
            vad_filter=False,
        )

        text = " ".join(seg.text.strip() for seg in segments if seg.text)
        return text.strip()

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
