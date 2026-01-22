import time
import sounddevice as sd
import numpy as np
import whisper
import scipy.signal
from typing import Optional, Callable

_WHISPER_MODEL = None


def get_whisper_model(model_size: str):
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        print("⏳ Whisper 모델 최초 1회 로딩 중...")
        _WHISPER_MODEL = whisper.load_model(model_size)
        print("✅ Whisper 모델 로딩 완료")
    return _WHISPER_MODEL


def normalize_text(text: str) -> str:
    noises = ["ㅋㅋ", "ㅎㅎ", "음", "어", "아", "그", ",", ".", "!", "?"]
    for n in noises:
        text = text.replace(n, "")
    return text.strip()


class WhisperSTT:
    """
    STT 전용 클래스
    - 텍스트까지만 생성
    - 의미 판단 ❌
    """

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[int] = None,
        listen_seconds: float = 1.0,
    ):
        self.device = device
        self.listen_seconds = listen_seconds
        self.input_rate = 48000
        self.target_rate = 16000
        self.on_text: Optional[Callable[[str], None]] = None
        self.model = get_whisper_model(model_size)

    def listen_once(self):
        frames = []

        def callback(indata, frames_count, time_info, status):
            frames.append(indata.copy())

        with sd.InputStream(
            samplerate=self.input_rate,
            device=self.device,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            time.sleep(self.listen_seconds)

        if not frames:
            return

        audio = np.concatenate(frames, axis=0).squeeze()

        if np.max(np.abs(audio)) < 0.02:
            return

        audio = scipy.signal.resample_poly(
            audio,
            self.target_rate,
            self.input_rate,
        )

        audio = audio.astype(np.float32)
        audio /= max(np.abs(audio).max(), 1e-6)

        result = self.model.transcribe(
            audio,
            language="ko",
            fp16=False,
            verbose=False,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
        )

        text = normalize_text(result.get("text", ""))

        if len(text) <= 2:
            return

        print(f"🧪 STT TEXT: {text}")

        if self.on_text:
            self.on_text(text)

    def start_listening(self):
        print("🎙 Whisper STT 시작 (Ctrl+C 종료)")
        try:
            while True:
                self.listen_once()
                time.sleep(0.25)
        except KeyboardInterrupt:
            print("\n🛑 Whisper STT 종료")
