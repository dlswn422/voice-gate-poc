import time
import sounddevice as sd
import numpy as np
import whisper
import scipy.signal
from typing import Optional, Callable

# =========================
# Whisper 모델 싱글톤
# =========================
_WHISPER_MODEL = None


def get_whisper_model(model_size: str):
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        print("⏳ Whisper 모델 최초 1회 로딩 중...")
        _WHISPER_MODEL = whisper.load_model(model_size)
        print("✅ Whisper 모델 로딩 완료")
    return _WHISPER_MODEL


# =========================
# 텍스트 정규화
# =========================
def normalize_text(text: str) -> str:
    noises = ["ㅋㅋ", "ㅎㅎ", "음", "어", "아", "그", ",", ".", "!", "?"]
    for n in noises:
        text = text.replace(n, "")
    return text.strip()


# =========================
# Intent 판별 (최종)
# =========================
def detect_intent(text: str) -> Optional[str]:
    open_keywords = ["열", "여", "올", "개", "오픈"]
    close_keywords = ["닫", "잠", "내", "클로즈"]

    if any(k in text for k in open_keywords):
        return "OPEN_GATE"

    if any(k in text for k in close_keywords):
        return "CLOSE_GATE"

    return None


class WhisperSTT:
    """
    Whisper STT 최종본 (현업 기준)

    ✔ Windows 마이크 입력 안정
    ✔ 48kHz → 16kHz 리샘플링
    ✔ 의미 없는 발화 제거
    ✔ Intent 중심 처리
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

        self.on_intent: Optional[Callable[[str, str], None]] = None
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

        # 🔕 무음 컷
        if np.max(np.abs(audio)) < 0.02:
            return

        # 🔁 48k → 16k
        audio = scipy.signal.resample_poly(
            audio,
            self.target_rate,
            self.input_rate,
        )

        # 🔊 정규화
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

        raw_text = normalize_text(result.get("text", ""))

        # ❌ 너무 짧은 발화 제거
        if len(raw_text) <= 2:
            return

        # ❌ 동작 단어 없는 발화 제거
        if not any(k in raw_text for k in ["열", "닫", "올", "내"]):
            return

        print(f"🧪 RAW STT TEXT: {raw_text}")

        intent = detect_intent(raw_text)

        if intent:
            print(f"🚦 INTENT DETECTED: {intent}")
            if self.on_intent:
                self.on_intent(intent, raw_text)

    def start_listening(self):
        print("🎙 Whisper STT 시작 (Ctrl+C 종료)")
        try:
            while True:
                self.listen_once()
                time.sleep(0.25)
        except KeyboardInterrupt:
            print("\n🛑 Whisper STT 종료")