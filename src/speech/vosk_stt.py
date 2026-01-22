import json
import time
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from typing import Optional, Callable


class VoskSTT:
    """
    Vosk 기반 음성 인식 클래스 (Windows 최종 안정 버전)

    - 노트북 내장 마이크 (WASAPI) 대응
    - Invalid device / Invalid sample rate 문제 해결
    - input overflow 최소화
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[int] = None,
    ):
        self.device = device

        # 🔑 WASAPI 내장 마이크는 보통 48000Hz만 지원
        if device is not None:
            self.sample_rate = int(sd.query_devices(device)["default_samplerate"])
        else:
            self.sample_rate = 48000

        # STT → 외부 전달 콜백
        self.on_text: Optional[Callable[[str], None]] = None

        # Vosk 초기화 (Vosk는 내부적으로 16k 처리 가능)
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, 16000)

    def start_listening(self):
        print("🎙 말해보세요 (Ctrl+C로 종료)")

        def callback(indata, frames, time_info, status):
            if status:
                print("⚠️ Audio status:", status)

            # CFFI buffer → bytes
            data = bytes(indata)

            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "").strip()

                if text:
                    print(f"🗣 인식 결과: {text}")
                    if self.on_text:
                        self.on_text(text)

        with sd.RawInputStream(
            samplerate=self.sample_rate,  # 🔑 48000Hz
            blocksize=16000,              # 🔑 overflow 방지
            dtype="int16",
            channels=1,
            callback=callback,
            device=self.device,
        ):
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n🛑 음성 인식 종료")