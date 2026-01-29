import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from typing import Optional, Callable


class FasterWhisperSTT:
    """
    VAD 기반 발화 종료 STT (최종 안정본)
    - 무음 구간에서는 로그 출력 없음
    - 음성 시작 / 발화 종료 / STT 결과만 출력
    - 발화 종료 시에만 STT 수행
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device_index: Optional[int] = None,
        sample_rate: int = 16000,
        chunk_seconds: float = 0.5,
        silence_threshold: float = 0.015,
        silence_chunks: int = 2,
    ):
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.silence_threshold = silence_threshold
        self.silence_chunks = silence_chunks
        self.device_index = device_index

        print("⏳ Faster-Whisper 모델 로딩 중...")
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            download_root="models",
        )
        print("✅ Faster-Whisper 모델 로딩 완료")

        self.on_text: Optional[Callable[[str], None]] = None

    def start_listening(self):
        print("🎙 STT 시작 (VAD 기반, Ctrl+C 종료)")

        buffer = []
        silent_count = 0
        is_speaking = False

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                device=self.device_index,
                channels=1,
                dtype="float32",
            ) as stream:

                while True:
                    data, _ = stream.read(
                        int(self.chunk_seconds * self.sample_rate)
                    )
                    audio = data.squeeze()
                    volume = np.max(np.abs(audio))

                    # 음성 감지
                    if volume >= self.silence_threshold:
                        if not is_speaking:
                            print("🗣 음성 감지 시작")
                            is_speaking = True

                        buffer.append(audio)
                        silent_count = 0
                    else:
                        if is_speaking:
                            silent_count += 1

                    # 발화 종료 판단
                    if is_speaking and silent_count >= self.silence_chunks:
                        print("🧾 발화 종료 감지 → STT 수행")
                        self._process_buffer(buffer)
                        buffer.clear()
                        silent_count = 0
                        is_speaking = False

        except KeyboardInterrupt:
            print("\n🛑 STT 종료")

    def _process_buffer(self, buffer):
        if not buffer:
            return

        audio = np.concatenate(buffer)

        segments, _ = self.model.transcribe(
            audio,
            language="ko",
            beam_size=8,
            vad_filter=True,
        )

        text = "".join(seg.text for seg in segments).strip()

        if not text:
            print("⚠️ STT 결과 없음")
            return

        print(f"🗣 [STT] {text}")

        if self.on_text:
            self.on_text(text)