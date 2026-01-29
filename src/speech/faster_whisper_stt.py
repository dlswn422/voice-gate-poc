import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from typing import Optional, Callable


class FasterWhisperSTT:
    """
    VAD 기반 실시간 STT 엔진

    특징:
    - 무음 구간에서는 아무 로그도 출력하지 않음
    - 음성 시작 / 발화 종료 / STT 결과만 로그로 출력
    - 발화 종료 시점에만 Whisper 추론 수행
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
        # 오디오 설정
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.silence_threshold = silence_threshold
        self.silence_chunks = silence_chunks
        self.device_index = device_index

        # Whisper 모델 로딩
        print("[STT] Loading Faster-Whisper model...")
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            download_root="models",
        )
        print("[STT] Faster-Whisper model loaded")

        # STT 결과 콜백
        self.on_text: Optional[Callable[[str], None]] = None

    def start_listening(self):
        """
        마이크 입력을 받아 VAD 기반으로 발화를 감지하고
        발화 종료 시 STT를 수행한다.
        """
        print("[STT] Listening started (Ctrl+C to stop)")

        buffer = []
        silent_count = 0
        is_speaking = False

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                device=self.device_index,
                channels=1,
                dtype="float32",
            ):
                while True:
                    data, _ = sd.rec(
                        int(self.chunk_seconds * self.sample_rate),
                        samplerate=self.sample_rate,
                        channels=1,
                        dtype="float32",
                        blocking=True,
                    )
                    audio = data.squeeze()
                    volume = np.max(np.abs(audio))

                    # 음성 시작 감지
                    if volume >= self.silence_threshold:
                        if not is_speaking:
                            print("[STT] Speech detected")
                            is_speaking = True

                        buffer.append(audio)
                        silent_count = 0
                    else:
                        if is_speaking:
                            silent_count += 1

                    # 발화 종료 판단
                    if is_speaking and silent_count >= self.silence_chunks:
                        print("[STT] Speech ended, running transcription")
                        self._process_buffer(buffer)
                        buffer.clear()
                        silent_count = 0
                        is_speaking = False

        except KeyboardInterrupt:
            print("[STT] Listening stopped")

    def _process_buffer(self, buffer):
        """
        누적된 오디오 버퍼를 Whisper로 변환한다.
        """
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
            print("[STT] No transcription result")
            return

        print(f"[STT] Transcribed text: {text}")

        # STT 결과를 상위 로직(AppEngine)으로 전달
        if self.on_text:
            self.on_text(text)