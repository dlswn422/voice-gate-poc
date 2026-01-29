import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from typing import Optional, Callable

from scipy.signal import resample_poly


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
        model_size: str = "small",
        device_index: Optional[int] = None,
        # Whisper 입력용(최종) 샘플레이트
        sample_rate: int = 16000,
        # 실제 마이크 스트림 샘플레이트 (WASAPI는 보통 48000이 안전)
        input_sample_rate: int = 48000,
        chunk_seconds: float = 0.5,
        silence_threshold: float = 0.015,
        silence_chunks: int = 2,
    ):
        self.sample_rate = sample_rate
        self.input_sample_rate = input_sample_rate
        self.chunk_seconds = chunk_seconds
        self.silence_threshold = silence_threshold
        self.silence_chunks = silence_chunks
        self.device_index = device_index

        print("[STT] Loading Faster-Whisper model...")
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="float32",
            download_root="models",
        )
        print("[STT] Faster-Whisper model loaded")

        self.on_text: Optional[Callable[[str], None]] = None

    def _resample_to_16k(self, audio_1d: np.ndarray) -> np.ndarray:
        """
        input_sample_rate -> sample_rate 로 리샘플링
        (예: 48000 -> 16000)
        """
        if self.input_sample_rate == self.sample_rate:
            return audio_1d.astype(np.float32)

        # 48000 -> 16000 은 3배 다운샘플: up=1, down=3
        if self.input_sample_rate == 48000 and self.sample_rate == 16000:
            return resample_poly(audio_1d, up=1, down=3).astype(np.float32)

        # 일반 케이스: 비율 계산
        up = self.sample_rate
        down = self.input_sample_rate
        return resample_poly(audio_1d, up=up, down=down).astype(np.float32)

    def start_listening(self):
        print("[STT] Listening started (Ctrl+C to stop)")

        buffer = []
        silent_count = 0
        is_speaking = False

        frames_per_chunk = int(self.chunk_seconds * self.input_sample_rate)

        try:
            with sd.InputStream(
                samplerate=self.input_sample_rate,  # 마이크는 48k로 오픈
                device=self.device_index,
                channels=1,
                dtype="float32",
                blocksize=frames_per_chunk,
            ) as stream:
                while True:
                    data, overflowed = stream.read(frames_per_chunk)
                    if overflowed:
                        # 오버플로우가 잦으면 chunk_seconds를 1.0으로 늘려보면 안정됨
                        pass

                    audio = np.asarray(data, dtype=np.float32).squeeze()
                    volume = float(np.max(np.abs(audio))) if audio.size else 0.0

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
        if not buffer:
            return

        audio_in = np.concatenate(buffer)
        audio_16k = self._resample_to_16k(audio_in)

        # ✅ Whisper 내부 VAD 끔 (onnxruntime 불필요)
        segments, _info = self.model.transcribe(
            audio_16k,
            language="ko",
            beam_size=1,        # ✅ CPU 속도 우선(원하면 5~8로 올려도 됨)
            vad_filter=False,   # ✅ 핵심 수정
        )

        text = "".join(seg.text for seg in segments).strip()

        if not text:
            print("[STT] No transcription result")
            return

        print(f"[STT] Transcribed text: {text}")

        if self.on_text:
            self.on_text(text)
