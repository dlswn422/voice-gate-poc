import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from typing import Optional, Callable


class FasterWhisperSTT:
    """
    발화 단위 VAD 기반 STT 엔진 (안정 최종본)

    동작:
    - 무음 대기
    - 음성 시작 감지
    - 발화 끝까지 버퍼링
    - Whisper는 발화당 1회 실행
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device_index: Optional[int] = None,
        sample_rate: int = 16000,
        chunk_seconds: float = 0.4,
        silence_threshold: float = 0.02,
        silence_chunks: int = 1,
    ):
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.silence_threshold = silence_threshold
        self.silence_chunks = silence_chunks
        self.device_index = device_index

        print("[STT] Loading Faster-Whisper model...")
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            download_root="models",
        )
        print("[STT] Faster-Whisper model loaded")

        self.on_text: Optional[Callable[[str], None]] = None

    def start_listening(self):
        print("[STT] Listening started (Ctrl+C to stop)")

        # sounddevice 초기화
        sd.stop()
        sd.default.device = self.device_index
        sd.default.samplerate = self.sample_rate
        sd.default.channels = 1

        buffer = []
        silent_count = 0
        is_speaking = False
        frames_per_chunk = int(self.chunk_seconds * self.sample_rate)

        try:
            while True:
                # 오디오 수집
                audio = sd.rec(
                    frames_per_chunk,
                    dtype="float32",
                )
                sd.wait()

                audio = audio.squeeze()
                volume = np.max(np.abs(audio))

                # 🔍 디버그용 (필요 없으면 지워도 됨)
                # print(f"[DEBUG] volume={volume:.4f}")

                # ------------------------------
                # 음성 시작 감지
                # ------------------------------
                if volume >= self.silence_threshold:
                    if not is_speaking:
                        print("[STT] Speech detected")
                        is_speaking = True

                    buffer.append(audio)
                    silent_count = 0

                else:
                    if is_speaking:
                        silent_count += 1

                # ------------------------------
                # 발화 종료 감지
                # ------------------------------
                if is_speaking and silent_count >= self.silence_chunks:
                    print("[STT] Speech ended, running transcription")
                    self._process_buffer(buffer)

                    buffer.clear()
                    silent_count = 0
                    is_speaking = False

        except KeyboardInterrupt:
            sd.stop()
            print("[STT] Listening stopped")

        except Exception as e:
            sd.stop()
            print("[STT ERROR]", repr(e))

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
            print("[STT] No transcription result")
            return

        print(f"[STT] Transcribed text: {text}")

        if self.on_text:
            self.on_text(text)