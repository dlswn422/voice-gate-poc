import os
import threading
import queue
from typing import Optional, Callable

import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from scipy.signal import resample_poly


# --------------------------------------------------
# Windows + ctranslate2 안정화
# --------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


class FasterWhisperSTT:
    """
    실시간 STT 엔진 (비동기 안정 + 로그 강화 최종본)

    구조
    - Audio Thread (sounddevice InputStream)
        - 마이크 입력
        - 발화 감지(VAD-lite)
        - 오디오 버퍼를 Queue에 enqueue
    - STT Worker Thread
        - Whisper 추론만 담당 (절대 Audio Thread 안에서 실행 ❌)
    """

    def __init__(
        self,
        model_size: str = "medium",
        device_index: Optional[int] = None,
        sample_rate: int = 16000,
        input_sample_rate: int = 48000,
        chunk_seconds: float = 0.5,
        silence_threshold: float = 0.015,
        silence_chunks: int = 2,
        min_utterance_seconds: float = 0.6,
        min_text_len: int = 2,
        beam_size: int = 1,
        temperature: float = 0.0,
        download_root: str = "models",
    ):
        self.sample_rate = sample_rate
        self.input_sample_rate = input_sample_rate
        self.chunk_seconds = chunk_seconds
        self.silence_threshold = silence_threshold
        self.silence_chunks = silence_chunks
        self.device_index = device_index

        self.min_utterance_seconds = min_utterance_seconds
        self.min_text_len = min_text_len

        self.beam_size = beam_size
        self.temperature = temperature

        self.on_text: Optional[Callable[[str], None]] = None

        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stop_event = threading.Event()

        # --------------------------------------------------
        # Whisper 모델 로드
        # --------------------------------------------------
        print("[STT] Loading Faster-Whisper model...")
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="float32",
            download_root=download_root,
        )
        print("[STT] Faster-Whisper model loaded")

        # --------------------------------------------------
        # STT 워커 스레드 시작
        # --------------------------------------------------
        self._worker_thread = threading.Thread(
            target=self._stt_worker,
            daemon=True,
        )
        self._worker_thread.start()

        # --------------------------------------------------
        # Whisper warm-up
        # --------------------------------------------------
        self._warmup()

    # ==================================================
    # Warm-up
    # ==================================================
    def _warmup(self):
        print("[STT] Warming up whisper...")
        dummy = np.zeros(int(self.sample_rate * 1.0), dtype=np.float32)
        try:
            list(
                self.model.transcribe(
                    dummy,
                    language="ko",
                    beam_size=1,
                    temperature=0.0,
                    vad_filter=False,
                )
            )
        except Exception as e:
            print(f"[STT] Warm-up error (ignored): {repr(e)}")
        print("[STT] Warm-up done")

    # ==================================================
    # Resample
    # ==================================================
    def _resample_to_16k(self, audio_1d: np.ndarray) -> np.ndarray:
        if self.input_sample_rate == self.sample_rate:
            return audio_1d.astype(np.float32)

        if self.input_sample_rate == 48000 and self.sample_rate == 16000:
            return resample_poly(audio_1d, up=1, down=3).astype(np.float32)

        return resample_poly(
            audio_1d,
            up=self.sample_rate,
            down=self.input_sample_rate,
        ).astype(np.float32)

    # ==================================================
    # Audio Thread
    # ==================================================
    def start_listening(self):
        print("[STT] Listening started (Ctrl+C to stop)")

        buffer = []
        silent_count = 0
        is_speaking = False

        frames_per_chunk = int(self.chunk_seconds * self.input_sample_rate)

        try:
            with sd.InputStream(
                samplerate=self.input_sample_rate,
                device=self.device_index,
                channels=1,
                dtype="float32",
                blocksize=frames_per_chunk,
            ) as stream:
                while not self._stop_event.is_set():
                    data, overflowed = stream.read(frames_per_chunk)

                    if overflowed:
                        print("[STT] ⚠️ Audio overflow detected")

                    audio = data.squeeze()
                    volume = float(np.max(np.abs(audio))) if audio.size else 0.0

                    if volume >= self.silence_threshold:
                        if not is_speaking:
                            print("[STT] Speech detected")
                            is_speaking = True
                        buffer.append(audio)
                        silent_count = 0
                    else:
                        if is_speaking:
                            silent_count += 1

                    if is_speaking and silent_count >= self.silence_chunks:
                        print(
                            f"[STT] Speech ended → enqueue audio "
                            f"(samples={sum(len(b) for b in buffer)})"
                        )
                        self._audio_queue.put(np.concatenate(buffer))
                        buffer.clear()
                        silent_count = 0
                        is_speaking = False

        except KeyboardInterrupt:
            print("[STT] Listening stopped")
            self.stop()

    # ==================================================
    # STT Worker Thread
    # ==================================================
    def _stt_worker(self):
        print("[STT] STT worker started")

        while not self._stop_event.is_set():
            try:
                audio_in = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            print(f"[STT] Dequeued audio (len={audio_in.shape[0]})")

            audio_16k = self._resample_to_16k(audio_in)
            print(f"[STT] Resampled to 16k (len={audio_16k.shape[0]})")

            min_samples = int(self.sample_rate * self.min_utterance_seconds)
            if audio_16k.size < min_samples:
                print("[STT] Ignored short utterance")
                continue

            print("[STT] Running whisper transcription...")
            try:
                segments, _ = self.model.transcribe(
                    audio_16k,
                    language="ko",
                    beam_size=self.beam_size,
                    temperature=self.temperature,
                    vad_filter=False,
                )
            except Exception as e:
                print(f"[STT] Whisper error: {repr(e)}")
                continue

            print("[STT] Whisper transcription finished")

            text = "".join(seg.text for seg in segments).strip()

            if not text or len(text) < self.min_text_len:
                print("[STT] Empty or too-short transcription")
                continue

            print(f"[STT] ✅ Transcribed text: {text}")

            if self.on_text:
                self.on_text(text)

    # ==================================================
    # Stop
    # ==================================================
    def stop(self):
        self._stop_event.set()
        print("[STT] Stopping STT engine")