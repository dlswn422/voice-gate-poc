# src/speech/faster_whisper_stt.py
from __future__ import annotations

import os
import threading
import queue
import time
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
    실시간 STT 엔진 (비동기 안정 + VAD 안정화 + Auto-stop 옵션)

    구조
    - Audio Thread (sounddevice InputStream)
        - 마이크 입력
        - 발화 감지(VAD-lite)
        - 발화 종료 시 오디오 버퍼를 Queue에 enqueue
    - STT Worker Thread
        - Whisper 추론만 담당 (절대 Audio Thread 안에서 실행 ❌)

    개선 사항
    1) VAD volume 계산을 max -> RMS로 변경 (튀는 노이즈에 강함)
    2) 시작 시 노이즈 바닥값을 측정해 silence_threshold 자동 보정(선택)
    3) idle_timeout_sec 옵션: 일정 시간 무음이면 listening 자체를 종료(선택)
    4) stream.read 예외/stop_event 처리 강화
    """

    def __init__(
        self,
        model_size: str = "medium",
        device_index: Optional[int] = None,
        sample_rate: int = 16000,
        input_sample_rate: int = 48000,
        chunk_seconds: float = 0.5,
        silence_threshold: float = 0.03,
        silence_chunks: int = 3,
        min_utterance_seconds: float = 0.6,
        min_text_len: int = 2,
        beam_size: int = 1,
        temperature: float = 0.0,
        download_root: str = "models",
        # ✅ 추가 옵션
        auto_calibrate_noise: bool = True,
        noise_calib_seconds: float = 1.0,
        noise_multiplier: float = 3.0,
        idle_timeout_sec: Optional[float] = None,  # 예: 20.0 (None이면 자동종료 없음)
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

        self.auto_calibrate_noise = auto_calibrate_noise
        self.noise_calib_seconds = noise_calib_seconds
        self.noise_multiplier = noise_multiplier
        self.idle_timeout_sec = idle_timeout_sec

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
            segments, _ = self.model.transcribe(
                dummy,
                language="ko",
                beam_size=1,
                temperature=0.0,
                vad_filter=False,
            )
            # consume generator / iterator
            list(segments)
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
    # Noise calibration (optional)
    # ==================================================
    def _measure_noise_floor(self, stream: sd.InputStream, frames_per_chunk: int, secs: float) -> float:
        """
        무음 상태에서 RMS 기반 노이즈 바닥값 측정
        """
        n_chunks = max(1, int(secs / max(self.chunk_seconds, 1e-6)))
        mx = 0.0
        for _ in range(n_chunks):
            data, overflowed = stream.read(frames_per_chunk)
            if overflowed:
                print("[STT] ⚠️ Audio overflow during noise calibration")
            audio = data.squeeze()
            # RMS
            v = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
            mx = max(mx, v)
        return mx

    # ==================================================
    # Audio Thread
    # ==================================================
    def start_listening(self):
        print("[STT] Listening started (Ctrl+C to stop)")

        buffer: list[np.ndarray] = []
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

                # ✅ 노이즈 자동 보정 (선택)
                if self.auto_calibrate_noise:
                    try:
                        print("[STT] Calibrating noise floor... (stay quiet)")
                        noise = self._measure_noise_floor(stream, frames_per_chunk, secs=self.noise_calib_seconds)
                        auto_th = max(self.silence_threshold, noise * self.noise_multiplier)
                        print(f"[STT] noise_floor={noise:.6f} -> silence_threshold={auto_th:.6f}")
                        self.silence_threshold = auto_th
                    except Exception as e:
                        print(f"[STT] Noise calibration failed (ignored): {repr(e)}")

                last_active_ts = time.time()

                while not self._stop_event.is_set():
                    # ✅ idle timeout: 일정 시간 무음이면 listening 종료 (선택)
                    if self.idle_timeout_sec is not None:
                        if (time.time() - last_active_ts) >= float(self.idle_timeout_sec):
                            print(f"[STT] Idle timeout reached ({self.idle_timeout_sec}s). Stopping listening.")
                            self.stop()
                            break

                    data, overflowed = stream.read(frames_per_chunk)

                    if overflowed:
                        print("[STT] ⚠️ Audio overflow detected")

                    audio = data.squeeze()
                    # ✅ RMS 기반 volume (튀는 노이즈에 강함)
                    volume = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0

                    if volume >= self.silence_threshold:
                        last_active_ts = time.time()

                        if not is_speaking:
                            print("[STT] Speech detected")
                            is_speaking = True

                        buffer.append(audio)
                        silent_count = 0
                    else:
                        # 무음
                        if is_speaking:
                            silent_count += 1

                    if is_speaking and silent_count >= self.silence_chunks:
                        total_samples = sum(len(b) for b in buffer)
                        print(f"[STT] Speech ended → enqueue audio (samples={total_samples})")

                        if total_samples > 0:
                            self._audio_queue.put(np.concatenate(buffer))

                        buffer.clear()
                        silent_count = 0
                        is_speaking = False

        except KeyboardInterrupt:
            print("[STT] Listening stopped (KeyboardInterrupt)")
            self.stop()
        except Exception as e:
            print(f"[STT] Listening error: {repr(e)}")
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

            try:
                print(f"[STT] Dequeued audio (len={audio_in.shape[0]})")

                audio_16k = self._resample_to_16k(audio_in)
                print(f"[STT] Resampled to 16k (len={audio_16k.shape[0]})")

                min_samples = int(self.sample_rate * self.min_utterance_seconds)
                if audio_16k.size < min_samples:
                    print("[STT] Ignored short utterance")
                    continue

                print("[STT] Running whisper transcription...")
                segments, _ = self.model.transcribe(
                    audio_16k,
                    language="ko",
                    beam_size=self.beam_size,
                    temperature=self.temperature,
                    vad_filter=False,
                )
                # segments는 iterator/generator 형태일 수 있으니 한 번 소비
                seg_list = list(segments)

                print("[STT] Whisper transcription finished")

                text = "".join(seg.text for seg in seg_list).strip()

                if not text or len(text) < self.min_text_len:
                    print("[STT] Empty or too-short transcription")
                    continue

                print(f"[STT] ✅ Transcribed text: {text}")

                if self.on_text:
                    self.on_text(text)

            except Exception as e:
                print(f"[STT] Worker error: {repr(e)}")
                continue

    # ==================================================
    # Stop
    # ==================================================
    def stop(self):
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        print("[STT] Stopping STT engine")
