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
# CPU 추론 시 과도한 스레드 사용 방지
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


class FasterWhisperSTT:
    """
    STT TRACE VERSION (FINAL)

    목적
    - 음성 → 텍스트 전체 파이프라인 latency 추적
    - STT 결과를 1차 Intent 분류에 "최적의 입력"으로 제공

    주요 로그
    - Speech detected 시점
    - Speech ended (VAD latency)
    - STT worker queue delay
    - Whisper inference time
    - Total STT latency

    설계 원칙
    - STT는 판단하지 않는다 (의미 해석 ❌)
    - 최대한 깨끗한 텍스트만 AppEngine으로 전달
    """

    # --------------------------------------------------
    # STT 전처리용 filler word 목록
    # --------------------------------------------------
    FILLER_WORDS = [
        "어", "음", "저기", "그", "아", "뭐지", "이제",
    ]

    def __init__(
        self,
        model_size: str = "medium",
        device_index: Optional[int] = None,
        sample_rate: int = 16000,
        input_sample_rate: int = 48000,
        chunk_seconds: float = 0.3,
        silence_threshold: float = 0.03,
        silence_chunks: int = 2,
        min_utterance_seconds: float = 0.4,
        min_text_len: int = 2,
        beam_size: int = 1,
        temperature: float = 0.0,
        download_root: str = "models",
        auto_calibrate_noise: bool = True,
        noise_calib_seconds: float = 1.0,
        noise_multiplier: float = 4.0,
        idle_timeout_sec: Optional[float] = None,
    ):
        # 오디오 관련 설정
        self.sample_rate = sample_rate
        self.input_sample_rate = input_sample_rate
        self.chunk_seconds = chunk_seconds
        self.silence_threshold = silence_threshold
        self.silence_chunks = silence_chunks
        self.device_index = device_index

        # 발화 필터링 기준
        self.min_utterance_seconds = min_utterance_seconds
        self.min_text_len = min_text_len

        # Whisper 추론 옵션
        self.beam_size = beam_size
        self.temperature = temperature

        # 노이즈 캘리브레이션
        self.auto_calibrate_noise = auto_calibrate_noise
        self.noise_calib_seconds = noise_calib_seconds
        self.noise_multiplier = noise_multiplier
        self.idle_timeout_sec = idle_timeout_sec

        # STT 결과 콜백
        self.on_text: Optional[Callable[[str], None]] = None

        # (audio, speech_end_ts)
        self._audio_queue: queue.Queue[tuple[np.ndarray, float]] = queue.Queue()
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
        print("[STT] Model ready")

        # --------------------------------------------------
        # STT 워커 스레드
        # --------------------------------------------------
        self._worker_thread = threading.Thread(
            target=self._stt_worker,
            daemon=True,
        )
        self._worker_thread.start()
        print("[STT] Worker started")

        # Whisper warm-up
        self._warmup()

    # ==================================================
    # Warm-up
    # ==================================================
    def _warmup(self):
        """초기 추론 지연 제거용 더미 추론"""
        dummy = np.zeros(int(self.sample_rate * 1.0), dtype=np.float32)
        try:
            list(
                self.model.transcribe(
                    dummy,
                    language="ko",
                    beam_size=1,
                    temperature=0.0,
                    vad_filter=False,
                )[0]
            )
        except Exception:
            pass

    # ==================================================
    # Resample
    # ==================================================
    def _resample_to_16k(self, audio: np.ndarray) -> np.ndarray:
        if self.input_sample_rate == self.sample_rate:
            return audio.astype(np.float32)

        if self.input_sample_rate == 48000:
            return resample_poly(audio, up=1, down=3).astype(np.float32)

        return resample_poly(
            audio,
            up=self.sample_rate,
            down=self.input_sample_rate,
        ).astype(np.float32)

    # ==================================================
    # STT 전처리
    # ==================================================
    def _clean_text(self, text: str) -> str:
        """
        STT 결과를 1차 Intent 분류에 적합하게 정리
        - filler word 제거
        - 중복 단어 정리
        """
        text = text.strip()

        # filler 제거
        tokens = [t for t in text.split() if t not in self.FILLER_WORDS]

        # 연속 중복 제거
        cleaned = []
        for t in tokens:
            if not cleaned or cleaned[-1] != t:
                cleaned.append(t)

        return " ".join(cleaned).strip()

    # ==================================================
    # Listening Thread
    # ==================================================
    def start_listening(self):
        print("[STT] 🎧 Listening... (Ctrl+C to stop)")

        buffer: list[np.ndarray] = []
        silent_count = 0
        is_speaking = False

        speech_start_ts: Optional[float] = None
        frames_per_chunk = int(self.chunk_seconds * self.input_sample_rate)

        try:
            with sd.InputStream(
                samplerate=self.input_sample_rate,
                device=self.device_index,
                channels=1,
                dtype="float32",
                blocksize=frames_per_chunk,
            ) as stream:

                # ------------------------------------------
                # 노이즈 자동 캘리브레이션
                # ------------------------------------------
                if self.auto_calibrate_noise:
                    noise = self._measure_noise_floor(
                        stream,
                        frames_per_chunk,
                        self.noise_calib_seconds,
                    )
                    self.silence_threshold = max(
                        self.silence_threshold,
                        noise * self.noise_multiplier,
                    )
                    print(f"[STT] 🔧 silence_threshold={self.silence_threshold:.5f}")

                while not self._stop_event.is_set():
                    data, overflowed = stream.read(frames_per_chunk)
                    if overflowed:
                        print("[STT] ⚠️ Audio overflow")

                    audio = data.squeeze()
                    volume = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0

                    if volume >= self.silence_threshold:
                        if not is_speaking:
                            speech_start_ts = time.time()
                            print("[STT] 🟢 Speech detected")

                        is_speaking = True
                        buffer.append(audio)
                        silent_count = 0
                    else:
                        if is_speaking:
                            silent_count += 1

                    # ------------------------------------------
                    # VAD 종료 조건
                    # ------------------------------------------
                    if is_speaking and silent_count >= self.silence_chunks:
                        speech_end_ts = time.time()
                        vad_latency = (speech_end_ts - speech_start_ts) * 1000

                        print(
                            f"[STT] 🔵 Speech ended "
                            f"(VAD latency={vad_latency:.0f} ms)"
                        )

                        if buffer:
                            self._audio_queue.put(
                                (np.concatenate(buffer), speech_end_ts)
                            )

                        buffer.clear()
                        silent_count = 0
                        is_speaking = False

        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            print("[STT] Listening error:", repr(e))
            self.stop()

    # ==================================================
    # Noise calibration
    # ==================================================
    def _measure_noise_floor(
        self,
        stream: sd.InputStream,
        frames_per_chunk: int,
        secs: float,
    ) -> float:
        n_chunks = max(1, int(secs / self.chunk_seconds))
        mx = 0.0
        for _ in range(n_chunks):
            data, _ = stream.read(frames_per_chunk)
            audio = data.squeeze()
            v = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
            mx = max(mx, v)
        return mx

    # ==================================================
    # STT Worker Thread
    # ==================================================
    def _stt_worker(self):
        print("[STT-WORKER] 🧵 Worker loop started")

        while not self._stop_event.is_set():
            try:
                audio_in, speech_end_ts = self._audio_queue.get(timeout=0.1)
                dequeue_ts = time.time()
                queue_delay = (dequeue_ts - speech_end_ts) * 1000

                print(
                    f"[STT-WORKER] 📥 Audio dequeued "
                    f"(queue_delay={queue_delay:.0f} ms)"
                )

            except queue.Empty:
                continue

            try:
                audio_16k = self._resample_to_16k(audio_in)

                # 너무 짧은 발화 제거
                min_samples = int(self.sample_rate * self.min_utterance_seconds)
                if audio_16k.size < min_samples:
                    print("[STT-WORKER] ⚠️ Too short audio, dropped")
                    continue

                # Whisper 추론
                t0 = time.time()
                segments, _ = self.model.transcribe(
                    audio_16k,
                    language="ko",
                    beam_size=self.beam_size,
                    temperature=self.temperature,
                    vad_filter=False,
                )
                t1 = time.time()

                whisper_ms = (t1 - t0) * 1000
                total_ms = (t1 - speech_end_ts) * 1000

                raw_text = "".join(seg.text for seg in segments).strip()
                text = self._clean_text(raw_text)

                print(
                    f"[STT-TIMING] "
                    f"queue={queue_delay:.0f} ms | "
                    f"whisper={whisper_ms:.0f} ms | "
                    f"total={total_ms:.0f} ms"
                )

                if not text or len(text) < self.min_text_len:
                    print("[STT-WORKER] ⚠️ Empty/short text, skipped")
                    continue

                print(f"[STT] 🎤 \"{text}\"")

                if self.on_text:
                    self.on_text(text)

            except Exception as e:
                print("[STT-WORKER] ❌ Worker error:", repr(e))

    # ==================================================
    # Stop
    # ==================================================
    def stop(self):
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        print("[STT] Shutdown")