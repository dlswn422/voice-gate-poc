# from __future__ import annotations

# import os
# import threading
# import queue
# import time
# import uuid
# from typing import Optional, Callable

# import sounddevice as sd
# import numpy as np
# from faster_whisper import WhisperModel
# from scipy.signal import resample_poly


# # --------------------------------------------------
# # Windows + ctranslate2 안정화
# # --------------------------------------------------
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")


# class FasterWhisperSTT:
#     """
#     STT TRACE VERSION (FINAL - SYNC SAFE)

#     ✔ 기존 로그 전부 유지
#     ✔ 최신 발화만 처리 (queue size = 1)
#     ✔ 처리 중 추가 발화 시 UX 안내 출력
#     ✔ utterance_id 기반 AppEngine 세션 동기화 가능

#     STT는 의미 판단 ❌
#     STT는 "정확하고 최신 텍스트"만 전달
#     """

#     FILLER_WORDS = ["어", "음", "저기", "그", "아", "뭐지", "이제"]

#     def __init__(
#         self,
#         model_size: str = "medium",
#         device_index: Optional[int] = None,
#         sample_rate: int = 16000,
#         input_sample_rate: int = 48000,
#         chunk_seconds: float = 0.3,
#         silence_threshold: float = 0.03,
#         silence_chunks: int = 2,
#         min_utterance_seconds: float = 0.4,
#         min_text_len: int = 2,
#         beam_size: int = 1,
#         temperature: float = 0.0,
#         download_root: str = "models",
#         auto_calibrate_noise: bool = True,
#         noise_calib_seconds: float = 1.0,
#         noise_multiplier: float = 4.0,
#     ):
#         self.sample_rate = sample_rate
#         self.input_sample_rate = input_sample_rate
#         self.chunk_seconds = chunk_seconds
#         self.silence_threshold = silence_threshold
#         self.silence_chunks = silence_chunks
#         self.device_index = device_index

#         self.min_utterance_seconds = min_utterance_seconds
#         self.min_text_len = min_text_len

#         self.beam_size = beam_size
#         self.temperature = temperature

#         self.auto_calibrate_noise = auto_calibrate_noise
#         self.noise_calib_seconds = noise_calib_seconds
#         self.noise_multiplier = noise_multiplier

#         # 콜백: (text, utterance_id)
#         self.on_text: Optional[Callable[[str, str], None]] = None

#         # 🔥 최신 발화만 유지
#         self._audio_queue: queue.Queue[
#             tuple[str, np.ndarray, float]
#         ] = queue.Queue(maxsize=1)

#         self._stop_event = threading.Event()
#         self._is_processing = False

#         # --------------------------------------------------
#         # Whisper 모델 로드
#         # --------------------------------------------------
#         print("[STT] Loading Faster-Whisper model...")
#         self.model = WhisperModel(
#             model_size,
#             device="cpu",
#             compute_type="float32",
#             download_root=download_root,
#         )
#         print("[STT] Model ready")

#         self._worker_thread = threading.Thread(
#             target=self._stt_worker,
#             daemon=True,
#         )
#         self._worker_thread.start()
#         print("[STT] Worker started")

#         self._warmup()

#     # ==================================================
#     # Warm-up
#     # ==================================================
#     def _warmup(self):
#         dummy = np.zeros(int(self.sample_rate * 1.0), dtype=np.float32)
#         try:
#             list(
#                 self.model.transcribe(
#                     dummy,
#                     language="ko",
#                     beam_size=1,
#                     temperature=0.0,
#                     vad_filter=False,
#                 )[0]
#             )
#         except Exception:
#             pass

#     # ==================================================
#     # Resample
#     # ==================================================
#     def _resample_to_16k(self, audio: np.ndarray) -> np.ndarray:
#         if self.input_sample_rate == self.sample_rate:
#             return audio.astype(np.float32)
#         if self.input_sample_rate == 48000:
#             return resample_poly(audio, up=1, down=3).astype(np.float32)
#         return resample_poly(
#             audio, up=self.sample_rate, down=self.input_sample_rate
#         ).astype(np.float32)

#     # ==================================================
#     # STT 전처리
#     # ==================================================
#     def _clean_text(self, text: str) -> str:
#         tokens = [t for t in text.strip().split() if t not in self.FILLER_WORDS]
#         cleaned = []
#         for t in tokens:
#             if not cleaned or cleaned[-1] != t:
#                 cleaned.append(t)
#         return " ".join(cleaned).strip()

#     # ==================================================
#     # Listening Thread
#     # ==================================================
#     def start_listening(self):
#         print("[STT] 🎧 Listening... (Ctrl+C to stop)")

#         buffer = []
#         silent_count = 0
#         is_speaking = False
#         speech_start_ts: Optional[float] = None

#         frames_per_chunk = int(self.chunk_seconds * self.input_sample_rate)

#         try:
#             with sd.InputStream(
#                 samplerate=self.input_sample_rate,
#                 device=self.device_index,
#                 channels=1,
#                 dtype="float32",
#                 blocksize=frames_per_chunk,
#             ) as stream:

#                 if self.auto_calibrate_noise:
#                     noise = self._measure_noise_floor(
#                         stream, frames_per_chunk, self.noise_calib_seconds
#                     )
#                     self.silence_threshold = max(
#                         self.silence_threshold,
#                         noise * self.noise_multiplier,
#                     )
#                     print(f"[STT] 🔧 silence_threshold={self.silence_threshold:.5f}")

#                 while not self._stop_event.is_set():
#                     data, overflowed = stream.read(frames_per_chunk)
#                     if overflowed:
#                         print("[STT] ⚠️ Audio overflow")

#                     audio = data.squeeze()
#                     volume = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0

#                     # 🔥 처리 중 UX
#                     if self._is_processing and volume >= self.silence_threshold:
#                         print("[STT] ⏳ 처리 중입니다. 잠시만 기다려 주세요.")
#                         continue

#                     if volume >= self.silence_threshold:
#                         if not is_speaking:
#                             speech_start_ts = time.time()
#                             print("[STT] 🟢 Speech detected")
#                         is_speaking = True
#                         buffer.append(audio)
#                         silent_count = 0
#                     else:
#                         if is_speaking:
#                             silent_count += 1

#                     if is_speaking and silent_count >= self.silence_chunks:
#                         speech_end_ts = time.time()
#                         vad_latency = (speech_end_ts - speech_start_ts) * 1000
#                         print(f"[STT] 🔵 Speech ended (VAD latency={vad_latency:.0f} ms)")

#                         if buffer:
#                             utterance_id = str(uuid.uuid4())

#                             while not self._audio_queue.empty():
#                                 try:
#                                     self._audio_queue.get_nowait()
#                                 except queue.Empty:
#                                     break

#                             self._audio_queue.put(
#                                 (utterance_id, np.concatenate(buffer), speech_end_ts)
#                             )

#                         buffer.clear()
#                         silent_count = 0
#                         is_speaking = False

#         except KeyboardInterrupt:
#             self.stop()
#         except Exception as e:
#             print("[STT] Listening error:", repr(e))
#             self.stop()

#     # ==================================================
#     # Noise calibration
#     # ==================================================
#     def _measure_noise_floor(self, stream, frames_per_chunk, secs):
#         n_chunks = max(1, int(secs / self.chunk_seconds))
#         mx = 0.0
#         for _ in range(n_chunks):
#             data, _ = stream.read(frames_per_chunk)
#             audio = data.squeeze()
#             v = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
#             mx = max(mx, v)
#         return mx

#     # ==================================================
#     # STT Worker Thread
#     # ==================================================
#     def _stt_worker(self):
#         print("[STT-WORKER] 🧵 Worker loop started")

#         while not self._stop_event.is_set():
#             try:
#                 utterance_id, audio_in, speech_end_ts = self._audio_queue.get(timeout=0.1)
#                 dequeue_ts = time.time()
#                 queue_delay = (dequeue_ts - speech_end_ts) * 1000
#                 print(f"[STT-WORKER] 📥 Audio dequeued (queue_delay={queue_delay:.0f} ms)")
#             except queue.Empty:
#                 continue

#             try:
#                 self._is_processing = True

#                 audio_16k = self._resample_to_16k(audio_in)
#                 if audio_16k.size < int(self.sample_rate * self.min_utterance_seconds):
#                     print("[STT-WORKER] ⚠️ Too short audio, dropped")
#                     continue

#                 t0 = time.time()
#                 segments, _ = self.model.transcribe(
#                     audio_16k,
#                     language="ko",
#                     beam_size=self.beam_size,
#                     temperature=self.temperature,
#                     vad_filter=False,
#                 )
#                 t1 = time.time()

#                 whisper_ms = (t1 - t0) * 1000
#                 total_ms = (t1 - speech_end_ts) * 1000

#                 raw_text = "".join(seg.text for seg in segments).strip()
#                 text = self._clean_text(raw_text)

#                 print(
#                     f"[STT-TIMING] queue={queue_delay:.0f} ms | "
#                     f"whisper={whisper_ms:.0f} ms | total={total_ms:.0f} ms"
#                 )

#                 if not text or len(text) < self.min_text_len:
#                     print("[STT-WORKER] ⚠️ Empty/short text, skipped")
#                     continue

#                 print(f"[STT] 🎤 \"{text}\"")

#                 if self.on_text:
#                     self.on_text(text, utterance_id=utterance_id)

#             except Exception as e:
#                 print("[STT-WORKER] ❌ Worker error:", repr(e))
#             finally:
#                 self._is_processing = False

#     # ==================================================
#     # Stop
#     # ==================================================
#     def stop(self):
#         if self._stop_event.is_set():
#             return
#         self._stop_event.set()
#         print("[STT] Shutdown")

import threading
from typing import Optional, Callable

class FasterWhisperSTT:
 

    def __init__(self, **kwargs):
        self.on_text: Optional[Callable[[str], None]] = None
     
        self._stop_event = threading.Event() 
        
    def _clean_text(self, text: str) -> str:
        FILLER_WORDS = ["어", "음", "저기", "그", "아", "뭐지", "이제"]
        text = text.strip()
        tokens = [t for t in text.split() if t not in FILLER_WORDS]
        cleaned = []
        for t in tokens:
            if not cleaned or cleaned[-1] != t:
                cleaned.append(t)
        return " ".join(cleaned).strip()

    def start_listening(self):
        print("\n" + "="*50)
        print("[TEST] 마이크 대신 여기에 텍스트를 입력하고 Enter를 치세요.")
        print("="*50 + "\n")

        try:
            while not self._stop_event.is_set():
                user_input = input("🎤 (STT Input) >> ").strip()

                if user_input.lower() in ['exit', 'quit']:
                    break
                
                if not user_input:
                    continue

                processed_text = self._clean_text(user_input)
                print(f"[STT-RESULT] \"{processed_text}\"")
                
                if self.on_text:
                    self.on_text(processed_text)

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        self._stop_event.set()
        print("[STT] 시뮬레이터 종료")