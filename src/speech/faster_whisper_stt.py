import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from typing import Optional, Callable

from scipy.signal import resample_poly


class FasterWhisperSTT:
    """
    실시간 STT 엔진 (Windows 안정화 + 품질 고도화)

    ✅ 개선 포인트(이번 수정)
    - max(abs) 대신 RMS 기반 에너지로 VAD(종료 감지) 안정화
    - 시작/종료 임계값을 분리(히스테리시스): start_threshold > end_threshold
    - max_utterance_seconds로 발화가 끝 안 잡혀도 강제 종료 -> 전사 진행
    - 로그 옵션 강화: speaking 상태에서 energy를 찍어 원인 파악 쉬움
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device_index: Optional[int] = None,
        sample_rate: int = 16000,
        input_sample_rate: int = 48000,
        chunk_seconds: float = 0.5,

        # ✅ (중요) 시작/종료 임계값 분리: start_threshold는 "말 시작", end_threshold는 "무음"
        start_threshold: float = 0.020,
        end_threshold: float = 0.012,

        # 무음 청크 수(0.5s chunk 기준 4면 약 2초 무음)
        silence_chunks: int = 4,

        # 너무 짧은 발화/텍스트 제거 기준
        min_utterance_seconds: float = 0.6,
        min_text_len: int = 2,

        # ✅ 발화가 끝을 못 잡아도 강제 종료(초)
        max_utterance_seconds: float = 8.0,

        # 디코딩 안정화
        beam_size: int = 5,
        temperature: float = 0.0,

        # 로그
        log_audio_stats: bool = True,
    ):
        self.sample_rate = sample_rate
        self.input_sample_rate = input_sample_rate
        self.chunk_seconds = chunk_seconds

        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.silence_chunks = silence_chunks

        self.device_index = device_index

        self.min_utterance_seconds = min_utterance_seconds
        self.min_text_len = min_text_len
        self.max_utterance_seconds = max_utterance_seconds

        self.beam_size = beam_size
        self.temperature = temperature
        self.log_audio_stats = log_audio_stats

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
        """input_sample_rate -> sample_rate (예: 48000 -> 16000)"""
        if self.input_sample_rate == self.sample_rate:
            return audio_1d.astype(np.float32)

        if self.input_sample_rate == 48000 and self.sample_rate == 16000:
            return resample_poly(audio_1d, up=1, down=3).astype(np.float32)

        up = self.sample_rate
        down = self.input_sample_rate
        return resample_poly(audio_1d, up=up, down=down).astype(np.float32)

    @staticmethod
    def _rms_energy(x: np.ndarray) -> float:
        """RMS 에너지(잡음/피크에 덜 민감)"""
        if x.size == 0:
            return 0.0
        # float64로 올려서 계산 안정성
        xx = x.astype(np.float64)
        return float(np.sqrt(np.mean(xx * xx) + 1e-12))

    def start_listening(self):
        """마이크 입력 수신 → 발화 감지 → 종료 시 STT 수행"""
        print("[STT] Listening started (Ctrl+C to stop)")

        buffer = []
        silent_count = 0
        is_speaking = False

        frames_per_chunk = int(self.chunk_seconds * self.input_sample_rate)

        # ✅ 강제 종료 타이머용
        speaking_chunk_count = 0
        max_speaking_chunks = max(1, int(self.max_utterance_seconds / self.chunk_seconds))

        # ✅ overflow 추적(디버깅)
        overflow_hits = 0

        try:
            with sd.InputStream(
                samplerate=self.input_sample_rate,
                device=self.device_index,
                channels=1,
                dtype="float32",
                blocksize=frames_per_chunk,
            ) as stream:
                while True:
                    data, overflowed = stream.read(frames_per_chunk)

                    if overflowed:
                        overflow_hits += 1
                        # overflow가 잦으면 chunk_seconds를 1.0으로 늘리는 것도 추천
                        if self.log_audio_stats and overflow_hits % 10 == 0:
                            print(f"[STT] ⚠ overflowed count={overflow_hits}")

                    audio = np.asarray(data, dtype=np.float32).squeeze()

                    # ✅ max(abs) 대신 RMS 사용
                    energy = self._rms_energy(audio)

                    # ---- 시작 감지 ----
                    if not is_speaking:
                        if energy >= self.start_threshold:
                            print("[STT] Speech detected")
                            is_speaking = True
                            buffer.append(audio)
                            silent_count = 0
                            speaking_chunk_count = 1
                            if self.log_audio_stats:
                                print(f"[STT] energy(start)={energy:.4f} start_th={self.start_threshold:.4f}")
                        else:
                            # 아직 무음 상태 (원인 파악용: 너무 민감/둔감 확인)
                            if self.log_audio_stats:
                                # 너무 시끄러운 환경이면 여기 energy가 계속 높게 찍힘
                                pass
                            continue

                    # ---- speaking 상태 ----
                    else:
                        buffer.append(audio)
                        speaking_chunk_count += 1

                        # ✅ 종료 감지(히스테리시스): end_threshold 아래로 떨어지면 무음 카운트++
                        if energy < self.end_threshold:
                            silent_count += 1
                        else:
                            silent_count = 0

                        if self.log_audio_stats:
                            # speaking 중 에너지/무음카운트 추적 (왜 안 끝나는지 바로 보임)
                            print(
                                f"[STT] energy={energy:.4f} end_th={self.end_threshold:.4f} "
                                f"silent_count={silent_count}/{self.silence_chunks} "
                                f"dur={speaking_chunk_count*self.chunk_seconds:.1f}s"
                            )

                        # ✅ 발화 종료 판단: 무음이 연속 silence_chunks 이상
                        if silent_count >= self.silence_chunks:
                            print("[STT] Speech ended, running transcription")
                            self._process_buffer(buffer)
                            buffer.clear()
                            silent_count = 0
                            is_speaking = False
                            speaking_chunk_count = 0
                            continue

                        # ✅ 강제 종료: 끝이 안 잡혀도 일정 시간 지나면 전사
                        if speaking_chunk_count >= max_speaking_chunks:
                            print("[STT] Max utterance reached, forcing transcription")
                            self._process_buffer(buffer)
                            buffer.clear()
                            silent_count = 0
                            is_speaking = False
                            speaking_chunk_count = 0
                            continue

        except KeyboardInterrupt:
            print("[STT] Listening stopped")

    def _process_buffer(self, buffer):
        """누적 오디오 → 16k 변환 → Whisper STT"""
        if not buffer:
            return

        audio_in = np.concatenate(buffer)
        audio_16k = self._resample_to_16k(audio_in)

        # 너무 짧은 발화 컷
        min_samples = int(self.sample_rate * self.min_utterance_seconds)
        if audio_16k.size < min_samples:
            if self.log_audio_stats:
                dur = audio_16k.size / float(self.sample_rate)
                print(f"[STT] Ignored short utterance: {dur:.2f}s")
            return

        if self.log_audio_stats:
            peak = float(np.max(np.abs(audio_16k))) if audio_16k.size else 0.0
            mean_abs = float(np.mean(np.abs(audio_16k))) if audio_16k.size else 0.0
            dur = audio_16k.size / float(self.sample_rate)
            print(f"[STT] audio_stats: sec={dur:.2f}, peak={peak:.4f}, mean_abs={mean_abs:.4f}")

        segments, _info = self.model.transcribe(
            audio_16k,
            language="ko",
            beam_size=self.beam_size,
            temperature=self.temperature,
            vad_filter=False,
        )

        text = "".join(seg.text for seg in segments).strip()

        if not text or len(text) < self.min_text_len:
            print("[STT] No transcription result")
            return

        print(f"[STT] Transcribed text: {text}")

        if self.on_text:
            self.on_text(text)
