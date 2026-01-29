import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from typing import Optional, Callable

from scipy.signal import resample_poly


class FasterWhisperSTT:
    """
    실시간 STT 엔진 (Windows 안정화 + 품질 고도화)

    목표(고도화 포인트)
    - Windows에서 InputStream 샘플레이트 이슈 방지(48k 입력 → 16k 변환)
    - Whisper 추론은 발화 종료 시점에만 수행(지연/비용 최소화)
    - Whisper 내부 VAD는 OFF (onnxruntime 의존 제거)
    - 짧은 발화/잡음성 입력을 컷하여 오인식 감소
    - 디코딩 파라미터(beam/temperature)로 결과 안정화
    """

    def __init__(
        self,
        # ✅ 팀 합의: large-v3 기본
        model_size: str = "large-v3",
        device_index: Optional[int] = None,
        # ✅ Whisper 입력(최종) 샘플레이트
        sample_rate: int = 16000,
        # ✅ 마이크 스트림 샘플레이트(Windows에서 48000이 안전한 경우가 많음)
        input_sample_rate: int = 48000,
        # ✅ 청크 단위(너무 작으면 오버플로우/CPU부하 증가 가능)
        chunk_seconds: float = 0.5,
        # ✅ 볼륨 기반 간이 VAD 임계값/무음 청크 수
        silence_threshold: float = 0.015,
        silence_chunks: int = 2,
        # ✅ 고도화: 너무 짧은 음성/텍스트 제거 기준
        min_utterance_seconds: float = 0.6,
        min_text_len: int = 2,
        # ✅ 디코딩 안정화(정확도/속도 트레이드오프)
        beam_size: int = 5,
        temperature: float = 0.0,
        # ✅ 튜닝/보고용 로그
        log_audio_stats: bool = False,
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

        # 가장 흔한 케이스 최적화: 48k -> 16k (down=3)
        if self.input_sample_rate == 48000 and self.sample_rate == 16000:
            return resample_poly(audio_1d, up=1, down=3).astype(np.float32)

        # 일반 케이스(비율 계산)
        up = self.sample_rate
        down = self.input_sample_rate
        return resample_poly(audio_1d, up=up, down=down).astype(np.float32)

    def start_listening(self):
        """마이크 입력 수신 → 발화 감지 → 종료 시 STT 수행"""
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
                while True:
                    data, overflowed = stream.read(frames_per_chunk)
                    if overflowed:
                        # 오버플로우가 잦으면 chunk_seconds를 1.0으로 늘리면 안정되는 경우가 많음
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
        """누적 오디오 → 16k 변환 → Whisper STT"""
        if not buffer:
            return

        audio_in = np.concatenate(buffer)
        audio_16k = self._resample_to_16k(audio_in)

        # ✅ 고도화 1) 너무 짧은 발화 컷(잡음/숨소리/짧은 클릭음 방지)
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

        # ✅ 고도화 2) 디코딩 안정화(temperature=0.0, beam_size로 품질↑)
        # ✅ Whisper 내부 VAD OFF (onnxruntime 불필요)
        segments, _info = self.model.transcribe(
            audio_16k,
            language="ko",
            beam_size=self.beam_size,
            temperature=self.temperature,
            vad_filter=False,
        )

        text = "".join(seg.text for seg in segments).strip()

        # ✅ 고도화 3) 너무 짧은 텍스트 컷(잡음성 출력 방지)
        if not text or len(text) < self.min_text_len:
            print("[STT] No transcription result")
            return

        print(f"[STT] Transcribed text: {text}")

        if self.on_text:
            self.on_text(text)
