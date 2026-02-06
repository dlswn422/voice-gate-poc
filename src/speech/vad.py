import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps


class VoiceActivityDetector:
    """
    Silero VAD 기반 실시간 음성 활동 감지기

    역할:
    - PCM(Float32) 오디오 스트림 입력
    - 현재 chunk가 음성인지 여부 판단
    - 일정 시간 무음 지속 시 '발화 종료' 판단

    전제:
    - sample_rate = 16000
    - mono channel
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        speech_threshold: float = 0.5,
        min_speech_duration_ms: int = 200,
        min_silence_duration_ms: int = 600,
    ):
        self.sample_rate = sample_rate
        self.speech_threshold = speech_threshold

        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms

        # 🔥 발화 종료 판단용 (초 단위)
        self.end_silence_sec = min_silence_duration_ms / 1000.0

        # Silero VAD 모델 로드 (CPU)
        self.model, self.utils = load_silero_vad()
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = self.utils

        # 내부 버퍼 (짧은 구간 판단용)
        self._recent_audio = np.array([], dtype=np.float32)
        self._recent_max_sec = 1.0  # 최근 1초만 유지

    def is_speech(self, pcm: np.ndarray) -> bool:
        """
        입력 PCM chunk가 음성인지 여부 반환

        Args:
            pcm: np.ndarray (float32, mono)

        Returns:
            bool: 음성으로 판단되면 True
        """

        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32)

        # 최근 오디오 버퍼에 추가
        self._recent_audio = np.concatenate([self._recent_audio, pcm])

        # 최근 N초만 유지
        max_len = int(self.sample_rate * self._recent_max_sec)
        if self._recent_audio.size > max_len:
            self._recent_audio = self._recent_audio[-max_len:]

        if self._recent_audio.size < int(self.sample_rate * 0.2):
            # 너무 짧으면 판단하지 않음
            return False

        # Torch tensor 변환
        audio_tensor = torch.from_numpy(self._recent_audio)

        # Silero VAD 실행
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            self.model,
            sampling_rate=self.sample_rate,
            threshold=self.speech_threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
        )

        return len(speech_timestamps) > 0