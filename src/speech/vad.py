import torch
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps


class VoiceActivityDetector:
    """
    Silero VAD 기반 음성 활동 감지기
    - PCM float32 (16kHz, mono)
    - 발화 시작 / 종료 판단
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

        # silero 모델 (단일 JIT 모델)
        self.model = load_silero_vad()
        self.model.to(self.device)

        # 오디오 설정
        self.sample_rate = 16000

        # VAD 파라미터
        self.min_speech_ms = 250
        self.min_silence_ms = 600

        # 🔥 WebSocket 스트리밍용 종료 기준 (중요)
        self.end_silence_sec = self.min_silence_ms / 1000.0

    def is_speech(self, pcm: np.ndarray) -> bool:
        """
        단일 PCM chunk에 음성이 포함되어 있는지 판단
        """
        if pcm is None or len(pcm) == 0:
            return False

        # ⚠️ non-writable warning 방지
        audio = torch.from_numpy(pcm.copy()).float().to(self.device)

        timestamps = get_speech_timestamps(
            audio,
            self.model,
            sampling_rate=self.sample_rate,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
        )

        return len(timestamps) > 0
