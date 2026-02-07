import torch
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps


class VoiceActivityDetector:
    """
    Silero VAD 기반 음성 활동 감지기 (스트리밍 최적화 버전)

    ✔ 역할:
        - "지금 사람이 말을 시작했는지?" 만 판단
    ❌ 하지 않는 것:
        - 말이 끝났는지 판단 ❌
        - 스트리밍 중 매 chunk마다 정밀 분석 ❌

    👉 말 종료 판단은 voice_ws.py에서
       RMS + 시간 기준으로 처리하는 구조
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

        # --------------------------------------------------
        # Silero VAD 모델 (JIT, 가볍고 정확)
        # --------------------------------------------------
        self.model = load_silero_vad()
        self.model.to(self.device)

        # 오디오 설정 (고정)
        self.sample_rate = 16000

        # --------------------------------------------------
        # 🔧 튜닝 포인트 (안전한 기본값)
        # --------------------------------------------------
        # 이 값들은 "말이 시작됐다"를 비교적 빠르게
        # 감지하기 위한 설정
        self.min_speech_ms = 200     # 기존 250 → 약간 빠르게
        self.min_silence_ms = 300    # 스트리밍 종료 판단에는 사용 안 함

    def is_speech(self, pcm: np.ndarray) -> bool:
        """
        단일 PCM chunk에 '말 시작 징후'가 있는지 판단

        ✔ True  → 사람이 말을 시작했다고 봄
        ✔ False → 아직 침묵 또는 잡음

        ⚠️ 주의:
        - 이 함수는 collecting 시작 전(=침묵 상태)에서만
          호출되는 것이 정상
        """

        if pcm is None or len(pcm) == 0:
            return False

        # numpy → torch
        # copy()는 non-writable warning 방지용
        audio = torch.from_numpy(pcm.copy()).float().to(self.device)

        # --------------------------------------------------
        # Silero VAD 호출
        # --------------------------------------------------
        # get_speech_timestamps는 원래
        # "긴 오디오 전체 분석" 용도이지만,
        # 여기서는 "말 시작 트리거"로만 사용
        timestamps = get_speech_timestamps(
            audio,
            self.model,
            sampling_rate=self.sample_rate,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
        )

        # 하나라도 잡히면 "말 시작"
        return len(timestamps) > 0
