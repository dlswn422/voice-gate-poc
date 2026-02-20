"""
audio_utils.py
──────────────
오디오 처리 관련 유틸리티를 담당합니다.
- 리샘플러 생성
- DeepFilterNet 롤링 버퍼 기반 디노이저
"""

import logging
from collections import deque

import numpy as np
import torch
import torchaudio.transforms as T

from config import AudioConfig, DeepFilterConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 리샘플러
# ──────────────────────────────────────────────────────────────
def build_resampler(
    audio_cfg: AudioConfig,
) -> T.Resample:
    """48kHz → 16kHz 리샘플러를 생성합니다."""
    return T.Resample(
        orig_freq=audio_cfg.mic_sample_rate,
        new_freq=audio_cfg.whisper_sample_rate,
    )


# ──────────────────────────────────────────────────────────────
# DeepFilterNet 스트리밍 디노이저
# ──────────────────────────────────────────────────────────────
class StreamingDenoiser:
    """
    DeepFilterNet을 청크 단위 스트리밍으로 구동하는 래퍼.

    롤링 버퍼(context_chunks × chunk_samples)를 유지하여
    프레임 경계에서도 문맥이 끊기지 않도록 합니다.
    각 호출에서는 버퍼 전체를 enhance에 넘기고,
    처리 결과 중 가장 최근 청크 길이만큼만 잘라 반환합니다.
    """

    def __init__(
        self,
        df_model,
        df_state,
        audio_cfg: AudioConfig,
        df_cfg: DeepFilterConfig,
    ):
        self._model = df_model
        self._state = df_state
        self._chunk_samples = audio_cfg.chunk_samples_48k
        maxlen = self._chunk_samples * df_cfg.context_chunks
        self._buffer: deque = deque(maxlen=maxlen)

    def process(self, chunk_48k: np.ndarray) -> np.ndarray:
        """
        단일 48kHz 청크를 노이즈 제거하여 반환합니다.

        Args:
            chunk_48k: shape (N,) float32, 48kHz mono PCM

        Returns:
            shape (N,) float32, 노이즈 제거된 48kHz mono PCM
        """
        from df.enhance import enhance  # 지연 임포트

        # 롤링 버퍼 업데이트
        self._buffer.extend(chunk_48k)

        buffer_np = np.array(self._buffer, dtype=np.float32)
        audio_t = torch.from_numpy(buffer_np).unsqueeze(0)  # (1, T)

        with torch.no_grad():
            enhanced_t = enhance(self._model, self._state, audio_t, atten_lim_db=10)

        out_np = enhanced_t.squeeze(0).cpu().numpy()

        # 버퍼 전체 중 마지막 청크 길이만 반환
        return out_np[-len(chunk_48k):]
