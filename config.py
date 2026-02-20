"""
config.py
─────────
전체 파이프라인의 설정값을 한 곳에서 관리합니다.
외부 모듈(LLM, TTS, UI, DB)을 연결할 때 이 파일만 수정하면 됩니다.
"""

from dataclasses import dataclass, field
from typing import Optional
import platform


# ──────────────────────────────────────────────────────────────
# 오디오 설정
# ──────────────────────────────────────────────────────────────
@dataclass
class AudioConfig:
    # 샘플레이트
    mic_sample_rate: int = 48_000       # 마이크 입력 / DeepFilterNet 처리 SR
    whisper_sample_rate: int = 16_000   # Faster-Whisper 입력 SR

    # 청크 크기 (DeepFilterNet 권장 프레임 단위)
    chunk_duration_ms: int = 32

    # 마이크 입력 장치 인덱스 (None = 시스템 기본값)
    input_device: Optional[int] = None

    @property
    def chunk_samples_48k(self) -> int:
        return int(self.mic_sample_rate * self.chunk_duration_ms / 1000)

    @property
    def chunk_samples_16k(self) -> int:
        return int(self.whisper_sample_rate * self.chunk_duration_ms / 1000)


# ──────────────────────────────────────────────────────────────
# VAD 설정
# ──────────────────────────────────────────────────────────────
@dataclass
class VADConfig:
    threshold: float = 0.7              # 발화 판단 확률 임계값 (높을수록 엄격)
    silence_trigger_sec: float = 0.6    # 이 시간(초) 이상 정적이면 추론 트리거
    min_speech_duration_sec: float = 0.5  # 이보다 짧은 발화는 Whisper에 넘기지 않음

    @property
    def silence_trigger_frames(self) -> int:
        return int(self.silence_trigger_sec * 1000 / AudioConfig().chunk_duration_ms)


# ──────────────────────────────────────────────────────────────
# DeepFilterNet 설정
# ──────────────────────────────────────────────────────────────
@dataclass
class DeepFilterConfig:
    # 프레임 간 문맥 유지를 위한 롤링 버퍼 크기 (청크 단위)
    context_chunks: int = 5


# ──────────────────────────────────────────────────────────────
# Whisper 설정
# ──────────────────────────────────────────────────────────────
@dataclass
class WhisperConfig:
    model_size: str = "medium"    # tiny / base / small / medium / large-v3
    language: str = "ko"
    task: str = "transcribe"
    beam_size: int = 3             # 지연 최소화 (그리디 디코딩에 가까움)
    vad_filter: bool = False        # 외부 VAD로 이미 게이팅하므로 내부 VAD 비활성화
    max_buffer_sec: float = 30.0    # 버퍼 최대 길이(초) 초과 시 강제 추론 트리거


# ──────────────────────────────────────────────────────────────
# CUDA / 시스템 설정
# ──────────────────────────────────────────────────────────────
@dataclass
class SystemConfig:
    # Windows에서 nvidia 패키지 DLL 경로를 PATH에 추가할 기본 경로
    # 본인 환경에 맞게 수정하세요
    nvidia_base_path: str = (
        r"C:\Users\Admin\AppData\Local\Programs\Python\Python311"
        r"\Lib\site-packages\nvidia"
    )

    @property
    def is_windows(self) -> bool:
        return platform.system() == "Windows"


# ──────────────────────────────────────────────────────────────
# 통합 설정 (파이프라인이 이 클래스 하나만 받으면 됨)
# ──────────────────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    deep_filter: DeepFilterConfig = field(default_factory=DeepFilterConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
