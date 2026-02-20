"""
models.py
─────────
DeepFilterNet / Silero VAD / Faster-Whisper 모델 로드를 담당합니다.
파이프라인 코드와 분리하여 모델 교체/업그레이드가 쉽도록 합니다.
"""

import logging
import os

import torch

from config import PipelineConfig, SystemConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 디바이스 선택
# ──────────────────────────────────────────────────────────────
def get_device() -> str:
    """CUDA 사용 가능 여부를 확인하고 적절한 디바이스 문자열을 반환합니다."""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("GPU 사용: %s", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        logger.info("GPU 없음 → CPU 사용")
    return device


# ──────────────────────────────────────────────────────────────
# Windows CUDA DLL 경로 패치
# ──────────────────────────────────────────────────────────────
def patch_cuda_dll_paths(cfg: SystemConfig) -> None:
    """
    Windows 환경에서 ctranslate2가 nvidia-cublas-cu12 등의 DLL을
    찾지 못하는 문제를 해결합니다.
    다른 OS에서는 아무 동작도 하지 않습니다.
    """
    if not cfg.is_windows:
        return

    dll_subpaths = [
        os.path.join("cublas", "bin"),
        os.path.join("cudnn", "bin"),
        os.path.join("cuda_runtime", "bin"),
    ]

    for sub in dll_subpaths:
        full_path = os.path.join(cfg.nvidia_base_path, sub)
        if os.path.exists(full_path):
            os.add_dll_directory(full_path)
            os.environ["PATH"] += os.pathsep + full_path
            logger.debug("DLL 경로 추가: %s", full_path)
        else:
            logger.warning("DLL 경로 없음 (확인 필요): %s", full_path)


# ──────────────────────────────────────────────────────────────
# DeepFilterNet
# ──────────────────────────────────────────────────────────────
def load_deepfilternet(cfg: PipelineConfig):
    """
    DeepFilterNet 모델을 초기화합니다.

    Returns:
        (model, df_state) 튜플
    """
    from df.enhance import init_df  # 지연 임포트 (설치 선택적)

    model, df_state, _ = init_df()
    model.eval()
    logger.info("DeepFilterNet 로드 완료 (SR=%d)", df_state.sr())
    return model, df_state


# ──────────────────────────────────────────────────────────────
# Silero VAD
# ──────────────────────────────────────────────────────────────
def load_silero_vad(device: str):
    """
    Silero VAD 모델을 로드합니다.

    Returns:
        vad_model (torch.nn.Module)
    """
    vad_model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    vad_model = vad_model.to(device)
    vad_model.eval()
    logger.info("Silero VAD 로드 완료")
    return vad_model


# ──────────────────────────────────────────────────────────────
# Faster-Whisper
# ──────────────────────────────────────────────────────────────
def load_faster_whisper(cfg: PipelineConfig, device: str):
    """
    Faster-Whisper 모델을 로드합니다.
    GPU 환경에서는 float16, CPU 환경에서는 int8 compute type을 사용합니다.

    Returns:
        WhisperModel 인스턴스
    """
    from faster_whisper import WhisperModel  # 지연 임포트

    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(
        cfg.whisper.model_size,
        device=device,
        compute_type=compute_type,
    )
    logger.info(
        "Faster-Whisper '%s' 로드 완료 (%s)",
        cfg.whisper.model_size,
        compute_type,
    )
    return model


# ──────────────────────────────────────────────────────────────
# 편의 함수: 전체 모델 일괄 로드
# ──────────────────────────────────────────────────────────────
def load_all_models(cfg: PipelineConfig) -> dict:
    """
    파이프라인에 필요한 모든 모델을 한 번에 로드합니다.

    Returns:
        {
            "device": str,
            "df_model": ...,
            "df_state": ...,
            "vad_model": ...,
            "whisper_model": ...,
        }
    """
    patch_cuda_dll_paths(cfg.system)
    device = get_device()

    df_model, df_state = load_deepfilternet(cfg)
    vad_model = load_silero_vad(device)
    whisper_model = load_faster_whisper(cfg, device)

    return {
        "device": device,
        "df_model": df_model,
        "df_state": df_state,
        "vad_model": vad_model,
        "whisper_model": whisper_model,
    }
