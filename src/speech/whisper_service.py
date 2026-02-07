import numpy as np
import soundfile as sf
import tempfile
import os

import src.app_state as app_state


# ==================================================
# 1️⃣ 파일 기반 STT (HTTP API용)
# --------------------------------------------------
# - 긴 음성
# - 실시간성 중요 ❌
# - 안정성 / 정확도 우선
# ==================================================
def transcribe_audio(audio_path: str) -> str:
    """
    파일 경로 기반 Whisper STT

    ✔ 사용처:
        - HTTP /voice API
        - 업로드된 음성 파일
        - 실시간 스트리밍 아님

    👉 기본 옵션 사용
       (속도보다 정확도/안정성 우선)
    """

    segments, _ = app_state.whisper_model.transcribe(
        audio_path,
        language="ko",
    )

    return " ".join(seg.text.strip() for seg in segments if seg.text)


# ==================================================
# 2️⃣ PCM 스트리밍 기반 STT (WebSocket용)
# --------------------------------------------------
# - 짧은 발화
# - 실시간성 중요
# - CPU 환경
# ==================================================
def transcribe_pcm_chunks(
    pcm_chunks: list[np.ndarray],
    whisper_model=None,
    sample_rate: int = 16000,
) -> str:
    """
    PCM(Float32) chunk 리스트를 Whisper로 변환 (스트리밍 최적화)

    ✔ 전제 조건:
        - mono
        - float32
        - 16kHz
        - "말이 끝난 뒤"에만 호출됨 (final STT)

    ✔ 목표:
        - 정확도 유지
        - 변환 시간 최소화
        - 이전 발화에 끌리지 않기
    """

    if not pcm_chunks:
        return ""

    if whisper_model is None:
        whisper_model = app_state.whisper_model

    # ==================================================
    # PCM 병합
    # --------------------------------------------------
    # VAD/WS에서 이미 "하나의 발화" 단위로
    # 잘라서 들어오기 때문에 단순 concat만 함
    # ==================================================
    audio = np.concatenate(pcm_chunks).astype(np.float32)

    if audio.size == 0:
        return ""

    # ==================================================
    # 임시 WAV 파일 생성
    # --------------------------------------------------
    # faster-whisper는 numpy 직접 입력도 가능하지만
    # 파일 입력이:
    #   ✔ 가장 안정적
    #   ✔ 디버깅 쉬움
    #   ✔ 예외 적음
    # ==================================================
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sf.write(tmp_path, audio, samplerate=sample_rate)

        # ==================================================
        # 🔥 Whisper 옵션 (스트리밍 최적화 핵심)
        # ==================================================
        segments, _ = whisper_model.transcribe(
            tmp_path,

            # ----------------------------
            # 언어 고정 (자동 감지 ❌)
            # → 속도 + 정확도 둘 다 ↑
            # ----------------------------
            language="ko",

            # ----------------------------
            # beam search 비활성화
            # beam_size > 1 이면 정확도는
            # 약간 오르지만 CPU 속도 급락
            # ----------------------------
            beam_size=1,
            best_of=1,

            # ----------------------------
            # deterministic decoding
            # 스트리밍/명령형 발화에 안정적
            # ----------------------------
            temperature=0.0,

            # ----------------------------
            # 🔥 매우 중요
            # 이전 발화 문맥을 이어받지 않음
            # (스트리밍에서는 필수)
            # ----------------------------
            condition_on_previous_text=False,

            # ----------------------------
            # Whisper 내부 VAD 비활성화
            # → 외부 VAD에서 이미 처리함
            # ----------------------------
            vad_filter=False,
        )

        # 결과 병합
        text = " ".join(seg.text.strip() for seg in segments if seg.text)
        return text.strip()

    finally:
        # 임시 파일 정리
        try:
            os.remove(tmp_path)
        except OSError:
            pass
