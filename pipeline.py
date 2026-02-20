"""
pipeline.py
───────────
실시간 ASR 파이프라인 핵심 클래스.

외부 모듈(LLM, TTS, UI, DB 등)과의 연결은 콜백으로 처리합니다.
파이프라인 자체는 오디오 처리에만 집중하며,
결과를 어떻게 활용할지는 호출자가 결정합니다.

스레드 구성:
    1. sounddevice 콜백  → audio_q (raw 48kHz PCM 적재)
    2. 처리 스레드       → 노이즈 제거 → VAD → 버퍼 관리 → 추론 → 콜백 호출
"""

import logging
from collections import deque
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import torch

from audio_utils import StreamingDenoiser, build_resampler
from config import PipelineConfig
from models import load_all_models

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 콜백 페이로드
# ──────────────────────────────────────────────────────────────
@dataclass
class TranscriptionResult:
    """
    전사 완료 시 콜백으로 전달되는 데이터 클래스.
    LLM, TTS, UI, DB 등 하위 모듈은 이 객체를 받아 처리합니다.
    """
    text: str               # 전사된 텍스트
    duration_sec: float     # 해당 발화 구간 길이(초)
    language: str           # 감지된 언어 코드


# ──────────────────────────────────────────────────────────────
# 파이프라인
# ──────────────────────────────────────────────────────────────
class RealtimeASRPipeline:
    """
    실시간 ASR 파이프라인.

    사용 예::

        def on_result(result: TranscriptionResult):
            print(result.text)  # → LLM / TTS / DB 연결 지점

        pipeline = RealtimeASRPipeline(
            cfg=PipelineConfig(),
            on_transcription=on_result,
        )
        pipeline.start()
        ...
        pipeline.stop()
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        on_transcription: Optional[Callable[[TranscriptionResult], None]] = None,
        is_tts_speaking: Optional[threading.Event] = None,
    ):
        """
        Args:
            cfg              : PipelineConfig 인스턴스
            on_transcription : 전사 완료 시 호출되는 콜백 함수.
                               None이면 logger.info로만 출력합니다.
            is_tts_speaking  : TTS 재생 중이면 set() 상태인 Event.
                               set 동안 오디오 콜백이 프레임을 폐기합니다.
                               None이면 항상 정상 처리합니다.
        """
        self.cfg = cfg
        self._on_transcription = on_transcription or self._default_on_transcription

        # 외부에서 주입된 TTS 재생 상태 플래그
        # set()  → TTS 재생 중: 오디오 콜백에서 프레임 즉시 폐기
        # clear() → TTS 종료: 정상 처리 재개
        self._is_tts_speaking: Optional[threading.Event] = is_tts_speaking

        # ── 모델 로드 ──────────────────────────────────────────
        models = load_all_models(cfg)
        self._device: str = models["device"]

        # ── 오디오 처리 컴포넌트 ───────────────────────────────
        self._denoiser = StreamingDenoiser(
            df_model=models["df_model"],
            df_state=models["df_state"],
            audio_cfg=cfg.audio,
            df_cfg=cfg.deep_filter,
        )
        self._resampler = build_resampler(cfg.audio)
        self._vad_model = models["vad_model"]
        self._whisper_model = models["whisper_model"]

        # VAD 내부 상태 초기화
        self._vad_model.reset_states()

        # ── 버퍼 / 상태 ────────────────────────────────────────
        self._audio_q: queue.Queue = queue.Queue()
        self._speech_buffer_16k: list[np.ndarray] = []
        self._pre_speech_buffer: deque = deque(maxlen=10)
        self._silence_frames: int = 0
        self._silence_trigger_frames: int = cfg.vad.silence_trigger_frames
        self._max_buffer_frames: int = int(
            cfg.whisper.max_buffer_sec * 1000 / cfg.audio.chunk_duration_ms
        )

        # ── 스레드 제어 ────────────────────────────────────────
        self._running = False
        self._proc_thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None

    # ──────────────────────────────────────────────────────────
    # 기본 콜백 (on_transcription 미지정 시)
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _default_on_transcription(result: TranscriptionResult) -> None:
        logger.info("[결과] %s  (%.1f초, lang=%s)", result.text, result.duration_sec, result.language)

    # ──────────────────────────────────────────────────────────
    # sounddevice 오디오 콜백
    # ──────────────────────────────────────────────────────────
    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        """
        raw PCM을 큐에 넣고 즉시 반환합니다 (언더런 방지).

        TTS 재생 중(_is_tts_speaking이 set 상태)에는 프레임을 큐에 넣지 않고
        즉시 폐기합니다. 이렇게 하면:
          - audio_q 가 쌓이지 않아 input overflow 가 발생하지 않습니다.
          - _process_loop 과 Whisper 가 TTS 음성을 처리하지 않습니다.
          - TTS 종료 후 flush 할 잔여 데이터도 최소화됩니다.
        """
        if status:
            # TTS 재생 중 overflow 는 의도된 폐기의 부작용이므로 debug 로 낮춥니다
            if self._is_tts_speaking and self._is_tts_speaking.is_set():
                logger.debug("[오디오 콜백] TTS 재생 중 상태 경고(무시): %s", status)
            else:
                logger.warning("[오디오 콜백] 상태 경고: %s", status)

        # TTS 재생 중이면 프레임 폐기 (무한루프 방지 + overflow 방지)
        if self._is_tts_speaking and self._is_tts_speaking.is_set():
            return

        mono = indata[:, 0].astype(np.float32)
        self._audio_q.put(mono.copy())

    # ──────────────────────────────────────────────────────────
    # VAD
    # ──────────────────────────────────────────────────────────
    def _is_speech(self, chunk_16k: np.ndarray) -> bool:
        """16kHz 청크에 대해 VAD 확률을 계산하고 임계값과 비교합니다."""
        tensor = torch.from_numpy(chunk_16k).to(self._device).unsqueeze(0)
        with torch.no_grad():
            prob = self._vad_model(tensor, self.cfg.audio.whisper_sample_rate).item()
        return prob >= self.cfg.vad.threshold

    # ──────────────────────────────────────────────────────────
    # Faster-Whisper 추론
    # ──────────────────────────────────────────────────────────
    def _transcribe_and_emit(self) -> None:
        """
        버퍼에 쌓인 16kHz 오디오를 전사하고 콜백을 호출합니다.
        최소 발화 길이보다 짧으면 버퍼를 비우고 그냥 반환합니다.
        """
        if not self._speech_buffer_16k:
            return

        audio_np = np.concatenate(self._speech_buffer_16k, axis=0)
        self._speech_buffer_16k.clear()

        duration = len(audio_np) / self.cfg.audio.whisper_sample_rate

        # 너무 짧은 발화는 무시 (Whisper 오인식 방지)
        if duration < self.cfg.vad.min_speech_duration_sec:
            logger.debug("짧은 발화 무시 (%.2f초 < %.2f초)", duration, self.cfg.vad.min_speech_duration_sec)
            return

        logger.debug("Whisper 추론 시작 (%.1f초 분량)", duration)

        try:
            segments, info = self._whisper_model.transcribe(
                audio_np,
                beam_size=self.cfg.whisper.beam_size,
                language=self.cfg.whisper.language,
                task=self.cfg.whisper.task,
                vad_filter=self.cfg.whisper.vad_filter,
            )

            text = "".join(seg.text for seg in segments).strip()
            if not text:
                return

            result = TranscriptionResult(
                text=text,
                duration_sec=duration,
                language=info.language,
            )
            # ★ 외부 모듈 연결 지점: LLM / TTS / UI / DB 등은 여기서 받습니다
            self._on_transcription(result)

        except Exception:
            logger.exception("Whisper 추론 중 오류 발생")

    # ──────────────────────────────────────────────────────────
    # 처리 스레드 메인 루프
    # ──────────────────────────────────────────────────────────
    def _process_loop(self) -> None:
        logger.info("처리 스레드 시작")

        while self._running:
            # 큐에서 청크 꺼내기 (타임아웃으로 종료 신호 감지)
            try:
                chunk_48k = self._audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            # 1. 노이즈 제거 (48kHz, 프레임 간 문맥 유지)
            clean_48k = self._denoiser.process(chunk_48k)

            # 2. 48kHz → 16kHz 리샘플링
            clean_t = torch.from_numpy(clean_48k).unsqueeze(0)  # (1, T)
            clean_16k = self._resampler(clean_t).squeeze(0).numpy()

            # 3. VAD 판단
            speech_detected = self._is_speech(clean_16k)

            if speech_detected:
                if not self._speech_buffer_16k and self._pre_speech_buffer:
                    self._speech_buffer_16k.extend(self._pre_speech_buffer)
                    self._pre_speech_buffer.clear()

                self._speech_buffer_16k.append(clean_16k)
                self._silence_frames = 0
            else:
                self._pre_speech_buffer.append(clean_16k)

                if self._speech_buffer_16k:
                    self._silence_frames += 1

                    # 4a. 정적 트리거
                    if self._silence_frames >= self._silence_trigger_frames:
                        self._transcribe_and_emit()
                        self._reset_vad_state()

            # 4b. 최대 버퍼 트리거 (문장이 끝나지 않을 경우 강제 추론)
            if len(self._speech_buffer_16k) >= self._max_buffer_frames:
                logger.warning("최대 버퍼 도달 → 강제 추론")
                self._transcribe_and_emit()
                self._reset_vad_state()

        logger.info("처리 스레드 종료")

    # ──────────────────────────────────────────────────────────
    # VAD 상태 초기화 헬퍼
    # ──────────────────────────────────────────────────────────
    def _reset_vad_state(self) -> None:
        self._silence_frames = 0
        self._vad_model.reset_states()

    # ──────────────────────────────────────────────────────────
    # 공개 인터페이스
    # ──────────────────────────────────────────────────────────
    def start(self) -> None:
        """파이프라인을 시작합니다."""
        if self._running:
            logger.warning("이미 실행 중입니다.")
            return

        self._running = True

        self._proc_thread = threading.Thread(
            target=self._process_loop,
            name="asr-process",
            daemon=True,
        )
        self._proc_thread.start()

        self._stream = sd.InputStream(
            samplerate=self.cfg.audio.mic_sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.cfg.audio.chunk_samples_48k,
            device=self.cfg.audio.input_device,
            callback=self._audio_callback,
        )
        self._stream.start()

        logger.info(
            "마이크 스트림 시작 (청크=%dms, SR=%dHz, device=%s)",
            self.cfg.audio.chunk_duration_ms,
            self.cfg.audio.mic_sample_rate,
            self.cfg.audio.input_device or "기본값",
        )

    def stop(self) -> None:
        """파이프라인을 안전하게 종료합니다."""
        self._running = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._proc_thread:
            self._proc_thread.join(timeout=3.0)
            self._proc_thread = None

        # 잔여 버퍼 처리
        if self._speech_buffer_16k:
            logger.info("잔여 버퍼 처리 중...")
            self._transcribe_and_emit()

        logger.info("파이프라인 종료 완료")
