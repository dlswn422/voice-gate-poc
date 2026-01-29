import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from typing import Optional, Callable, List


class FasterWhisperSTT:
    """
    VAD 기반 실시간 STT 엔진

    개선 포인트:
    - silence_threshold / silence_chunks 조정으로 헛감지/헛전사 감소
    - 너무 짧은 소리(min_speech_seconds)는 STT 실행 안 함
    - initial_prompt로 도메인 단어 힌트 제공(차단기/입차/출차 등)
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device_index: Optional[int] = None,
        sample_rate: int = 16000,
        chunk_seconds: float = 0.5,
        silence_threshold: float = 0.03,   # ✅ 0.015 -> 0.03 (헛감지 줄임)
        silence_chunks: int = 4,           # ✅ 2 -> 4 (너무 빨리 끊기는 문제 완화)
        min_speech_seconds: float = 0.6,   # ✅ 너무 짧은 소리는 STT 실행 안 함
        beam_size: int = 5,                # ✅ CPU에서 무난
        use_whisper_vad: bool = True,      # faster-whisper 내부 vad_filter 사용 여부
        debug_level: int = 0,              # 0: 기본, 1: 볼륨 로그
    ):
        # 오디오 설정
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.silence_threshold = silence_threshold
        self.silence_chunks = silence_chunks
        self.min_speech_seconds = min_speech_seconds
        self.device_index = device_index
        self.beam_size = beam_size
        self.use_whisper_vad = use_whisper_vad
        self.debug_level = debug_level

        # Whisper 모델 로딩
        print("[STT] Loading Faster-Whisper model...")
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            download_root="models",
        )
        print("[STT] Faster-Whisper model loaded")

        # STT 결과 콜백
        self.on_text: Optional[Callable[[str], None]] = None

        # ✅ 도메인 힌트 (오인식 줄이기)
        self.initial_prompt = (
            "주차장 키오스크 음성. 차단기, 입차, 출차, 정산, 결제, 방문자 등록, 번호판, "
            "차단기 올려줘, 차단기 내려줘, 차단기 열어줘, 차단기 안 열려요, 출차가 안 돼요."
        )

    def start_listening(self):
        """
        마이크 입력을 받아 VAD 기반으로 발화를 감지하고
        발화 종료 시 STT를 수행한다.
        """
        print("[STT] Listening started (Ctrl+C to stop)")

        buffer: List[np.ndarray] = []
        silent_count = 0
        is_speaking = False

        try:
            # ✅ InputStream은 “장치 열기 유지” 목적.
            # 실제 수집은 sd.rec(blocking=True)로 chunk 단위로 받음(간단/안정).
            with sd.InputStream(
                samplerate=self.sample_rate,
                device=self.device_index,
                channels=1,
                dtype="float32",
            ):
                while True:
                    data = sd.rec(
                        int(self.chunk_seconds * self.sample_rate),
                        samplerate=self.sample_rate,
                        channels=1,
                        dtype="float32",
                        blocking=True,
                    )

                    audio = data.squeeze()
                    volume = float(np.max(np.abs(audio)))

                    if self.debug_level >= 1:
                        print(f"[STT][DBG] volume={volume:.4f}")

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

                        # ✅ 너무 짧은 발화는 버림 (헛감지/헛전사 줄임)
                        total_samples = sum(len(x) for x in buffer)
                        speech_seconds = total_samples / float(self.sample_rate)

                        if speech_seconds < self.min_speech_seconds:
                            print(
                                f"[STT] Ignored short speech ({speech_seconds:.2f}s < {self.min_speech_seconds:.2f}s)"
                            )
                        else:
                            self._process_buffer(buffer)

                        buffer.clear()
                        silent_count = 0
                        is_speaking = False

        except KeyboardInterrupt:
            print("[STT] Listening stopped")

    def _process_buffer(self, buffer: List[np.ndarray]):
        """
        누적된 오디오 버퍼를 Whisper로 변환한다.
        """
        if not buffer:
            return

        audio = np.concatenate(buffer)

        segments, _ = self.model.transcribe(
            audio,
            language="ko",
            beam_size=self.beam_size,
            vad_filter=self.use_whisper_vad,
            initial_prompt=self.initial_prompt,
        )

        text = "".join(seg.text for seg in segments).strip()

        if not text:
            print("[STT] No transcription result")
            return

        print(f"[STT] Transcribed text: {text}")

        # STT 결과를 상위 로직(AppEngine)으로 전달
        if self.on_text:
            self.on_text(text)
