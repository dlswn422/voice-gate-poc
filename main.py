"""
main.py  ─  통합 진입점 (v2)
════════════════════════════════════════════════════════════════
전체 파이프라인:

  [마이크]
     │ 48kHz PCM
     ▼
  DeepFilterNet 노이즈 제거  ← audio_utils.py
     │
     ▼
  Silero VAD → 발화 구간 추출  ← pipeline.py
     │
     ▼
  Faster-Whisper STT  ← models.py
     │  TranscriptionResult.text
     ▼
  ┌──────────────── on_transcription() ─────────────────
  │                                                      
  │  [Step 1] LLaMA classify()   → intent               
  │                ↓                                    
  │  [Step 2] dispatcher.dispatch() → raw DB data        
  │                ↓                                     
  │  [Step 3] LLaMA generate_reply_stream()              
  │           stream=True → 문장 단위로 TTS 큐 투입       
  └──────────────────────────────────────────────────────
                │
                ▼
  TTS Worker Thread (Queue 순차 재생)
                │
                ▼
  🔊 스피커 출력

스레드 구성:
  ① ASR 처리 스레드 (pipeline.py 내부)  : 오디오 → STT
  ② async 이벤트 루프 스레드             : LLaMA 비동기 호출
  ③ TTS Worker 스레드                    : TTS 큐 순차 재생
  ④ 메인 스레드                          : CLI 시뮬레이터 (input 루프)

동시성 제어:
  - TTS 큐(tts_queue)로 오디오 경합 방지 → 항상 순서 보장
  - SessionManager 로 CURRENT_PLATE 스레드 안전 업데이트
  - asyncio.run_coroutine_threadsafe 로 동기/비동기 브리지
════════════════════════════════════════════════════════════════
"""

import asyncio
import logging
import queue
import re
import sys
import importlib.util
import importlib.machinery
import threading
import time
import warnings
from types import ModuleType

import io

import numpy as np
import soundfile as sf
import torch
import sounddevice as sd
from supabase import create_client, Client

from config import PipelineConfig
from pipeline import RealtimeASRPipeline, TranscriptionResult
from intent import classify, generate_reply_stream
from dispatcher import dispatch

try:
    from events import entry_event, exit_event
    _HAS_EVENTS = True
except ImportError:
    _HAS_EVENTS = False
    entry_event = exit_event = None

# ──────────────────────────────────────────────────────────────
# 경고 억제 (huggingface_hub FutureWarning 등)
# ──────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ──────────────────────────────────────────────────────────────
# MeCab / eunjeon 패치 (MeloTTS 내부 의존성 속이기)
# ──────────────────────────────────────────────────────────────
def _apply_mecab_patch() -> bool:
    """
    MeloTTS 가 내부적으로 eunjeon / MeCab 을 import 하려 할 때
    실제 설치된 mecab 바인딩으로 리다이렉트합니다.
    패치 실패 시 False 를 반환하지만 시스템은 계속 기동합니다.
    """
    try:
        mecab_mod = None
        for name in ("mecab", "MeCab", "mecab_ko"):
            try:
                mecab_mod = importlib.import_module(name)
                break
            except ImportError:
                continue
        if not mecab_mod:
            return False

        MeCabClass = getattr(mecab_mod, "MeCab", getattr(mecab_mod, "Mecab", None))

        fake_eunjeon = ModuleType("eunjeon")
        fake_eunjeon.Mecab = MeCabClass
        fake_eunjeon.__spec__ = importlib.machinery.ModuleSpec("eunjeon", None)
        sys.modules["eunjeon"] = fake_eunjeon

        fake_mecab_jp = ModuleType("MeCab")
        fake_mecab_jp.MeCab   = MeCabClass
        fake_mecab_jp.Tagger  = MeCabClass
        fake_mecab_jp.__spec__ = importlib.machinery.ModuleSpec("MeCab", None)
        sys.modules["MeCab"] = fake_mecab_jp

        orig_find_spec = importlib.util.find_spec
        def _patched_find_spec(name, package=None):
            if name in ("eunjeon", "MeCab"):
                return importlib.machinery.ModuleSpec(name, None)
            return orig_find_spec(name, package)
        importlib.util.find_spec = _patched_find_spec
        return True
    except Exception:
        return False


_apply_mecab_patch()


# ══════════════════════════════════════════════════════════════
# KoreanMeloTTS: MeloTTS 기반 한국어 TTS 엔진
# ══════════════════════════════════════════════════════════════
def _kor_number(num_str: str) -> str:
    """
    숫자 문자열(쉼표 포함)을 한국어 읽기 표현으로 변환합니다.
    예) "3,000" → "삼천"
    """
    clean = num_str.replace(",", "")
    if not clean.isdigit():
        return num_str

    units     = ["", "십", "백", "천"]
    big_units = ["", "만", "억", "조"]
    digits    = "0일이삼사오육칠팔구"

    num = int(clean)
    if num == 0:
        return "영"

    result, chunk_count = "", 0
    while num > 0:
        chunk, chunk_str = num % 10000, ""
        for i in range(4):
            d = chunk % 10
            if d > 0:
                prefix = digits[d] if not (d == 1 and i > 0) else ""
                chunk_str = prefix + units[i] + chunk_str
            chunk //= 10
        if chunk_str:
            result = chunk_str + big_units[chunk_count] + " " + result
        num //= 10000
        chunk_count += 1
    return result.strip()


class KoreanMeloTTS:
    """
    MeloTTS KR 모델을 래핑한 한국어 TTS 엔진.

    speak(text) 하나만 외부에서 호출합니다.
    숫자는 자동으로 한국어 읽기로 변환한 뒤 합성합니다.

    speed 파라미터:
      - 1.0  : 기본 속도
      - 1.05 : 약간 빠름 (안내방송 권장)
      - 1.3  : 빠른 안내 (짧은 멘트에 적합)
    """

    # 합성 속도 기본값 — 필요 시 인스턴스 생성 후 변경 가능
    DEFAULT_SPEED: float = 1.05

    def __init__(self, speed: float = DEFAULT_SPEED):
        from melo.api import TTS as MeloTTS          # 지연 임포트 (로딩 시간 격리)
        self.speed  = speed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = MeloTTS(language="KR", device=self.device)
        self._spk   = self._model.hps.data.spk2id["KR"]

    def speak(self, text: str) -> None:
        """
        텍스트를 음성으로 합성하고 즉시 재생합니다 (동기, 블로킹).
        TTS Worker 스레드에서만 호출되므로 동시성 문제 없습니다.
        """
        # 숫자(쉼표 포함) → 한국어 읽기 변환
        processed = re.sub(r"[\d,]+", lambda m: _kor_number(m.group()), text)
        processed = processed.replace("  ", " ").strip()

        # 메모리 버퍼로 합성 후 바로 재생 (디스크 저장 없음)
        buf = io.BytesIO()
        self._model.tts_to_file(processed, self._spk, buf, speed=self.speed, format="wav")
        buf.seek(0)
        audio_np, samplerate = sf.read(buf, dtype="float32")
        sd.play(audio_np, samplerate=samplerate)
        sd.wait()


# TTS 엔진 싱글톤 (main() 에서 초기화)
_tts_engine: KoreanMeloTTS | None = None


# ──────────────────────────────────────────────────────────────
# 로깅
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Supabase 클라이언트 (싱글톤)
# ══════════════════════════════════════════════════════════════
SUPABASE_URL = "https://hiuwgianxzqukemkjsxm.supabase.co"
SUPABASE_KEY = "sb_publishable_iQMpJQ084nk1BUvLT-DUEg_JOOkKHjX"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ══════════════════════════════════════════════════════════════
# 세션 매니저: 현재 응대 차량 번호 스레드 안전 관리
# ══════════════════════════════════════════════════════════════
class SessionManager:
    """
    현재 응대 중인 차량 번호를 스레드 안전하게 관리합니다.
    입/출차 이벤트 발생 시 동적으로 업데이트됩니다.
    """

    def __init__(self, default_plate: str = "미등록"):
        self._lock = threading.Lock()
        self._plate = default_plate

    @property
    def plate(self) -> str:
        with self._lock:
            return self._plate

    @plate.setter
    def plate(self, value: str) -> None:
        with self._lock:
            old = self._plate
            self._plate = value
        logger.info("[Session] 차량 번호 업데이트: %s → %s", old, value)

    def clear(self) -> None:
        self.plate = "미등록"


session = SessionManager(default_plate="12가3456")  # 초기 테스트용


# ══════════════════════════════════════════════════════════════
# TTS Worker Thread + Queue
# ══════════════════════════════════════════════════════════════
tts_queue: queue.Queue = queue.Queue()
_TTS_SENTINEL = object()  # Worker 종료 신호용 센티넬

# TTS 재생 중 STT 수집 차단용 플래그
# set()  → TTS 재생 중: pipeline._audio_callback 에서 프레임 폐기
# clear() → TTS 종료: 정상 처리 재개
is_tts_speaking: threading.Event = threading.Event()


def _tts_worker() -> None:
    """
    TTS 큐에서 메시지를 하나씩 꺼내 KoreanMeloTTS 로 순차 재생합니다.
    LLaMA 응답과 입/출차 멘트가 동시에 들어와도 순서가 보장됩니다.
    재생 구간 동안 is_tts_speaking 을 set() 하여 ASR 수집을 차단합니다.
    """
    logger.info("[TTS Worker] 시작")
    while True:
        item = tts_queue.get()
        if item is _TTS_SENTINEL:
            tts_queue.task_done()
            break
        try:
            is_tts_speaking.set()          # ← TTS 재생 시작: STT 차단
            if _tts_engine is not None:
                _tts_engine.speak(item)
            else:
                logger.warning("[TTS Worker] 엔진 미초기화 — 텍스트: %s", item)
        except Exception as exc:
            logger.error("[TTS Worker] 재생 실패: %s", exc)
        finally:
            is_tts_speaking.clear()        # ← TTS 재생 종료: STT 재개
            tts_queue.task_done()
    logger.info("[TTS Worker] 종료")


def tts_say(message: str) -> None:
    """TTS 큐에 메시지를 투입합니다 (논블로킹, 스레드 안전)."""
    msg = message.strip()
    if msg:
        tts_queue.put(msg)


# ══════════════════════════════════════════════════════════════
# asyncio 이벤트 루프 (백그라운드 전용 스레드)
# classify(), generate_reply_stream() 은 async 이므로 별도 루프 필요
# ══════════════════════════════════════════════════════════════
_async_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()


def _start_async_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


_async_thread = threading.Thread(
    target=_start_async_loop,
    args=(_async_loop,),
    name="async-loop",
    daemon=True,
)
_async_thread.start()


def _run_async(coro) -> object:
    """코루틴을 백그라운드 루프에서 실행하고 결과를 반환합니다 (동기 브리지)."""
    future = asyncio.run_coroutine_threadsafe(coro, _async_loop)
    return future.result(timeout=35.0)  # 콜드 스타트(모델 로딩) 포함 여유값


# ══════════════════════════════════════════════════════════════
# ★ 핵심 콜백: STT → Step1(분류) → Step2(DB) → Step3(응답생성) → TTS
# ══════════════════════════════════════════════════════════════
def on_transcription(result: TranscriptionResult) -> None:
    """
    pipeline.py 가 발화 전사를 완료하면 자동으로 호출됩니다.

    3-Step 파이프라인:
      Step 1 | classify()             : LLaMA → intent (JSON)
      Step 2 | dispatch()             : DB → raw 데이터 조회
      Step 3 | generate_reply_stream(): LLaMA → 자연어 멘트 스트리밍 생성
    """
    stt_text = result.text.strip()
    if not stt_text:
        return

    plate = session.plate
    logger.info("══ [STT] '%s'  (%.1f초) | 차량: %s", stt_text, result.duration_sec, plate)
    t_total = time.perf_counter()

    # ── Step 1: LLaMA 분류 (JSON 모드, Greedy, MAX_TOKENS=100) ─
    try:
        clf = _run_async(classify(stt_text))
    except Exception as exc:
        logger.error("[Step1] 실패: %s", exc)
        tts_say("죄송합니다, 잠시 시스템 오류가 발생했습니다.")
        return

    logger.info(
        "[Step1 완료] intent=%-10s  (%.0fms)",
        clf.intent, clf.latency_ms,
    )


    # ── none: DB·LLaMA Step3 모두 생략, 폴백 멘트 즉시 출력 ──
    if clf.intent == "none":
        tts_say("잘 못 들었습니다. 다시 말씀해 주시겠습니까?")
        return
    # ── Step 2: DB 조회 ───────────────────────────────────────
    try:
        db_result = dispatch(
            supabase=supabase,
            plate_number=plate,
            intent=clf.intent,
        )
    except Exception as exc:
        logger.error("[Step2] 실패: %s", exc)
        tts_say("데이터 조회 중 오류가 발생했습니다.")
        return

    # 에스컬레이션: Step3 생략하고 고정 멘트 즉시 출력
    if db_result.get("escalate"):
        tts_say(
            "고객님, 불편을 드려 죄송합니다. "
            "현재 담당 관리자를 즉시 호출했습니다. 잠시만 기다려 주십시오."
        )
        _notify_admin(plate, clf.intent, db_result["raw_data"].get("reason", ""))
        return

    # ── Step 3: LLaMA 자연어 응답 생성 (stream=True) ─────────
    # 문장 구분자(. ! ?)를 기준으로 청크를 묶어 TTS 큐에 투입합니다.
    # → 생성 중인 동안 앞 문장부터 즉시 재생되므로 체감 지연이 최소화됩니다.
    try:
        async def _stream_to_tts() -> str:
            buffer = ""
            full_reply = ""
            sentence_enders = {".", "!", "?", "。", "！", "？"}

            async for chunk in generate_reply_stream(stt_text, db_result["raw_data"]):
                buffer += chunk
                full_reply += chunk
                # 문장 단위 감지 → 즉시 TTS 큐 투입
                if buffer and buffer[-1] in sentence_enders:
                    tts_say(buffer.strip())
                    buffer = ""

            # 마지막 잔여 버퍼
            if buffer.strip():
                tts_say(buffer.strip())

            return full_reply

        reply_text = _run_async(_stream_to_tts())

    except Exception as exc:
        logger.error("[Step3] 실패: %s", exc)
        tts_say("안내 생성 중 오류가 발생했습니다.")
        return

    total_ms = (time.perf_counter() - t_total) * 1000
    logger.info("[Pipeline 완료] %.0fms | '%s'", total_ms, reply_text[:80])


# ══════════════════════════════════════════════════════════════
# 입/출차 이벤트 처리 (LLaMA 미경유)
# ══════════════════════════════════════════════════════════════
def process_event(plate_number: str, event_type: str) -> None:
    """LPR 센서 트리거 → DB → 고정 멘트 → TTS 큐."""
    if not _HAS_EVENTS:
        logger.warning("events 모듈 없음 — 이벤트 처리 불가")
        return

    if event_type == "entry":
        result = entry_event.handle_entry_event(supabase, plate_number)
    elif event_type == "exit":
        result = exit_event.handle_exit_event(supabase, plate_number)
    else:
        tts_say("알 수 없는 이벤트입니다.")
        return

    if result.get("status") == "success":
        tts_say(result.get("tts_message", ""))
    else:
        logger.error("[Event] 실패: %s", result.get("message"))
        tts_say("처리 중 오류가 발생했습니다. 관리자를 호출해 드릴까요?")


# ══════════════════════════════════════════════════════════════
# 관리자 알림
# ══════════════════════════════════════════════════════════════
def _notify_admin(plate: str, intent: str, reason: str) -> None:
    logger.warning("🚨 [관리자 호출] plate=%s | intent=%s | reason=%s", plate, intent, reason)
    # 실제 구현: Push 알림 / 내부 메시지 큐 등
    # admin_push_queue.put({"plate": plate, "intent": intent, "ts": time.time()})


# ══════════════════════════════════════════════════════════════
# CLI (메인 스레드에서 블로킹)
# 입/출차 센서 이벤트(1, 2)와 상태 조회(i, 0)만 담당합니다.
# 음성 Intent 처리는 실제 마이크(ASR 파이프라인)를 통해서만 이루어집니다.
# ══════════════════════════════════════════════════════════════
_MENU = """\
╔══════════════════════════════════════════
║        🅿️  주차 AI 시스템  운영 콘솔       
╠══════════════════════════════════════════
║   1. 입차 처리 (차량번호 입력)             
║   2. 출차 처리 (현재 세션 차량)            
║   3. 세션 차량 변경 (차량번호만 교체)      
║                                          
║   i. 현재 세션 정보                       
║   0. 종료                                
╚══════════════════════════════════════════"""


def _cli_loop(pipeline: RealtimeASRPipeline) -> None:
    """
    입/출차 이벤트 전용 입력 루프 (메인 스레드).
    ASR 파이프라인은 이미 백그라운드 스레드에서 실시간으로 동작 중입니다.
    """
    while True:
        print(_MENU)
        choice = input(f"\n👉 선택 [현재 차량: {session.plate}]: ").strip()

        if choice == "1":
            plate = input("   차량번호 입력 (예: 12가3456): ").strip()
            if plate:
                session.plate = plate
                process_event(plate, "entry")

        elif choice == "2":
            plate = session.plate
            if plate == "미등록":
                print("   ⚠️  등록된 차량 없음. 입차 먼저 처리하세요.")
            else:
                process_event(plate, "exit")
                session.clear()

        elif choice == "3":
            plate = input("   변경할 차량번호 입력 (예: 12가3456): ").strip()
            if plate:
                session.plate = plate
                print(f"   ✅ 세션 차량이 '{plate}'(으)로 변경되었습니다.")
            else:
                print("   ⚠️  차량번호를 입력하지 않았습니다.")

        elif choice.lower() == "i":
            print(f"\n   📋 현재 세션 차량 : {session.plate}")
            print(f"   📋 TTS 큐 대기수  : {tts_queue.qsize()}건")

        elif choice == "0":
            print("\n시스템을 종료합니다.")
            break

        else:
            print("   잘못된 입력입니다.")

        input("\n   ▶ [Enter] 메뉴로 돌아가기...")


# ══════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════
def main() -> None:
    global _tts_engine

    logger.info("사용 가능한 오디오 입력 장치:\n%s", sd.query_devices())

    cfg = PipelineConfig()
    # cfg.audio.input_device = 2          # 특정 마이크 지정 시 주석 해제
    # cfg.whisper.model_size = "small"    # 경량 모델로 변경 시

    # ── KoreanMeloTTS 엔진 초기화 ─────────────────────────────
    logger.info("🔊 [TTS] MeloTTS KR 모델 로딩 중...")
    try:
        _tts_engine = KoreanMeloTTS(speed=1.3)
        logger.info("✅ [TTS] 모델 로딩 완료 (device=%s)", _tts_engine.device)
    except Exception as exc:
        logger.error("❌ [TTS] 모델 로딩 실패: %s — 텍스트 출력으로 폴백합니다.", exc)
        _tts_engine = None

    # ── TTS Worker 시작 ───────────────────────────────────────
    tts_thread = threading.Thread(
        target=_tts_worker,
        name="tts-worker",
        daemon=True,
    )
    tts_thread.start()

    # ══════════════════════════════════════════════════════════
    # LLM GPU Warm-up (콜드 스타트 방지)
    # ──────────────────────────────────────────────────────────
    # Ollama 는 평소 모델을 VRAM 에서 내려둡니다.
    # 첫 실제 발화 시 모델 로딩(10초 이상)으로 TimeoutError 가 발생하는 것을
    # 방지하기 위해 ASR 파이프라인 시작 전에 더미 요청으로 강제 적재합니다.
    # 예외가 발생해도 전체 시스템 크래시로 이어지지 않도록 처리합니다.
    # ══════════════════════════════════════════════════════════
    logger.info("🔥 [Warm-up] LLM GPU 사전 적재 시작...")
    try:
        _run_async(classify("시스템 예열"))
        logger.info("✅ [Warm-up] LLM 적재 완료. 첫 발화부터 저지연 응답 가능합니다.")
    except Exception as exc:
        # Warm-up 실패는 경고로만 기록 — 시스템은 계속 기동
        logger.warning(
            "⚠️  [Warm-up] LLM 사전 적재 실패 (첫 발화 시 지연 발생 가능): %s", exc
        )

    # ── ASR 파이프라인 시작 (내부 백그라운드 스레드) ─────────
    pipeline = RealtimeASRPipeline(
        cfg=cfg,
        on_transcription=on_transcription,
        is_tts_speaking=is_tts_speaking,
    )

    try:
        pipeline.start()
        logger.info(
            "🎙️  실시간 파이프라인 가동\n"
            "    마이크 → DeepFilter → VAD → Whisper → LLaMA(분류) → DB → LLaMA(응답 생성) → TTS"
        )

        # 메인 스레드 → CLI 시뮬레이터 (블로킹)
        _cli_loop(pipeline)

    except KeyboardInterrupt:
        logger.info("Ctrl+C 수신")
    finally:
        pipeline.stop()
        # TTS 큐 완전 소진 후 Worker 종료
        tts_queue.join()
        tts_queue.put(_TTS_SENTINEL)
        tts_thread.join(timeout=3.0)
        # async 루프 종료
        _async_loop.call_soon_threadsafe(_async_loop.stop)
        logger.info("✅ 모든 리소스 해제 완료")


if __name__ == "__main__":
    main()
