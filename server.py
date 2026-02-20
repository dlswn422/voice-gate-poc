"""
server.py — FastAPI 래퍼 서버
═══════════════════════════════════════════════════════════════
기존 main.py 의 CLI 터미널 기능을 웹 프론트엔드에서 사용할 수 있도록
FastAPI + WebSocket 으로 래핑합니다.

기존 모듈을 **수정 없이** import 만 하여 재활용합니다:
  - intent.py   : classify(), generate_reply_stream()
  - dispatcher.py : dispatch()
  - events/      : entry_event, exit_event
  - main.py      : KoreanMeloTTS, _kor_number, _apply_mecab_patch
  - pipeline.py  : TranscriptionResult (타입만)
  - models.py    : load_all_models (VAD + Whisper 모델)
  - config.py    : PipelineConfig
  - audio_utils.py : StreamingDenoiser, build_resampler

엔드포인트:
  WS   /ws/voice             브라우저 마이크 PCM → STT → LLaMA → TTS
  POST /api/plate/recognize   차량번호 입력 → 입/출차 처리
  POST /api/payment/demo      결제 시뮬레이션
  GET  /tts/{filename}        TTS wav 파일 서빙
═══════════════════════════════════════════════════════════════
"""

import asyncio
import io
import json
import logging
import re
import struct
import uuid
import warnings
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# FastAPI
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# ── 기존 모듈 임포트 (수정 없이 재활용) ──────────────────────
from config import PipelineConfig
from intent import classify, generate_reply_stream
from dispatcher import dispatch
from events import entry_event, exit_event
from supabase import create_client, Client

# MeCab 패치 (main.py 와 동일)
from main import _apply_mecab_patch, KoreanMeloTTS, _kor_number

# 모델 로드 유틸
from models import load_all_models
from audio_utils import StreamingDenoiser, build_resampler

# ──────────────────────────────────────────────────────────────
# 경고 억제
# ──────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("server")

# ══════════════════════════════════════════════════════════════
# Supabase 클라이언트 (main.py 와 동일한 설정)
# ══════════════════════════════════════════════════════════════
SUPABASE_URL = "https://hiuwgianxzqukemkjsxm.supabase.co"
SUPABASE_KEY = "sb_publishable_iQMpJQ084nk1BUvLT-DUEg_JOOkKHjX"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ══════════════════════════════════════════════════════════════
# TTS 엔진 + 파일 서빙 디렉토리
# ══════════════════════════════════════════════════════════════
TTS_DIR = Path(__file__).parent / "tts_cache"
TTS_DIR.mkdir(exist_ok=True)

_tts_engine: Optional[KoreanMeloTTS] = None

def _tts_to_file(text: str) -> Optional[str]:
    """텍스트를 wav 파일로 합성하고 파일명을 반환합니다."""
    if _tts_engine is None:
        logger.warning("[TTS] 엔진 미초기화 — 텍스트만 반환: %s", text)
        return None

    # 숫자 → 한국어 읽기 변환 (main.py 와 동일)
    processed = re.sub(r"[\d,]+", lambda m: _kor_number(m.group()), text)
    processed = processed.replace("  ", " ").strip()

    filename = f"{uuid.uuid4().hex}.wav"
    filepath = TTS_DIR / filename

    buf = io.BytesIO()
    _tts_engine._model.tts_to_file(
        processed, _tts_engine._spk, buf,
        speed=_tts_engine.speed, format="wav"
    )
    buf.seek(0)

    with open(filepath, "wb") as f:
        f.write(buf.read())

    return filename


# (asyncio 이벤트 루프는 FastAPI가 제공하므로 별도 스레드 불요 — await 직접 호출)


# ══════════════════════════════════════════════════════════════
# VAD + Whisper 모델 로딩 (서버 기동 시 1회)
# ══════════════════════════════════════════════════════════════
_cfg = PipelineConfig()
_models = None
_denoiser = None
_resampler = None


def _init_models():
    """모델 초기화 (최초 1회)"""
    global _models, _denoiser, _resampler
    logger.info("🔧 모델 로딩 시작...")
    _models = load_all_models(_cfg)
    _denoiser = StreamingDenoiser(
        df_model=_models["df_model"],
        df_state=_models["df_state"],
        audio_cfg=_cfg.audio,
        df_cfg=_cfg.deep_filter,
    )
    _resampler = build_resampler(_cfg.audio)
    logger.info("✅ 모델 로딩 완료")


# ══════════════════════════════════════════════════════════════
# FastAPI 앱
# ══════════════════════════════════════════════════════════════
app = FastAPI(title="ParkMate API Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    global _tts_engine
    # TTS 엔진 초기화
    logger.info("🔊 [TTS] MeloTTS KR 모델 로딩 중...")
    try:
        _tts_engine = KoreanMeloTTS(speed=1.3)
        logger.info("✅ [TTS] 모델 로딩 완료 (device=%s)", _tts_engine.device)
    except Exception as exc:
        logger.error("❌ [TTS] 모델 로딩 실패: %s", exc)
        _tts_engine = None

    # VAD + Whisper 모델 초기화
    _init_models()

    # LLM Warm-up
    logger.info("🔥 [Warm-up] LLM GPU 사전 적재 시작...")
    try:
        await classify("시스템 예열")
        logger.info("✅ [Warm-up] LLM 적재 완료")
    except Exception as exc:
        logger.warning("⚠️ [Warm-up] 실패: %s", exc)


# ──────────────────────────────────────────────────────────────
# GET /tts/{filename} — TTS wav 파일 서빙
# ──────────────────────────────────────────────────────────────
@app.get("/tts/{filename}")
async def serve_tts(filename: str):
    filepath = TTS_DIR / filename
    if not filepath.exists():
        return JSONResponse({"error": "file not found"}, status_code=404)
    return FileResponse(str(filepath), media_type="audio/wav")


# ──────────────────────────────────────────────────────────────
# POST /api/plate/recognize — 입/출차 처리
#   → CLI 메뉴 1(입차), 2(출차) 와 동일
# ──────────────────────────────────────────────────────────────
@app.post("/api/plate/recognize")
async def plate_recognize(
    plate_number: str = Form(...),
    direction: str = Form("ENTRY"),  # "ENTRY" 또는 "EXIT"
):
    """
    차량번호와 방향(ENTRY/EXIT)을 받아 입/출차 처리합니다.
    이미지 업로드 대신 텍스트 직접 입력 (CLI 와 동일).
    """
    plate = plate_number.strip()
    if not plate:
        return JSONResponse({"success": False, "message": "차량번호를 입력해주세요."})

    try:
        if direction.upper() == "ENTRY":
            result = entry_event.handle_entry_event(supabase, plate)
        elif direction.upper() == "EXIT":
            result = exit_event.handle_exit_event(supabase, plate)
        else:
            return JSONResponse({"success": False, "message": "direction은 ENTRY 또는 EXIT이어야 합니다."})

        if result.get("status") == "success":
            tts_msg = result.get("tts_message", "")
            tts_filename = _tts_to_file(tts_msg) if tts_msg else None

            return JSONResponse({
                "success": True,
                "plate": plate,
                "direction": direction.upper(),
                "message": tts_msg,
                "tts_url": f"/tts/{tts_filename}" if tts_filename else None,
                "data": result.get("data"),
                "parking_session_id": result.get("data", {}).get("session_id") if direction.upper() == "ENTRY" else None,
            })
        else:
            return JSONResponse({
                "success": False,
                "message": result.get("message", result.get("tts_message", "처리 실패")),
            })

    except Exception as e:
        logger.error("[plate_recognize] 오류: %s", e)
        return JSONResponse({"success": False, "message": str(e)})


# ──────────────────────────────────────────────────────────────
# POST /api/payment/demo — 결제 시뮬레이션
# ──────────────────────────────────────────────────────────────
@app.post("/api/payment/demo")
async def payment_demo(body: dict):
    """
    결제 시뮬레이션: v1_payment_log 테이블에 기록합니다.
    body: { parking_session_id, result: "SUCCESS"|"FAIL", reason?: str }
    """
    session_id = body.get("parking_session_id")
    result = body.get("result", "SUCCESS")
    reason = body.get("reason")

    if not session_id:
        return JSONResponse({"success": False, "detail": "parking_session_id 필요"}, status_code=400)

    try:
        log_data = {
            "session_id": session_id,
            "status": result,
            "err_msg": reason if result == "FAIL" else None,
            "paid_at": datetime.now(timezone.utc).isoformat(),
        }

        supabase.table("v1_payment_log").insert(log_data).execute()

        return JSONResponse({"success": True, "result": result})

    except Exception as e:
        logger.error("[payment_demo] 오류: %s", e)
        return JSONResponse({"success": False, "detail": str(e)}, status_code=500)


# ══════════════════════════════════════════════════════════════
# WebSocket /ws/voice — 브라우저 마이크 → STT → LLaMA → TTS
# ══════════════════════════════════════════════════════════════
@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    """
    브라우저에서 float32 PCM (16kHz) 오디오를 받아
    VAD → Whisper STT → classify → dispatch → generate_reply_stream → TTS
    파이프라인을 실행합니다.

    프로토콜 (클라이언트 → 서버):
      - binary: float32 PCM 오디오 버퍼
      - text/JSON:
        { type: "tts_end" }           → TTS 재생 완료 알림
        { type: "voice_mode", value } → PAYMENT/NORMAL 모드 전환
        { type: "vehicle_result", direction, ... } → 입/출차 결과 처리
        { type: "user_activity" }     → 무음 타이머 리셋

    프로토콜 (서버 → 클라이언트):
      { type: "assistant_state", state: "LISTENING"|"THINKING"|"SPEAKING" }
      { type: "assistant_message", text, intent?, tts_url?, end_session? }
    """
    await ws.accept()
    logger.info("[WS] 클라이언트 연결")

    # ── 세션 상태 ──
    voice_mode = "NORMAL"  # NORMAL | PAYMENT
    plate_number = "미등록"
    is_speaking = False
    barge_in_speech_count = 0  # barge-in 연속 음성 프레임 카운터

    # ── VAD 상태 (세션별 독립) ──
    speech_buffer: list[np.ndarray] = []
    pre_speech_buffer: deque = deque(maxlen=10)
    silence_frames = 0
    silence_trigger = _cfg.vad.silence_trigger_frames
    max_buffer_frames = int(_cfg.whisper.max_buffer_sec * 1000 / _cfg.audio.chunk_duration_ms)

    # VAD 모델은 상태를 가지므로, 세션별로 reset
    if _models:
        _models["vad_model"].reset_states()

    # 연결 즉시 LISTENING 상태 알림
    await ws.send_json({"type": "assistant_state", "state": "LISTENING"})

    async def send_state(state: str):
        try:
            await ws.send_json({"type": "assistant_state", "state": state})
        except Exception:
            pass

    async def send_message(text: str, intent: Optional[str] = None, tts_url: Optional[str] = None, end_session: bool = False):
        try:
            msg: dict = {"type": "assistant_message", "text": text}
            if intent:
                msg["intent"] = intent
            if tts_url:
                msg["tts_url"] = tts_url
            if end_session:
                msg["end_session"] = True
            await ws.send_json(msg)
        except Exception:
            pass

    def transcribe(audio_16k: np.ndarray) -> Optional[str]:
        """Whisper 전사 (동기, 블로킹)"""
        if _models is None:
            return None
        duration = len(audio_16k) / _cfg.audio.whisper_sample_rate
        if duration < _cfg.vad.min_speech_duration_sec:
            return None

        try:
            segments, info = _models["whisper_model"].transcribe(
                audio_16k,
                beam_size=_cfg.whisper.beam_size,
                language=_cfg.whisper.language,
                task=_cfg.whisper.task,
                vad_filter=_cfg.whisper.vad_filter,
            )
            text = "".join(seg.text for seg in segments).strip()
            return text if text else None
        except Exception as exc:
            logger.error("[WS STT] 오류: %s", exc)
            return None

    async def process_stt_text(stt_text: str):
        """STT → classify → dispatch → generate_reply_stream → 문장 단위 TTS 스트리밍"""
        nonlocal is_speaking, plate_number

        await send_state("THINKING")

        # Step 1: classify (await 직접 호출 — 스레드 브릿지 제거)
        try:
            clf = await classify(stt_text)
        except Exception as exc:
            logger.error("[Step1] 실패: %s", exc)
            tts_file = _tts_to_file("죄송합니다, 잠시 시스템 오류가 발생했습니다.")
            await send_message(
                "죄송합니다, 잠시 시스템 오류가 발생했습니다.",
                tts_url=f"/tts/{tts_file}" if tts_file else None,
            )
            return

        logger.info("[Step1] intent=%s (%.0fms)", clf.intent, clf.latency_ms)

        if clf.intent == "none":
            msg = "잘 못 들었습니다. 다시 말씀해 주시겠습니까?"
            tts_file = _tts_to_file(msg)
            is_speaking = True
            await send_state("SPEAKING")
            await send_message(msg, tts_url=f"/tts/{tts_file}" if tts_file else None)
            return

        # Step 2: dispatch (동기 함수 — 빠름)
        try:
            db_result = dispatch(supabase=supabase, plate_number=plate_number, intent=clf.intent)
        except Exception as exc:
            logger.error("[Step2] 실패: %s", exc)
            msg = "데이터 조회 중 오류가 발생했습니다."
            tts_file = _tts_to_file(msg)
            await send_message(msg, tts_url=f"/tts/{tts_file}" if tts_file else None)
            return

        if db_result.get("escalate"):
            msg = "고객님, 불편을 드려 죄송합니다. 현재 담당 관리자를 즉시 호출했습니다. 잠시만 기다려 주십시오."
            tts_file = _tts_to_file(msg)
            is_speaking = True
            await send_state("SPEAKING")
            await send_message(msg, intent=clf.intent.upper(), tts_url=f"/tts/{tts_file}" if tts_file else None)
            return

        # Step 3: 문장 단위 스트리밍 TTS (첫 문장 완성 즉시 전송)
        try:
            is_speaking = True
            await send_state("SPEAKING")

            sentence_buf = []       # 토큰 누적 버퍼
            full_reply_parts = []   # 전체 응답 텍스트 수집
            first_sent = True

            async for chunk in generate_reply_stream(stt_text, db_result["raw_data"]):
                full_reply_parts.append(chunk)
                sentence_buf.append(chunk)
                joined = "".join(sentence_buf)

                # 문장 구분자(. ! ?) 또는 줄바꿈 감지 시 즉시 TTS + 전송
                if any(joined.rstrip().endswith(c) for c in (".", "!", "?")):
                    sentence = joined.strip()
                    if sentence:
                        tts_file = _tts_to_file(sentence)
                        await send_message(
                            sentence,
                            intent=clf.intent.upper() if first_sent else None,
                            tts_url=f"/tts/{tts_file}" if tts_file else None,
                        )
                        first_sent = False
                    sentence_buf.clear()

            # 잔여 텍스트 처리
            remaining = "".join(sentence_buf).strip()
            if remaining:
                tts_file = _tts_to_file(remaining)
                await send_message(
                    remaining,
                    intent=clf.intent.upper() if first_sent else None,
                    tts_url=f"/tts/{tts_file}" if tts_file else None,
                )

            full_reply = "".join(full_reply_parts).strip()
            logger.info("[Pipeline 완료] '%s'", full_reply[:80])

        except Exception as exc:
            logger.error("[Step3] 실패: %s", exc)
            msg = "안내 생성 중 오류가 발생했습니다."
            tts_file = _tts_to_file(msg)
            await send_message(msg, tts_url=f"/tts/{tts_file}" if tts_file else None)

    def is_speech(chunk_16k: np.ndarray) -> bool:
        """VAD 판단"""
        if _models is None:
            return False
        tensor = torch.from_numpy(chunk_16k).to(_models["device"]).unsqueeze(0)
        with torch.no_grad():
            prob = _models["vad_model"](tensor, _cfg.audio.whisper_sample_rate).item()
        return prob >= _cfg.vad.threshold

    # ── WebSocket 메인 루프 ──
    try:
        while True:
            data = await ws.receive()

            # ── 텍스트 메시지 (JSON) ──
            if "text" in data:
                try:
                    msg = json.loads(data["text"])
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "tts_end":
                    is_speaking = False
                    await send_state("LISTENING")

                elif msg_type == "voice_mode":
                    voice_mode = msg.get("value", "NORMAL")
                    logger.info("[WS] voice_mode → %s", voice_mode)

                elif msg_type == "user_activity":
                    pass  # 무음 타이머 리셋 (현재 미사용)

                elif msg_type == "vehicle_result":
                    # 입/출차 결과 → 안내 멘트 생성
                    direction = msg.get("direction", "")
                    if direction == "ENTRY_DENIED":
                        m = "현재 주차장이 만차입니다. 잠시 후 다시 시도해 주세요."
                        tts_file = _tts_to_file(m)
                        is_speaking = True
                        await send_state("SPEAKING")
                        await send_message(m, tts_url=f"/tts/{tts_file}" if tts_file else None)

                elif msg_type == "payment_result":
                    value = msg.get("value", "")
                    if value == "SUCCESS":
                        m = "결제가 완료되었습니다. 안녕히 가십시오."
                    else:
                        m = "결제에 실패했습니다. 다시 시도해 주시거나 관리실에 문의해 주세요."
                    tts_file = _tts_to_file(m)
                    is_speaking = True
                    await send_state("SPEAKING")
                    await send_message(m, tts_url=f"/tts/{tts_file}" if tts_file else None)

                elif msg_type == "set_plate":
                    plate_number = msg.get("plate", "미등록")
                    logger.info("[WS] plate → %s", plate_number)

                continue

            # ── 바이너리 메시지 (오디오 PCM) ──
            if "bytes" in data and data["bytes"]:
                if voice_mode == "PAYMENT":
                    logger.debug("[WS] 오디오 무시 (voice_mode=%s)", voice_mode)
                    continue

                raw_bytes = data["bytes"]

                # float32 PCM 파싱 (브라우저 AudioContext → 16kHz)
                try:
                    n_samples = len(raw_bytes) // 4
                    audio_chunk = np.array(
                        struct.unpack(f"<{n_samples}f", raw_bytes[:n_samples * 4]),
                        dtype=np.float32,
                    )
                except Exception:
                    continue

                if len(audio_chunk) == 0:
                    continue

                # ── VAD 처리 ──
                # Silero VAD는 16kHz 기준 512 샘플(32ms) 단위로만 정확히 동작.
                # 브라우저 ScriptProcessor(4096샘플)를 512샘플씩 분할 처리.
                VAD_CHUNK = 512
                any_speech = False

                for i in range(0, len(audio_chunk) - VAD_CHUNK + 1, VAD_CHUNK):
                    sub = audio_chunk[i:i + VAD_CHUNK]
                    if is_speech(sub):
                        any_speech = True
                        break

                # ── Barge-in 감지: SPEAKING 중 음성이 감지되면 TTS 중단 ──
                if is_speaking:
                    if any_speech:
                        barge_in_speech_count += 1
                        if barge_in_speech_count >= 2:  # 연속 2프레임 이상 → barge-in 확정
                            logger.info("🔇 [Barge-in] 사용자 음성 감지 → TTS 중단 요청")
                            is_speaking = False
                            barge_in_speech_count = 0
                            # TTS 큐 비우기 위해 클라이언트에 알림
                            try:
                                await ws.send_json({"type": "barge_in"})
                            except Exception:
                                pass
                            # 현재 프레임부터 speech_buffer 수집 시작
                            speech_buffer.clear()
                            silence_frames = 0
                            if _models:
                                _models["vad_model"].reset_states()
                            speech_buffer.append(audio_chunk)
                    else:
                        barge_in_speech_count = 0
                    continue  # SPEAKING 중에는 일반 VAD 로직 스킵

                if any_speech:
                    if not speech_buffer and pre_speech_buffer:
                        speech_buffer.extend(pre_speech_buffer)
                        pre_speech_buffer.clear()
                    speech_buffer.append(audio_chunk)
                    silence_frames = 0
                else:
                    pre_speech_buffer.append(audio_chunk)
                    if speech_buffer:
                        silence_frames += 1

                        # 정적 트리거 → 전사
                        if silence_frames >= silence_trigger:
                            audio_np = np.concatenate(speech_buffer, axis=0)
                            speech_buffer.clear()
                            silence_frames = 0
                            if _models:
                                _models["vad_model"].reset_states()

                            stt_text = transcribe(audio_np)
                            if stt_text:
                                logger.info("[WS STT] '%s'", stt_text)
                                await process_stt_text(stt_text)

                # 최대 버퍼 트리거
                if len(speech_buffer) >= max_buffer_frames:
                    audio_np = np.concatenate(speech_buffer, axis=0)
                    speech_buffer.clear()
                    silence_frames = 0
                    if _models:
                        _models["vad_model"].reset_states()

                    stt_text = transcribe(audio_np)
                    if stt_text:
                        logger.info("[WS STT max] '%s'", stt_text)
                        await process_stt_text(stt_text)

    except WebSocketDisconnect:
        logger.info("[WS] 클라이언트 연결 해제")
    except Exception as exc:
        logger.error("[WS] 오류: %s", exc)


# ══════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
