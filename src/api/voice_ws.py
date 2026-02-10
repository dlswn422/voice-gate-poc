from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

import numpy as np
import time
import json
import asyncio

import src.app_state as app_state
from src.speech.whisper_service import transcribe_pcm_chunks
from src.speech.tts import synthesize

router = APIRouter()

# ==================================================
# 🎧 Outdoor Parking Lot Voice Tuning (FINAL - STABLE)
# ==================================================
# 🎯 기준 환경
# - 실외 주차장 키오스크
# - 차량 엔진음, 바람, 주변 대화 존재
# - 마이크에 AGC / Noise Suppression / Echo Cancellation ON

# --------------------------------------------------
# ▶ 무음 판단 RMS 기준
# --------------------------------------------------
SILENCE_RMS_THRESHOLD = 0.0030

# --------------------------------------------------
# ▶ 발화 종료 판단 침묵 시간 (초)
# --------------------------------------------------
END_SILENCE_SEC = 0.55

# --------------------------------------------------
# ▶ 발화 중 잠깐 멈췄을 때 STT pre-run 시작 시점
# (현재 로직에서는 적극 사용 안 함: 유지)
# --------------------------------------------------
PRERUN_SILENCE_SEC = 0.3

# --------------------------------------------------
# ▶ 최소 음성 길이 (초)
# --------------------------------------------------
MIN_AUDIO_SEC = 0.35

# --------------------------------------------------
# ▶ STT pre-run 시 뒤쪽 잡음 컷 길이
# (현재 로직에서는 적극 사용 안 함: 유지)
# --------------------------------------------------
CUT_AUDIO_SEC = 0.2

# ▶ 샘플레이트 (Whisper 기준)
SAMPLE_RATE = 16000

# --------------------------------------------------
# ▶ 발화 시작 인정 프레임 수
# --------------------------------------------------
MIN_SPEECH_FRAMES = 2

# --------------------------------------------------
# ▶ TTS 종료 직후 입력 무시 시간
# --------------------------------------------------
IGNORE_INPUT_AFTER_TTS_SEC = 0.1

# --------------------------------------------------
# ▶ 최대 발화 허용 시간
# --------------------------------------------------
MAX_SPEECH_SEC = 6.0

# ▶ 발화 종료 직후 짧은 무시 구간 (현재 미사용: 유지)
POST_SPEECH_IGNORE_SEC = 0.25

# --------------------------------------------------
# ▶ 무음 정책 (세션 자동 종료)
# --------------------------------------------------
NO_INPUT_WARN_SEC = 12.0
NO_INPUT_END_SEC = 25.0


# ==================================================
# 🔒 WebSocket 안전 유틸 (중복 close/send 방지)
# ==================================================
async def safe_send(ws: WebSocket, payload: dict):
    try:
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_json(payload)
    except Exception:
        pass


async def safe_close(ws: WebSocket):
    try:
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.close()
    except Exception:
        pass


# ==================================================
# 🧠 의미 없는 발화 필터 (2차 방어선)
# ==================================================
def is_meaningful_text(text: str) -> bool:
    if not text:
        return False

    t = text.strip()

    if len(t) < 3:
        return False

    meaningless = {
        "어", "음", "아", "네", "예",
        "어어", "음음", "응", "어?", "음?"
    }
    return t not in meaningless


# ==================================================
# 🎤 Voice WebSocket 엔드포인트
# ==================================================
@router.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket):
    await websocket.accept()
    print("[WS] 🔌 Client connected")

    # ▶ 서버 기준 IO 상태
    io_state = "LISTENING"

    # ==================================================
    # 🔑 음성 동작 모드
    # ==================================================
    # NORMAL        : 일반 대화
    # PAYMENT       : 결제 팝업 중 (음성 완전 무시)
    # POST_PAYMENT  : 결제 완료 후 업로드 대기 (음성 세션 종료 상태)
    voice_mode = "NORMAL"

    # ▶ 출차 컨텍스트
    exit_context = "NONE"  # NONE | UNPAID | PAID

    pcm_buffer: list[np.ndarray] = []
    collecting = False

    speech_start_ts = 0.0
    last_non_silence_ts = 0.0
    ignore_until_ts = 0.0

    speech_frame_count = 0
    prerun_task: asyncio.Task | None = None

    last_activity_ts = time.time()
    no_input_warned = False

    meaningless_count = 0
    MAX_MEANINGLESS_COUNT = 3

    try:
        while True:
            now = time.time()

            # ==================================================
            # 🛑 POST_PAYMENT 상태 → 음성 엔진 완전 정지
            # ==================================================
            if voice_mode == "POST_PAYMENT":
                message = await websocket.receive()

                # ✅ disconnect 처리 (중요)
                if message.get("type") == "websocket.disconnect":
                    print("[WS] 🔌 disconnect received (POST_PAYMENT)")
                    break

                # 프론트 제어 메시지만 수신
                if "text" in message:
                    try:
                        msg = json.loads(message["text"])
                        if msg.get("type") == "voice_mode":
                            voice_mode = msg.get("value", "NORMAL")
                            print(f"[WS] 🎛 voice_mode = {voice_mode}")
                    except Exception:
                        pass
                continue

            # --------------------------------------------------
            # 🕒 무음 경고 / 종료 처리
            # --------------------------------------------------
            if io_state == "LISTENING" and not collecting:
                idle = now - last_activity_ts

                if idle >= NO_INPUT_END_SEC:
                    print("[WS] ⛔ No input timeout → END")
                    msg = "안내를 종료할게요."
                    await safe_send(websocket, {
                        "type": "assistant_message",
                        "text": msg,
                        "tts_url": synthesize(msg),
                        "end_session": True,
                    })
                    await safe_close(websocket)
                    break

                if idle >= NO_INPUT_WARN_SEC and not no_input_warned:
                    no_input_warned = True
                    print("[WS] ⚠️ No input → WARNING")
                    msg = "말씀이 없으시면 안내를 종료할게요."
                    await safe_send(websocket, {
                        "type": "assistant_message",
                        "text": msg,
                        "tts_url": synthesize(msg),
                        "end_session": False,
                    })

            # ==================================================
            # 📩 수신
            # ==================================================
            message = await websocket.receive()

            # ✅ disconnect 처리 (중요)
            if message.get("type") == "websocket.disconnect":
                print("[WS] 🔌 disconnect received")
                break

            # --------------------------------------------------
            # 🔁 프론트 → 제어 메시지
            # --------------------------------------------------
            if "text" in message:
                try:
                    msg = json.loads(message["text"])

                    # ▶ 음성 모드 변경
                    if msg.get("type") == "voice_mode":
                        voice_mode = msg.get("value", "NORMAL")
                        print(f"[WS] 🎛 voice_mode = {voice_mode}")
                        continue

                    # ▶ 출차 컨텍스트
                    if msg.get("type") == "exit_context":
                        exit_context = msg.get("value", "NONE")
                        print(f"[WS] 🚦 exit_context = {exit_context}")
                        continue

                    # ▶ TTS 종료 알림
                    if msg.get("type") == "tts_end":
                        io_state = "LISTENING"
                        collecting = False
                        pcm_buffer.clear()
                        prerun_task = None
                        speech_frame_count = 0
                        ignore_until_ts = time.time() + IGNORE_INPUT_AFTER_TTS_SEC

                        # 활동 갱신 (TTS 종료도 세션 활동으로 간주)
                        last_activity_ts = time.time()
                        no_input_warned = False

                        await safe_send(websocket, {
                            "type": "assistant_state",
                            "state": "LISTENING",
                        })
                        continue
                except Exception:
                    pass

            # --------------------------------------------------
            # 🔒 PAYMENT 모드 → 음성 완전 무시
            # --------------------------------------------------
            if voice_mode == "PAYMENT":
                continue

            # --------------------------------------------------
            # 🎧 오디오 프레임 수신
            # --------------------------------------------------
            if "bytes" not in message or io_state != "LISTENING":
                continue

            if now < ignore_until_ts:
                continue

            pcm = np.frombuffer(message["bytes"], dtype=np.float32)
            if pcm.size == 0:
                continue

            rms = float(np.sqrt(np.mean(pcm * pcm)))

            # --------------------------------------------------
            # 🎤 발화 시작 감지
            # --------------------------------------------------
            if not collecting:
                if rms > SILENCE_RMS_THRESHOLD:
                    speech_frame_count += 1
                else:
                    speech_frame_count = 0

                if speech_frame_count >= MIN_SPEECH_FRAMES:
                    collecting = True
                    pcm_buffer.clear()
                    prerun_task = None
                    speech_frame_count = 0
                    speech_start_ts = now
                    last_non_silence_ts = now

                    # ✅ 발화 시작도 활동으로 간주 (idle 끊김 방지)
                    last_activity_ts = time.time()
                    no_input_warned = False
                continue

            # --------------------------------------------------
            # 🎙 발화 수집
            # --------------------------------------------------
            pcm_buffer.append(pcm)
            if rms > SILENCE_RMS_THRESHOLD:
                last_non_silence_ts = now

            silence_time = now - last_non_silence_ts
            speech_duration = now - speech_start_ts

            # --------------------------------------------------
            # 🛑 발화 종료
            # --------------------------------------------------
            if silence_time >= END_SILENCE_SEC or speech_duration >= MAX_SPEECH_SEC:
                collecting = False

                total_samples = sum(len(c) for c in pcm_buffer)
                if total_samples / SAMPLE_RATE < MIN_AUDIO_SEC:
                    pcm_buffer.clear()
                    prerun_task = None

                    # ✅ 사용자가 말은 했으니 활동 갱신
                    last_activity_ts = time.time()
                    no_input_warned = False
                    continue

                io_state = "THINKING"
                await safe_send(websocket, {
                    "type": "assistant_state",
                    "state": "THINKING",
                })

                # ✅ 말한 직후 활동 갱신 (idle 끊김 방지)
                last_activity_ts = time.time()
                no_input_warned = False

                # --------------------------------------------------
                # 🧠 STT (PERF 로그)
                # --------------------------------------------------
                t0 = time.time()
                text = transcribe_pcm_chunks(
                    pcm_buffer,
                    whisper_model=app_state.whisper_model,
                )
                stt_dt = time.time() - t0
                pcm_buffer.clear()
                print(f"[PERF] STT took {stt_dt:.2f}s, text='{text}'")

                if not is_meaningful_text(text):
                    io_state = "LISTENING"
                    continue

                # --------------------------------------------------
                # 🧠 AppEngine (PERF 로그)
                # --------------------------------------------------
                t1 = time.time()
                result = app_state.app_engine.handle_text(text)
                engine_dt = time.time() - t1
                intent = result.get("intent")
                print(f"[PERF] ENGINE took {engine_dt:.2f}s, intent={intent}")

                # ▶ 출차 미결제 방어
                if exit_context == "UNPAID" and intent == "EXIT":
                    msg = "출차를 위해서는 결제가 필요해요."
                    await safe_send(websocket, {
                        "type": "assistant_message",
                        "text": msg,
                        "tts_url": synthesize(msg),
                        "end_session": False,
                    })
                    io_state = "LISTENING"
                    continue

                reply_text = result.get("text")
                if reply_text:
                    io_state = "SPEAKING"
                    await safe_send(websocket, {
                        **result,
                        "tts_url": synthesize(reply_text),
                    })
                else:
                    io_state = "LISTENING"

    except WebSocketDisconnect:
        print("[WS] ❌ Client disconnected")
    except Exception as e:
        print("[WS] 💥 Error:", repr(e))
        await safe_close(websocket)
