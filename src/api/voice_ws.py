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
SILENCE_RMS_THRESHOLD = 0.0045

# --------------------------------------------------
# ▶ 발화 종료 판단 침묵 시간 (초)
# --------------------------------------------------
END_SILENCE_SEC = 0.25

# --------------------------------------------------
# ▶ 발화 중 잠깐 멈췄을 때 STT pre-run 시작 시점
# --------------------------------------------------
PRERUN_SILENCE_SEC = 0.3

# --------------------------------------------------
# ▶ 최소 음성 길이 (초)
# --------------------------------------------------
MIN_AUDIO_SEC = 0.5

# --------------------------------------------------
# ▶ STT pre-run 시 뒤쪽 잡음 컷 길이
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
IGNORE_INPUT_AFTER_TTS_SEC = 0.2

# --------------------------------------------------
# ▶ 최대 발화 허용 시간
# --------------------------------------------------
MAX_SPEECH_SEC = 4.0

# ▶ 발화 종료 직후 짧은 무시 구간
POST_SPEECH_IGNORE_SEC = 0.25

# --------------------------------------------------
# ▶ 무음 정책 (세션 자동 종료)
# --------------------------------------------------
NO_INPUT_WARN_SEC = 5.0
NO_INPUT_END_SEC = 9.0


# ==================================================
# 🔒 WebSocket 안전 유틸
# ==================================================
async def safe_send(ws: WebSocket, payload: dict):
    if ws.application_state == WebSocketState.CONNECTED:
        await ws.send_json(payload)


async def safe_close(ws: WebSocket):
    if ws.application_state == WebSocketState.CONNECTED:
        await ws.close()


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

    pcm_buffer: list[np.ndarray] = []
    collecting = False

    speech_start_ts = 0.0
    last_non_silence_ts = 0.0
    ignore_until_ts = 0.0

    speech_frame_count = 0
    prerun_task: asyncio.Task | None = None

    last_activity_ts = time.time()
    no_input_warned = False

    # --------------------------------------------------
    # ADD ▶ 의미 없는 발화 연속 카운트
    # --------------------------------------------------
    meaningless_count = 0
    MAX_MEANINGLESS_COUNT = 3  # 의미 없는 발화 허용 횟수 (튜닝 포인트)

    try:
        while True:
            now = time.time()

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

            message = await websocket.receive()

            # --------------------------------------------------
            # 🔁 프론트 → TTS 종료 알림
            # --------------------------------------------------
            if "text" in message:
                try:
                    msg = json.loads(message["text"])
                    if msg.get("type") == "tts_end":
                        print("[WS] 🔁 TTS ended → LISTENING")
                        io_state = "LISTENING"
                        collecting = False
                        pcm_buffer.clear()
                        prerun_task = None
                        speech_frame_count = 0
                        ignore_until_ts = time.time() + IGNORE_INPUT_AFTER_TTS_SEC
                        last_activity_ts = time.time()

                        await safe_send(websocket, {
                            "type": "assistant_state",
                            "state": "LISTENING",
                        })
                        continue
                except Exception:
                    pass

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
                    print("[WS] 🎤 Speech started")
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
            # ⚡ STT pre-run
            # --------------------------------------------------
            if prerun_task is None and silence_time >= PRERUN_SILENCE_SEC:
                audio = np.concatenate(pcm_buffer).astype(np.float32)
                cut = int(SAMPLE_RATE * CUT_AUDIO_SEC)
                if audio.size > cut:
                    audio = audio[:-cut]

                prerun_task = asyncio.create_task(
                    asyncio.to_thread(
                        transcribe_pcm_chunks,
                        [audio],
                        app_state.whisper_model,
                    )
                )
                print("[WS] ⚡ STT pre-run")

            # --------------------------------------------------
            # 🛑 발화 종료
            # --------------------------------------------------
            if silence_time >= END_SILENCE_SEC or speech_duration >= MAX_SPEECH_SEC:
                collecting = False

                total_samples = sum(len(c) for c in pcm_buffer)
                if total_samples / SAMPLE_RATE < MIN_AUDIO_SEC:
                    print("[WS] ❌ Too short → ignored")
                    pcm_buffer.clear()
                    prerun_task = None
                    last_activity_ts = time.time()

                    await safe_send(websocket, {
                        "type": "assistant_state",
                        "state": "LISTENING",
                    })
                    continue

                print("[WS] 🛑 Speech ended → THINKING")
                io_state = "THINKING"
                ignore_until_ts = time.time() + POST_SPEECH_IGNORE_SEC

                await safe_send(websocket, {
                    "type": "assistant_state",
                    "state": "THINKING",
                })

                if prerun_task:
                    try:
                        text = await prerun_task
                    except Exception:
                        text = ""
                else:
                    text = transcribe_pcm_chunks(
                        pcm_buffer,
                        whisper_model=app_state.whisper_model,
                    )

                pcm_buffer.clear()
                prerun_task = None

                # --------------------------------------------------
                # ❌ 의미 없는 발화 처리
                # --------------------------------------------------
                if not is_meaningful_text(text):
                    meaningless_count += 1
                    last_activity_ts = time.time()
                    print(f"[WS] ❌ Meaningless speech count = {meaningless_count}")

                    if meaningless_count >= MAX_MEANINGLESS_COUNT:
                        msg = "음성이 잘 인식되지 않아 안내를 종료할게요."
                        await safe_send(websocket, {
                            "type": "assistant_message",
                            "text": msg,
                            "tts_url": synthesize(msg),
                            "end_session": True,
                        })
                        await safe_close(websocket)
                        break

                    io_state = "LISTENING"
                    await safe_send(websocket, {
                        "type": "assistant_state",
                        "state": "LISTENING",
                    })
                    continue

                # --------------------------------------------------
                # ✅ 의미 있는 발화 → 카운터 리셋
                # --------------------------------------------------
                meaningless_count = 0
                print("[WS] ✅ Meaningful speech → reset meaningless_count")

                print(f"[STT] {text}")
                last_activity_ts = time.time()
                no_input_warned = False

                # --------------------------------------------------
                # 🧠 AppEngine
                # --------------------------------------------------
                result = app_state.app_engine.handle_text(text)

                reply_text = result.get("text", "")
                end_session = result.get("end_session", False)

                if reply_text:
                    io_state = "SPEAKING"
                    tts_url = synthesize(reply_text)
                    payload = dict(result)
                    payload["tts_url"] = tts_url
                    await safe_send(websocket, payload)
                else:
                    io_state = "LISTENING"
                    await safe_send(websocket, {
                        "type": "assistant_state",
                        "state": "LISTENING",
                    })

                if end_session:
                    print("[WS] 🛑 Session ended by engine")
                    await safe_close(websocket)
                    break

    except WebSocketDisconnect:
        print("[WS] ❌ Client disconnected")

    except Exception as e:
        print("[WS] 💥 Error:", repr(e))
        await safe_close(websocket)
