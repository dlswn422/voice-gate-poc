from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

import numpy as np
import time
import json

import src.app_state as app_state
from src.speech.whisper_service import transcribe_pcm_chunks
from src.speech.tts import synthesize

router = APIRouter()

# ==================================================
# 🎧 Outdoor Parking Lot Voice Tuning
# ==================================================
SILENCE_RMS_THRESHOLD = 0.0030
END_SILENCE_SEC = 0.45
MIN_AUDIO_SEC = 0.35
SAMPLE_RATE = 16000
MIN_SPEECH_FRAMES = 1
IGNORE_INPUT_AFTER_TTS_SEC = 0.2
MAX_SPEECH_SEC = 6.0
NO_INPUT_WARN_SEC = 5.0
NO_INPUT_END_SEC = 9.0


# ==================================================
# Utils
# ==================================================
async def safe_send(ws: WebSocket, payload: dict):
    if ws.application_state == WebSocketState.CONNECTED:
        await ws.send_json(payload)


async def safe_close(ws: WebSocket):
    if ws.application_state == WebSocketState.CONNECTED:
        await ws.close()


def is_meaningful_text(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    if len(t) < 3:
        return False
    return t not in {"어", "음", "아", "네", "예", "응"}


# ==================================================
# WebSocket
# ==================================================
@router.websocket("/ws/voice")
async def voice_session_ws(websocket: WebSocket):
    await websocket.accept()
    print("[WS] 🔌 Voice connected")

    # -----------------------------
    # Session State
    # -----------------------------
    io_state = "LISTENING"          # LISTENING | THINKING | SPEAKING
    voice_mode = "NORMAL"           # NORMAL | PAYMENT
    exit_context = "NONE"           # NONE | UNPAID

    pcm_buffer: list[np.ndarray] = []
    collecting = False
    speech_frame_count = 0
    last_non_silence_ts = 0.0
    ignore_until_ts = 0.0

    last_activity_ts = time.time()
    no_input_warned = False

    try:
        while True:
            now = time.time()

            # ==================================================
            # ⏰ No-input timeout (결제 중 제외)
            # ==================================================
            if (
                io_state == "LISTENING"
                and not collecting
                and now >= ignore_until_ts
                and voice_mode != "PAYMENT"
            ):
                idle = now - last_activity_ts

                if idle >= NO_INPUT_END_SEC:
                    print("[TIMEOUT] ❌ No input → END SESSION")

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
                    print("[TIMEOUT] ⚠️ No input warning")

                    no_input_warned = True
                    io_state = "SPEAKING"
                    last_activity_ts = time.time()

                    msg = "말씀이 없으시면 안내를 종료할게요."
                    await safe_send(websocket, {
                        "type": "assistant_message",
                        "text": msg,
                        "tts_url": synthesize(msg),
                    })

            message = await websocket.receive()

            # ==================================================
            # 📩 Frontend control messages
            # ==================================================
            if "text" in message:
                try:
                    msg = json.loads(message["text"])

                    # ❗❗❗ 추가된 핵심 ❗❗❗
                    if msg.get("type") == "user_activity":
                        print("[ACTIVITY] 🧩 User activity detected")
                        last_activity_ts = time.time()
                        continue

                    if msg.get("type") == "tts_end":
                        print("[TTS END] 🔊 → LISTENING")

                        io_state = "LISTENING"
                        collecting = False
                        pcm_buffer.clear()
                        speech_frame_count = 0
                        ignore_until_ts = time.time() + IGNORE_INPUT_AFTER_TTS_SEC
                        last_activity_ts = time.time()

                        await safe_send(websocket, {
                            "type": "assistant_state",
                            "state": "LISTENING",
                        })
                        continue

                    if msg.get("type") == "voice_mode":
                        voice_mode = msg.get("value", "NORMAL")
                        print(f"[MODE] 🎛 voice_mode = {voice_mode}")
                        continue

                    if msg.get("type") in ("vehicle_result", "payment_result"):
                        io_state = "SPEAKING"
                        last_activity_ts = time.time()
                        voice_mode = "NORMAL"

                        if msg["type"] == "vehicle_result":
                            direction = msg.get("direction")
                            reason = msg.get("reason")
                            exit_context = msg.get("exit_context", "NONE")

                            if direction == "ENTRY_DENIED" and reason == "FULL":
                                text = (
                                    "현재 주차장이 만차입니다.\n"
                                    "근처 주차장을 찾을 수 없습니다."
                                )
                            elif direction == "ENTRY":
                                text = (
                                    "입차가 정상적으로 등록되었습니다.\n"
                                    "문제가 있으면 말씀해주세요."
                                )
                            elif direction == "EXIT":
                                if exit_context == "UNPAID":
                                    text = (
                                        "미결제 상태입니다.\n"
                                        "결제 후 출차가 가능합니다.\n"
                                        "혹시 문제가 있으신가요?"
                                    )
                                else:
                                    text = (
                                        "출차를 진행합니다.\n"
                                        "문제가 있으면 말씀해주세요."
                                    )
                            else:
                                continue

                        else:  # payment_result
                            if msg.get("value") == "SUCCESS":
                                exit_context = "NONE"
                                text = (
                                    "결제가 완료되었습니다.\n"
                                    "차량 번호판을 다시 업로드해 주세요."
                                )
                            else:
                                text = (
                                    "결제에 실패했습니다.\n"
                                    "혹시 문제가 있으신가요?"
                                )

                        await safe_send(websocket, {
                            "type": "assistant_message",
                            "text": text,
                            "tts_url": synthesize(text),
                        })
                        continue

                except Exception as e:
                    print("[ERROR] ❌ Front message parse error:", e)

            # ==================================================
            # 🔒 PAYMENT MODE
            # ==================================================
            if voice_mode == "PAYMENT":
                continue

            # ==================================================
            # 🎧 Audio frame
            # ==================================================
            if "bytes" not in message or io_state != "LISTENING":
                continue

            if now < ignore_until_ts:
                continue

            pcm = np.frombuffer(message["bytes"], dtype=np.float32)
            if pcm.size == 0:
                continue

            rms = float(np.sqrt(np.mean(pcm * pcm)))

            # -----------------------------
            # 🎤 Speech start
            # -----------------------------
            if not collecting:
                if rms > SILENCE_RMS_THRESHOLD:
                    speech_frame_count += 1
                else:
                    speech_frame_count = 0

                if speech_frame_count >= MIN_SPEECH_FRAMES:
                    collecting = True
                    pcm_buffer.clear()
                    speech_frame_count = 0
                    last_non_silence_ts = now
                    print(f"[SPEECH START] 🎤 rms={rms:.4f}")
                continue
            
            # -----------------------------
            # 🎙 Collecting
            # -----------------------------
            pcm_buffer.append(pcm)
            if rms > SILENCE_RMS_THRESHOLD:
                last_non_silence_ts = now

            # -----------------------------
            # 🛑 Speech end
            # -----------------------------
            if now - last_non_silence_ts >= END_SILENCE_SEC:
                collecting = False
                duration = sum(len(c) for c in pcm_buffer) / SAMPLE_RATE
                print(f"[SPEECH END] 🎤 duration={duration:.2f}s")

                if duration < MIN_AUDIO_SEC:
                    pcm_buffer.clear()
                    continue

                # 🔥 THINKING 진입 (UX용 상태 브로드캐스트)
                io_state = "THINKING"
                await safe_send(websocket, {
                    "type": "assistant_state",
                    "state": "THINKING",
                })

                # 🔤 STT
                text = transcribe_pcm_chunks(
                    pcm_buffer,
                    whisper_model=app_state.whisper_model,
                )
                pcm_buffer.clear()

                if not is_meaningful_text(text):
                    io_state = "LISTENING"
                    await safe_send(websocket, {
                        "type": "assistant_state",
                        "state": "LISTENING",
                    })
                    continue

                last_activity_ts = time.time()

                # 🤖 AppEngine 처리
                result = app_state.app_engine.handle_text(text)
                reply = result.get("text")

                if reply:
                    io_state = "SPEAKING"
                    await safe_send(websocket, {
                        **result,
                        "type": "assistant_message",
                        "tts_url": synthesize(reply),
                    })
                else:
                    io_state = "LISTENING"
                    await safe_send(websocket, {
                        "type": "assistant_state",
                        "state": "LISTENING",
                    })

    except WebSocketDisconnect:
        print("[WS] ❌ Voice disconnected")
