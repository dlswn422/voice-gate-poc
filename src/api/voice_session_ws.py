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
SILENCE_RMS_THRESHOLD = 0.0045
END_SILENCE_SEC = 0.25
MIN_AUDIO_SEC = 0.5
SAMPLE_RATE = 16000
MIN_SPEECH_FRAMES = 2
IGNORE_INPUT_AFTER_TTS_SEC = 0.2
MAX_SPEECH_SEC = 4.0
NO_INPUT_WARN_SEC = 5.0
NO_INPUT_END_SEC = 9.0


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


@router.websocket("/ws/voice")
async def voice_session_ws(websocket: WebSocket):
    await websocket.accept()
    print("[WS] 🔌 Voice connected")

    # -----------------------------
    # Session State
    # -----------------------------
    io_state = "LISTENING"       # LISTENING | THINKING | SPEAKING
    voice_mode = "NORMAL"        # NORMAL | PAYMENT
    exit_context = "NONE"        # NONE | UNPAID

    pcm_buffer = []
    collecting = False
    speech_frame_count = 0
    speech_start_ts = 0.0
    last_non_silence_ts = 0.0
    ignore_until_ts = 0.0

    last_activity_ts = time.time()
    no_input_warned = False

    try:
        while True:
            now = time.time()

            # ==================================================
            # ⏰ No-input timeout
            # ==================================================
            if io_state == "LISTENING" and not collecting:
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
                    no_input_warned = True
                    print("[TIMEOUT] ⚠️ No input warning")
                    msg = "말씀이 없으시면 안내를 종료할게요."
                    last_activity_ts = time.time()
                    await safe_send(websocket, {
                        "type": "assistant_message",
                        "text": msg,
                        "tts_url": synthesize(msg),
                    })

            message = await websocket.receive()

            # ==================================================
            # 📩 Frontend Control Messages
            # ==================================================
            if "text" in message:
                try:
                    msg = json.loads(message["text"])

                    # ▶ TTS 종료
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

                    # ▶ 음성 모드
                    if msg.get("type") == "voice_mode":
                        voice_mode = msg.get("value", "NORMAL")
                        print(f"[MODE] 🎛 voice_mode = {voice_mode}")
                        continue

                    # ▶ 🚗 번호판 결과
                    if msg.get("type") == "vehicle_result":
                        direction = msg.get("direction")
                        reason = msg.get("reason")
                        exit_context = msg.get("exit_context", "NONE")

                        if direction == "ENTRY_DENIED" and reason == "FULL":
                            text = (
                                "현재 주차장이 만차입니다.\n"
                                "근처 주차장을 찾을 수 없습니다."
                            )

                            await safe_send(websocket, {
                                "type": "assistant_message",
                                "text": text,
                                "tts_url": synthesize(text),
                                # end_session 없음 → 무음 타임아웃 종료
                            })
                            continue

                        if direction == "ENTRY":
                            text = (
                                "입차가 정상적으로 등록되었습니다.\n"
                                "문제가 있으면 말씀해주세요."
                            )
                            
                        elif direction == "EXIT":
                            if exit_context == "UNPAID":
                                text = (
                                "미결제 상태입니다\n. 결제 후 출차가 가능합니다.\n"
                                "혹시 문제가 있으신가요?"
                            )
                            else:
                                text = (
                                "출차를 진행합니다.\n"
                                "문제가 있으면 말씀해주세요."
                                )

                        await safe_send(websocket, {
                            "type": "assistant_message",
                            "text": text,
                            "tts_url": synthesize(text),
                        })

                    # ▶ 💳 결제 결과
                    if msg.get("type") == "payment_result":
                        result = msg.get("value")

                        if result == "SUCCESS":
                            # ✅ 성공 → 시스템 플로우
                            exit_context = "NONE"
                            text = "결제가 완료되었습니다\n. 출차를 진행하세요."

                            last_activity_ts = time.time()
                            io_state = "SPEAKING"

                            # 🔥 추가: 음성 입력 다시 허용
                            voice_mode = "NORMAL"

                            await safe_send(websocket, {
                                "type": "assistant_message",
                                "text": text,
                                "tts_url": synthesize(text),
                            })
                            continue

                        else:
                            # ❌ 실패 → 상담 플로우
                            text = (
                                "결제에 실패했습니다.\n"
                                "혹시 문제가 있으신가요?"
                            )

                            last_activity_ts = time.time()
                            io_state = "SPEAKING"

                            # 🔥 이미 잘됨
                            voice_mode = "NORMAL"

                            await safe_send(websocket, {
                                "type": "assistant_message",
                                "text": text,
                                "tts_url": synthesize(text),
                            })
                            continue

                except Exception as e:
                    print("[ERROR] ❌ Front message parse error:", e)

            # ==================================================
            # 🔒 PAYMENT MODE → mic ignore
            # ==================================================
            if voice_mode == "PAYMENT":
                continue

            # ==================================================
            # 🎧 Audio Frame
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
            # 🎤 Speech Start
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
                    speech_start_ts = now
                    last_non_silence_ts = now
                    print("[SPEECH START] 🎤")
                continue

            # -----------------------------
            # 🎙 Collecting
            # -----------------------------
            pcm_buffer.append(pcm)
            if rms > SILENCE_RMS_THRESHOLD:
                last_non_silence_ts = now

            # -----------------------------
            # 🛑 Speech End
            # -----------------------------
            if now - last_non_silence_ts >= END_SILENCE_SEC:
                collecting = False
                duration = sum(len(c) for c in pcm_buffer) / SAMPLE_RATE
                print(f"[SPEECH END] 🎤 duration={duration:.2f}s")

                if duration < MIN_AUDIO_SEC:
                    print("[SPEECH DROP] ⛔ Too short")
                    pcm_buffer.clear()
                    continue

                io_state = "THINKING"
                print("[STT] 🧠 Transcribing...")

                text = transcribe_pcm_chunks(
                    pcm_buffer,
                    whisper_model=app_state.whisper_model,
                )
                pcm_buffer.clear()

                print(f"[STT RESULT] 📝 '{text}'")

                if not is_meaningful_text(text):
                    print("[STT IGNORE] 🤷 Meaningless")
                    io_state = "LISTENING"
                    continue

                last_activity_ts = time.time()

                result = app_state.app_engine.handle_text(text)
                reply = result.get("text")

                if reply:
                    io_state = "SPEAKING"
                    print("[TTS] 🗣 assistant reply")
                    await safe_send(websocket, {
                        **result,
                        "type": "assistant_message",
                        "tts_url": synthesize(reply),
                    })
                else:
                    io_state = "LISTENING"

    except WebSocketDisconnect:
        print("[WS] ❌ Voice disconnected")