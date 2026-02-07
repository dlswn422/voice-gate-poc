from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

import numpy as np
import time
import json
import asyncio

import src.app_state as app_state
from src.speech.whisper_service import transcribe_pcm_chunks
from src.speech.tts import synthesize
from src.nlu.dialog_llm_stream import dialog_llm_stream

router = APIRouter()

# ==================================================
# 🔧 튜닝 포인트 (웹 마이크 기준 확정값)
# ==================================================
SILENCE_RMS_THRESHOLD = 0.003
END_SILENCE_SEC = 0.25
PRERUN_SILENCE_SEC = 0.2
MIN_AUDIO_SEC = 0.5
CUT_AUDIO_SEC = 0.2
SAMPLE_RATE = 16000

MIN_SPEECH_FRAMES = 3
IGNORE_INPUT_AFTER_TTS_SEC = 0.35


# ==================================================
# 🔒 WebSocket 안전 유틸
# ==================================================
async def safe_send(ws: WebSocket, payload: dict):
    if ws.application_state == WebSocketState.CONNECTED:
        await ws.send_json(payload)


async def safe_close(ws: WebSocket):
    if ws.application_state == WebSocketState.CONNECTED:
        await ws.close()


@router.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket):
    await websocket.accept()
    print("[WS] 🔌 Client connected")

    pcm_buffer: list[np.ndarray] = []
    collecting = False
    last_non_silence_ts = 0.0

    prerun_task: asyncio.Task | None = None
    speech_frame_count = 0
    ignore_until_ts = 0.0

    app_state.app_engine.state = "LISTENING"

    try:
        while True:
            message = await websocket.receive()

            # --------------------------------------------------
            # 프론트 제어 메시지 (TTS 종료)
            # --------------------------------------------------
            if "text" in message:
                try:
                    msg = json.loads(message["text"])
                    if msg.get("type") == "tts_end":
                        print("[WS] 🔁 TTS ended → LISTENING")
                        app_state.app_engine.state = "LISTENING"
                        collecting = False
                        pcm_buffer.clear()
                        prerun_task = None
                        speech_frame_count = 0
                        ignore_until_ts = time.time() + IGNORE_INPUT_AFTER_TTS_SEC
                        continue
                except Exception:
                    pass

            # --------------------------------------------------
            # 오디오 프레임
            # --------------------------------------------------
            if "bytes" not in message:
                continue

            if time.time() < ignore_until_ts:
                continue

            pcm = np.frombuffer(message["bytes"], dtype=np.float32)
            if pcm.size == 0:
                continue

            now = time.time()
            rms = np.sqrt(np.mean(pcm * pcm))

            # --------------------------------------------------
            # THINKING / SPEAKING 중 입력 무시
            # --------------------------------------------------
            if app_state.app_engine.state in ("THINKING", "SPEAKING"):
                collecting = False
                pcm_buffer.clear()
                prerun_task = None
                speech_frame_count = 0
                continue

            # --------------------------------------------------
            # 🎤 발화 시작 감지 (RMS 기반)
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
                    last_non_silence_ts = now
                    print("[WS] 🎤 Speech started")
                continue

            # --------------------------------------------------
            # 발화 수집 중
            # --------------------------------------------------
            if rms > SILENCE_RMS_THRESHOLD:
                pcm_buffer.append(pcm)
                last_non_silence_ts = now
                continue

            silence_time = now - last_non_silence_ts

            # --------------------------------------------------
            # STT pre-run
            # --------------------------------------------------
            if collecting and prerun_task is None and silence_time >= PRERUN_SILENCE_SEC:
                print("[WS] ⚡ STT pre-run")

                audio = np.concatenate(pcm_buffer).astype(np.float32)
                cut_samples = int(SAMPLE_RATE * CUT_AUDIO_SEC)
                if audio.size > cut_samples:
                    audio = audio[:-cut_samples]

                prerun_task = asyncio.create_task(
                    asyncio.to_thread(
                        transcribe_pcm_chunks,
                        [audio],
                        app_state.whisper_model,
                    )
                )

            # --------------------------------------------------
            # 🛑 발화 종료
            # --------------------------------------------------
            if collecting and silence_time >= END_SILENCE_SEC:
                collecting = False
                print("[WS] 🛑 Speech ended")

                total_samples = sum(len(c) for c in pcm_buffer)
                total_audio_sec = total_samples / SAMPLE_RATE

                if total_audio_sec < MIN_AUDIO_SEC:
                    pcm_buffer.clear()
                    prerun_task = None
                    continue

                if prerun_task:
                    try:
                        text = await prerun_task
                        print("[WS] ⚡ pre-run STT reused")
                    except Exception:
                        text = ""
                else:
                    text = transcribe_pcm_chunks(
                        pcm_buffer,
                        whisper_model=app_state.whisper_model,
                    )

                pcm_buffer.clear()
                prerun_task = None

                if not text:
                    app_state.app_engine.state = "LISTENING"
                    continue

                print(f"[STT] {text}")

                # ==================================================
                # 🧠 Dialog LLM Stream 연결 (핵심)
                # ==================================================
                async for event in dialog_llm_stream(
                    user_text=text,
                    history=app_state.app_engine.dialog_history,
                    context={
                        "session_id": app_state.app_engine.session_id,
                        "intent": app_state.app_engine.first_intent,
                        "turn_count_user": app_state.app_engine.second_turn_count_user,
                        "hard_turn_limit": 6,
                        "slots": app_state.app_engine.second_slots,
                        "pending_slot": app_state.app_engine.second_pending_slot,
                    },
                    debug=True,
                ):
                    # THINKING / MESSAGE / DONE 그대로 전달
                    await safe_send(websocket, event)

                    # MESSAGE → TTS 실행
                    if event.get("type") == "assistant_message":
                        reply_text = event.get("text", "")
                        if reply_text:
                            app_state.app_engine.state = "SPEAKING"
                            tts_url = synthesize(reply_text)
                            await safe_send(websocket, {
                                "type": "tts_start",
                                "tts_url": tts_url,
                            })

                    # DONE → 엔진 상태 동기화
                    if event.get("type") == "assistant_done":
                        app_state.app_engine.second_slots = event.get("slots") or {}
                        app_state.app_engine.second_pending_slot = event.get("pending_slot")
                        app_state.app_engine.first_intent = (
                            event.get("new_intent") or app_state.app_engine.first_intent
                        )
                        break

    except WebSocketDisconnect:
        print("[WS] ❌ Client disconnected")

    except Exception as e:
        print("[WS] 💥 Error:", repr(e))
        await safe_close(websocket)
