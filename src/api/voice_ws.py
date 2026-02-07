from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np
import time
import json

import src.app_state as app_state
from src.speech.vad import VoiceActivityDetector
from src.speech.whisper_service import transcribe_pcm_chunks
from src.speech.tts import synthesize

router = APIRouter()

SILENCE_RMS_THRESHOLD = 0.008
END_SILENCE_SEC = 0.7


@router.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket):
    await websocket.accept()
    print("[WS] 🔌 Client connected")

    vad = VoiceActivityDetector()
    pcm_buffer: list[np.ndarray] = []

    collecting = False
    last_non_silence_ts = 0.0

    # 🔥 최초 상태는 반드시 LISTENING
    app_state.app_engine.state = "LISTENING"

    try:
        while True:
            # ==================================================
            # 0️⃣ 메시지 수신 (audio or control)
            # ==================================================
            message = await websocket.receive()

            # ---------- (A) 프론트 제어 메시지 ----------
            if "text" in message:
                try:
                    msg = json.loads(message["text"])
                    if msg.get("type") == "tts_end":
                        # 🔥 여기서 반드시 LISTENING으로 복귀
                        app_state.app_engine.state = "LISTENING"
                        collecting = False
                        pcm_buffer.clear()
                        print("[WS] 🔁 TTS ended → LISTENING")
                        continue
                except Exception:
                    continue

            # ---------- (B) 오디오 프레임 ----------
            if "bytes" not in message:
                continue

            pcm = np.frombuffer(message["bytes"], dtype=np.float32)
            if pcm.size == 0:
                continue

            now = time.time()

            # ==================================================
            # 1️⃣ RMS 계산
            # ==================================================
            rms = np.sqrt(np.mean(pcm * pcm))

            # ==================================================
            # 🔒 2️⃣ 서버 차단 구간
            # ==================================================
            if app_state.app_engine.state in ("SPEAKING", "THINKING"):
                # 🔥 여기서 last_non_silence_ts 갱신 금지
                collecting = False
                pcm_buffer.clear()
                continue

            # ==================================================
            # 3️⃣ 발화 시작 판단
            # ==================================================
            is_speech = vad.is_speech(pcm) or rms > SILENCE_RMS_THRESHOLD

            if is_speech:
                if not collecting:
                    collecting = True
                    pcm_buffer.clear()
                    print("[WS] 🎤 Speech started")

                pcm_buffer.append(pcm)
                last_non_silence_ts = now
                continue

            # ==================================================
            # 4️⃣ 발화 종료 판단
            # ==================================================
            if collecting:
                if now - last_non_silence_ts >= END_SILENCE_SEC:
                    print("[WS] 🛑 Speech ended")
                    collecting = False

                    if not pcm_buffer:
                        continue

                    # ==================================================
                    # 5️⃣ STT
                    # ==================================================
                    text = transcribe_pcm_chunks(
                        pcm_buffer,
                        whisper_model=app_state.whisper_model,
                    )
                    pcm_buffer.clear()

                    if not text:
                        continue

                    print(f"[STT] {text}")

                    # ==================================================
                    # 6️⃣ AppEngine → THINKING
                    # ==================================================
                    app_state.app_engine.state = "THINKING"
                    reply = app_state.app_engine.handle_text(text)
                    print(f"[BOT] {reply}")

                    # ==================================================
                    # 7️⃣ TTS → SPEAKING
                    # ==================================================
                    app_state.app_engine.state = "SPEAKING"
                    tts_url = synthesize(reply)

                    # ==================================================
                    # 8️⃣ 프론트 전송
                    # ==================================================
                    await websocket.send_json(
                        {
                            "type": "bot_text",
                            "text": reply,
                            "tts_url": tts_url,
                        }
                    )

    except WebSocketDisconnect:
        print("[WS] ❌ Client disconnected")

    except Exception as e:
        print("[WS] 💥 Error:", repr(e))
        await websocket.close()