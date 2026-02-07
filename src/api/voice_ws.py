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

    # 🔥 최초 진입 상태는 반드시 LISTENING
    app_state.app_engine.state = "LISTENING"

    try:
        while True:
            # ==================================================
            # 0️⃣ 메시지 수신 (audio frame or control message)
            # ==================================================
            message = await websocket.receive()

            # ---------- (A) 프론트 → 서버 제어 메시지 ----------
            if "text" in message:
                try:
                    msg = json.loads(message["text"])
                    if msg.get("type") == "tts_end":
                        # 🔁 TTS 종료 → 다시 사용자 발화 수신 가능
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
            # 1️⃣ RMS 계산 (침묵 판단 보조)
            # ==================================================
            rms = np.sqrt(np.mean(pcm * pcm))

            # ==================================================
            # 🔒 2️⃣ 서버 차단 구간
            # - THINKING : LLM 응답 생성 중
            # - SPEAKING : TTS 재생 중
            # ==================================================
            if app_state.app_engine.state in ("SPEAKING", "THINKING"):
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
            if collecting and now - last_non_silence_ts >= END_SILENCE_SEC:
                print("[WS] 🛑 Speech ended")
                collecting = False

                if not pcm_buffer:
                    continue

                # 🔥🔥🔥 핵심: 말 끝난 즉시 THINKING 이벤트
                await websocket.send_json({
                    "type": "assistant_state",
                    "state": "THINKING",
                })
                print("[WS] 💭 THINKING sent to client")

                # ==================================================
                # 5️⃣ STT
                # ==================================================
                text = transcribe_pcm_chunks(
                    pcm_buffer,
                    whisper_model=app_state.whisper_model,
                )
                pcm_buffer.clear()

                if not text:
                    app_state.app_engine.state = "LISTENING"
                    continue

                print(f"[STT] {text}")

                # ==================================================
                # 6️⃣ AppEngine → THINKING (실제 처리)
                # ==================================================
                app_state.app_engine.state = "THINKING"
                result = app_state.app_engine.handle_text(text)

                reply_text = result.get("text", "")
                conversation_state = result.get(
                    "conversation_state", "WAITING_USER"
                )
                end_session = result.get("end_session", False)

                print(f"[BOT] {reply_text} ({conversation_state})")

                # ==================================================
                # 7️⃣ TTS → SPEAKING (텍스트 있을 때만)
                # ==================================================
                tts_url = None
                if reply_text:
                    app_state.app_engine.state = "SPEAKING"
                    tts_url = synthesize(reply_text)

                # ==================================================
                # 8️⃣ 프론트로 전송 (응답)
                # ==================================================
                await websocket.send_json({
                    "type": "assistant_message",
                    "text": reply_text,
                    "tts_url": tts_url,
                    "conversation_state": conversation_state,
                    "end_session": end_session,
                })

                # ==================================================
                # 9️⃣ 대화 종료 처리
                # ==================================================
                if end_session:
                    app_state.app_engine.state = "LISTENING"
                    collecting = False
                    pcm_buffer.clear()
                    print("[WS] 🛑 Conversation ended")

    except WebSocketDisconnect:
        print("[WS] ❌ Client disconnected")

    except Exception as e:
        print("[WS] 💥 Error:", repr(e))
        await websocket.close()
