from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np
import time
import json

import src.app_state as app_state
from src.speech.vad import VoiceActivityDetector
from src.speech.whisper_service import transcribe_pcm_chunks
from src.speech.tts import synthesize

router = APIRouter()

# ==================================================
# 🔧 튜닝 포인트 (안전한 기본값)
# ==================================================

# RMS 기준: 배경 소음 vs 실제 발화 구분용
SILENCE_RMS_THRESHOLD = 0.008

# 말이 끝났다고 판단하는 침묵 시간
END_SILENCE_SEC = 0.4

# 너무 짧은 발화는 STT 안 태우기
MIN_AUDIO_SEC = 0.5


@router.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket):
    await websocket.accept()
    print("[WS] 🔌 Client connected")

    # --------------------------------------------------
    # VAD (발화 시작 감지용)
    # --------------------------------------------------
    vad = VoiceActivityDetector()

    pcm_buffer: list[np.ndarray] = []
    collecting = False
    last_non_silence_ts = 0.0

    # 최초 상태는 대기
    app_state.app_engine.state = "LISTENING"

    try:
        while True:
            # ==================================================
            # 0️⃣ 메시지 수신
            # ==================================================
            message = await websocket.receive()

            # ---------- 프론트 제어 ----------
            if "text" in message:
                try:
                    msg = json.loads(message["text"])
                    if msg.get("type") == "tts_end":
                        app_state.app_engine.state = "LISTENING"
                        collecting = False
                        pcm_buffer.clear()
                        print("[WS] 🔁 TTS ended → LISTENING")
                        continue
                except Exception:
                    continue

            # ---------- 오디오 프레임 ----------
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
            # 2️⃣ 서버 차단 구간
            # ==================================================
            if app_state.app_engine.state in ("SPEAKING", "THINKING"):
                collecting = False
                pcm_buffer.clear()
                continue

            # ==================================================
            # 3️⃣ 발화 시작 판단
            # ==================================================
            if not collecting:
                is_speech = vad.is_speech(pcm) or rms > SILENCE_RMS_THRESHOLD
            else:
                is_speech = rms > SILENCE_RMS_THRESHOLD

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
            if collecting and (now - last_non_silence_ts) >= END_SILENCE_SEC:
                print("[WS] 🛑 Speech ended")
                collecting = False

                if not pcm_buffer:
                    continue

                total_samples = sum(len(chunk) for chunk in pcm_buffer)
                total_audio_sec = total_samples / 16000.0

                if total_audio_sec < MIN_AUDIO_SEC:
                    pcm_buffer.clear()
                    app_state.app_engine.state = "LISTENING"
                    print("[WS] ⏭️ Too short audio → skip STT")
                    continue

                # ==================================================
                # 5️⃣ THINKING 알림 (UX)
                # ==================================================
                await websocket.send_json({
                    "type": "assistant_state",
                    "state": "THINKING",
                })
                print("[WS] 💭 THINKING sent to client")

                # ==================================================
                # 6️⃣ STT
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
                # 7️⃣ AppEngine
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
                # 8️⃣ TTS
                # ==================================================
                tts_url = None
                if reply_text:
                    app_state.app_engine.state = "SPEAKING"
                    tts_url = synthesize(reply_text)

                # ==================================================
                # 9️⃣ 프론트 응답
                # ==================================================
                await websocket.send_json({
                    "type": "assistant_message",
                    "text": reply_text,
                    "tts_url": tts_url,
                    "conversation_state": conversation_state,
                    "end_session": end_session,
                })

                # ==================================================
                # 🔟 대화 완전 종료
                # ==================================================
                if end_session:
                    app_state.app_engine.state = "IDLE"
                    collecting = False
                    pcm_buffer.clear()
                    print("[WS] 🛑 Conversation ended → IDLE")

    except WebSocketDisconnect:
        print("[WS] ❌ Client disconnected")

    except Exception as e:
        print("[WS] 💥 Error:", repr(e))
        await websocket.close()
