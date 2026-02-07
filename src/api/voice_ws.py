from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np
import time
import json
import asyncio

import src.app_state as app_state
from src.speech.vad import VoiceActivityDetector
from src.speech.whisper_service import transcribe_pcm_chunks
from src.speech.tts import synthesize

router = APIRouter()

# ==================================================
# 🔧 튜닝 포인트 (CPU 서버 기준, 안정성 검증된 값)
# ==================================================

# 배경 소음 vs 실제 발화 구분용 RMS 임계값
SILENCE_RMS_THRESHOLD = 0.012

# "말이 끝났다"고 판단하는 최종 침묵 시간
# 너무 줄이면 문장 중간에서 끊김
END_SILENCE_SEC = 0.4

# 🔥 핵심 최적화 포인트
# 말이 끝났을 확률이 높은 시점에
# Whisper STT를 미리 시작하는 기준
PRERUN_SILENCE_SEC = 0.2

# 너무 짧은 발화는 STT를 태우지 않음
# (속도 + 오인식 방지)
MIN_AUDIO_SEC = 0.5

# Whisper에 넘기지 않을 말 끝 무음 길이
CUT_AUDIO_SEC = 0.2

# 오디오 샘플링 레이트 (고정)
SAMPLE_RATE = 16000


@router.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket):
    """
    WebSocket 기반 실시간 음성 처리 엔드포인트

    전체 흐름:
    1) 음성 수신
    2) 발화 시작 감지 (Silero VAD + RMS)
    3) 발화 종료 감지 (RMS + 시간)
    4) STT pre-run (침묵 중 미리 실행)
    5) STT 결과 확정
    6) AppEngine → TTS → 프론트 응답
    """

    await websocket.accept()
    print("[WS] 🔌 Client connected")

    # --------------------------------------------------
    # VAD
    # - 역할: "발화 시작 감지" 전용
    # - 스트리밍 중 매 chunk 호출 ❌
    # --------------------------------------------------
    vad = VoiceActivityDetector()

    # PCM 오디오 누적 버퍼
    pcm_buffer: list[np.ndarray] = []

    # 현재 발화 수집 중인지 여부
    collecting = False

    # 마지막으로 소리가 감지된 시점
    last_non_silence_ts = 0.0

    # --------------------------------------------------
    # 🔥 STT pre-run 관련 상태
    # --------------------------------------------------
    # Whisper를 백그라운드에서 미리 실행하는 task
    prerun_task: asyncio.Task | None = None

    # pre-run 결과를 실제로 사용했는지 여부 (디버그용)
    prerun_used = False

    # 초기 상태
    app_state.app_engine.state = "LISTENING"

    try:
        while True:
            # ==================================================
            # 0️⃣ 메시지 수신
            # ==================================================
            message = await websocket.receive()

            # --------------------------------------------------
            # (A) 프론트에서 오는 제어 메시지
            # --------------------------------------------------
            if "text" in message:
                try:
                    msg = json.loads(message["text"])

                    # TTS 재생이 끝났다는 신호
                    # → 다시 음성 입력 받을 준비
                    if msg.get("type") == "tts_end":
                        app_state.app_engine.state = "LISTENING"
                        collecting = False
                        pcm_buffer.clear()
                        prerun_task = None
                        prerun_used = False
                        print("[WS] 🔁 TTS ended → LISTENING")
                        continue

                except Exception:
                    continue

            # --------------------------------------------------
            # (B) 오디오 프레임
            # --------------------------------------------------
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
            # THINKING / SPEAKING 중에는
            # 사용자 음성 무시 (중복 처리 방지)
            # ==================================================
            if app_state.app_engine.state in ("SPEAKING", "THINKING"):
                collecting = False
                pcm_buffer.clear()
                prerun_task = None
                prerun_used = False
                continue

            # ==================================================
            # 3️⃣ 발화 시작 판단
            # --------------------------------------------------
            # ✔ 아직 collecting 전:
            #    - Silero VAD + RMS
            # ✔ collecting 중:
            #    - RMS만 사용 (속도 + 안정성)
            # ==================================================
            if not collecting:
                is_speech = vad.is_speech(pcm) or rms > SILENCE_RMS_THRESHOLD
            else:
                is_speech = rms > SILENCE_RMS_THRESHOLD

            if is_speech:
                if not collecting:
                    collecting = True
                    pcm_buffer.clear()
                    prerun_task = None
                    prerun_used = False
                    print("[WS] 🎤 Speech started")

                pcm_buffer.append(pcm)
                last_non_silence_ts = now
                continue

            # ==================================================
            # 4️⃣ STT pre-run 시작 지점
            # --------------------------------------------------
            # 말이 끝났을 "확률"이 높아지는 시점에
            # Whisper를 백그라운드에서 미리 실행
            # ==================================================
            silence_time = now - last_non_silence_ts

            if (
                collecting
                and prerun_task is None
                and silence_time >= PRERUN_SILENCE_SEC
            ):
                print("[WS] ⚡ STT pre-run started")

                # Whisper에 넘길 오디오 생성
                # (말 끝 무음 CUT_AUDIO_SEC 만큼 제거)
                cut_samples = int(SAMPLE_RATE * CUT_AUDIO_SEC)
                audio = np.concatenate(pcm_buffer).astype(np.float32)
                if audio.size > cut_samples:
                    audio = audio[:-cut_samples]

                # Whisper STT를 별도 스레드에서 실행
                prerun_task = asyncio.create_task(
                    asyncio.to_thread(
                        transcribe_pcm_chunks,
                        [audio],
                        app_state.whisper_model,
                    )
                )

            # ==================================================
            # 5️⃣ 발화 종료 판단 (최종)
            # ==================================================
            if collecting and silence_time >= END_SILENCE_SEC:
                collecting = False
                print("[WS] 🛑 Speech ended")

                # 전체 발화 길이 계산
                total_samples = sum(len(c) for c in pcm_buffer)
                total_audio_sec = total_samples / SAMPLE_RATE

                # 너무 짧은 발화는 무시
                if total_audio_sec < MIN_AUDIO_SEC:
                    pcm_buffer.clear()
                    prerun_task = None
                    prerun_used = False
                    continue

                # 프론트에 "생각 중" 상태 알림
                await websocket.send_json({
                    "type": "assistant_state",
                    "state": "THINKING",
                })

                # --------------------------------------------------
                # pre-run 결과 재사용
                # --------------------------------------------------
                if prerun_task:
                    try:
                        text = await prerun_task
                        prerun_used = True
                        print("[WS] ⚡ pre-run STT reused")
                    except Exception:
                        text = ""
                else:
                    # pre-run이 없으면 일반 STT 실행
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
                # 6️⃣ AppEngine 처리
                # ==================================================
                app_state.app_engine.state = "THINKING"
                result = app_state.app_engine.handle_text(text)

                reply_text = result.get("text", "")
                conversation_state = result.get(
                    "conversation_state", "WAITING_USER"
                )
                end_session = result.get("end_session", False)

                # ==================================================
                # 7️⃣ TTS
                # ==================================================
                tts_url = None
                if reply_text:
                    app_state.app_engine.state = "SPEAKING"
                    tts_url = synthesize(reply_text)

                # ==================================================
                # 8️⃣ 프론트 응답
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
                    app_state.app_engine.state = "IDLE"
                    collecting = False
                    pcm_buffer.clear()
                    print("[WS] 🛑 Conversation ended → IDLE")

    except WebSocketDisconnect:
        print("[WS] ❌ Client disconnected")

    except Exception as e:
        print("[WS] 💥 Error:", repr(e))
        await websocket.close()
