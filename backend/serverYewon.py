from __future__ import annotations

import os
import json
import asyncio
import time
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI

load_dotenv()

app = FastAPI()

# ----------------------------
# Single-session guard (optional)
# ----------------------------
_ACTIVE_SESSION_LOCK = asyncio.Lock()


def make_aoai_client() -> AzureOpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
    if not endpoint or not api_key or not api_version:
        raise RuntimeError("AZURE_OPENAI_* env missing")
    return AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)


AOAI = make_aoai_client()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "parking-llm").strip()

SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "").strip()
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "").strip()
SPEECH_LANG = os.getenv("AZURE_SPEECH_LANGUAGE", "ko-KR").strip()
SPEECH_VOICE = os.getenv("AZURE_SPEECH_VOICE", "ko-KR-SunHiNeural").strip()

if not SPEECH_KEY or not SPEECH_REGION:
    raise RuntimeError("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION env missing")


# ✅ 예원 식당 키오스크 시스템 프롬프트(문자열로! 반드시 따옴표/삼중따옴표 사용)
SYSTEM_PROMPT_YEWON = """너는 “예원 식당” 키오스크 안내/주문 도우미 AI다. 사용자는 키오스크 앞에서 짧게 말할 수 있으며, 음성 인식(STT) 결과에는 오탈자/문장부호/띄어쓰기 오류가 있을 수 있다.

[매장 정보]
- 매장명: 예원 식당
- 위치/소개: 자카르타 SCBD(경제 중심지) JL. 세노파티에 위치한 한식당. 퀄리티 있는 한식과 한인 쉐프 상주, 숙성고기로 인기가 많다.
- 영업시간: 11:00 ~ 22:00
- 전화번호: 02172120533
- 가격대: 15k ~ 30k
- 추천: 점심 한상 메뉴, 패키지 단체 메뉴

[목표]
- 사용자가 키오스크에서 “주문/결제/쿠폰·할인/매장·포장/변경·취소/영수증/오류”를 빠르게 해결하도록 돕는다.
- 답변은 짧고 친절하게, 필요할 때만 확인 질문 1개를 한다.
- 기본 문맥은 “지금 예원 식당 키오스크 이용 중”이다.

[말투/길이]
- 항상 한국어 존댓말.
- 1~2문장으로 답한다. (최대 3문장 예외)
- 해결 단계 중심으로 말한다. 장황한 설명 금지.
- 사과는 필요할 때 1회만 짧게.

[STT(음성인식) 처리 규칙]
- 문장부호(?, !)는 자동 보정일 수 있으니 의도 해석에 과도하게 반영하지 않는다.
- 짧은 명사구/단답(예: “신용카드”, “포장”, “점심 한상”, “패키지”, “영수증”)은 “직전 질문에 대한 답”으로 우선 해석한다.
- 불완전한 문장이라도 ‘키오스크 주문/결제 문맥’에서 의미를 최대한 복원한다.

[대화 문맥 유지 규칙(중요)]
- 이 대화는 기본적으로 “예원 식당 키오스크 주문/결제” 상담이다.
- 사용자의 답이 짧아도 직전 문맥을 유지해서 이어간다.
- 사용자가 키오스크 문맥으로 답하고 있는데도 “무슨 문의인지” 같은 범위 확인 질문을 반복하지 않는다.

[범위/거절 규칙]
- 의료/법률/투자/정치/연애 등 키오스크와 무관한 주제는 답하지 않는다.
- 다만 단어가 겹칠 수 있으니(예: “결제”, “환불”) 우선 키오스크 상황으로 해석한다.
- 키오스크/주문과 무관함이 명백할 때만 1문장으로 확인한다:
  “예원 식당 키오스크 주문/결제 관련 문의 맞으실까요? 어떤 문제가 있으신가요?”

[문제 분류(내부적으로만 사용)]
1) 주문 진행: 메뉴 선택, 옵션, 수량, 세트/패키지, 추천, 품절
2) 매장/포장/픽업: 매장식사/포장 선택, 픽업 방법
3) 결제: 카드/간편결제/현금, 승인 실패, 결제 중단
4) 할인/쿠폰: 쿠폰 적용, 할인코드, 프로모션
5) 변경/취소/환불: 주문 실수, 결제 전/후 변경 가능 여부(정책 확인 필요)
6) 영수증: 영수증 출력/재출력
7) 기기/오류: 화면 멈춤, 버튼 반응 없음, 오류 메시지

[질문 정책(1개만)]
- 정보가 부족하면 확인 질문은 “딱 1개”만 한다.
- 질문 우선순위:
  A) 지금 단계(메뉴 고르는 중/결제 화면/결제 완료 후) 또는 화면에 보이는 문구
  B) 매장/포장 또는 결제수단(카드/간편결제/현금)
  C) 점심 한상/패키지(단체) 중 어떤 걸 찾는지
- 이미 질문을 했고 사용자가 답하면, 그 답으로 다음 단계 안내로 이어간다(반복 질문 금지).

[추천/업셀(과하지 않게)]
- 추천 요청이 오면 “점심 한상” 또는 “패키지 단체 메뉴”를 우선 제안하되, 1~2개만 말한다.
- 가격대(15k~30k)를 넘는 제안은 하지 말고, 부담 없는 옵션을 안내한다.

[결제 대응]
- “결제가 안 돼요” → 결제수단(카드/간편결제/현금) 또는 화면 문구 1개만 질문한다.
- 사용자가 결제수단을 말하면, 그 수단 기준으로 가장 흔한 해결 1~2단계를 안내하고, 필요하면 질문 1개만 한다.
- 해결이 어렵다면 “다른 결제수단 선택” 또는 “직원 도움 요청”을 짧게 안내한다.

[취소/환불(안전하게)]
- 결제 후 취소/환불은 매장 정책이 필요할 수 있으므로 단정하지 말고,
  “결제 완료 상태인지” 1개만 확인한 뒤,
  주문번호/영수증 확인 + 직원 도움 요청 같은 안전한 안내를 한다.

[매장 안내]
- 영업시간/전화 문의가 오면 정확히 안내한다:
  - 영업시간: 11:00 ~ 22:00
  - 전화: 02172120533

[표정 신호(있을 때만)]
- 표정 신호는 오차 가능성이 있으므로 사실로 단정하지 말고, 말투/질문 방식에만 반영한다.
- angry/frustrated 추정: 사과 1회 + 해결 중심 + 질문 1개만.
- confused 추정: 더 단순하게, 한 단계씩, 질문 1개만.
- positive 추정: 친절하지만 짧게.

[출력 형식]
- 답변만 출력한다.
- 목록이 필요하면 1,2 정도로 짧게만.
- 카메라/표정/추정 같은 내부 신호를 사용자에게 언급하지 않는다.
"""


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def build_expression_policy(vision: dict | None) -> str:
    """
    Expression-only policy for LLM (tone/strategy only).
    3-tier confidence:
      - conf < 0.25: ignore
      - 0.25~0.45: weak (tone only)
      - >=0.45: strong (tone + strategy)
    """
    if not isinstance(vision, dict):
        return ""

    expr = None
    if vision.get("type") == "vision_expression":
        expr = vision.get("expression") or {}
    elif isinstance(vision.get("expression"), dict):
        expr = vision.get("expression") or {}

    if not isinstance(expr, dict) or not expr:
        return ""

    label = str(expr.get("label", "neutral"))
    conf = _safe_float(expr.get("confidence", 0))
    valence = _safe_float(expr.get("valence", 0))
    arousal = _safe_float(expr.get("arousal", 0))

    if conf < 0.25:
        return "\n[표정 신호] confidence 낮음 → 표정 정보는 무시하고 기본 톤.\n"

    if conf < 0.45:
        return (
            "\n[표정 신호(약한 참고)] "
            f"label={label}, conf={conf:.3f}, valence={valence:.3f}, arousal={arousal:.3f}.\n"
            "- 사실로 단정하지 말고 말투만 약하게 조정.\n"
            "- 더 짧게, 더 명확하게, 질문은 최대 1개.\n"
        )

    if label in ("angry", "frustrated"):
        return (
            "\n[표정 신호(강한 참고)] "
            f"label={label}, conf={conf:.3f}, valence={valence:.3f}, arousal={arousal:.3f}.\n"
            "- 단정 금지. 전략만 적용.\n"
            "- 사과 1회 + 해결 중심 + 질문 1개만.\n"
        )
    if label == "confused":
        return (
            "\n[표정 신호(강한 참고)] "
            f"label={label}, conf={conf:.3f}, valence={valence:.3f}, arousal={arousal:.3f}.\n"
            "- 단정 금지. 한 단계씩 쉽게 + 질문 1개만.\n"
        )
    if label == "positive":
        return (
            "\n[표정 신호(강한 참고)] "
            f"label={label}, conf={conf:.3f}, valence={valence:.3f}, arousal={arousal:.3f}.\n"
            "- 단정 금지. 친절하지만 짧게.\n"
        )

    return (
        "\n[표정 신호(강한 참고)] "
        f"label={label}, conf={conf:.3f}, valence={valence:.3f}, arousal={arousal:.3f}.\n"
        "- 단정 금지. 기본 톤.\n"
    )


def llm_reply_sync(user_text: str, vision_state: dict | None) -> str:
    """Blocking call (run via asyncio.to_thread)."""
    expr_policy = build_expression_policy(vision_state)

    resp = AOAI.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_YEWON + expr_policy},
            {"role": "user", "content": user_text},
        ],
        temperature=0.3,
        max_tokens=220,
    )
    return (resp.choices[0].message.content or "").strip()


def synth_wav_sync(text: str) -> bytes:
    """Blocking TTS (run via asyncio.to_thread). Returns WAV bytes."""
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_synthesis_voice_name = SPEECH_VOICE
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
    )
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    r = synthesizer.speak_text_async(text).get()
    if r.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        return b""
    return r.audio_data




# ----------------------------
# Warmup (optional): reduce first-turn TTS stutter
# ----------------------------
@app.on_event("startup")
async def _warmup():
    # Warm up Azure TTS once to avoid cold-start stutter on the first real reply.
    try:
        await asyncio.to_thread(synth_wav_sync, "안녕하세요.")
    except Exception:
        pass

@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    # (선택) 단일 세션만 허용
    if _ACTIVE_SESSION_LOCK.locked():
        await ws.accept()
        await ws.send_text(json.dumps({"type": "error", "message": "이미 상담이 진행 중입니다. 잠시 후 다시 시도하세요."}, ensure_ascii=False))
        await ws.close(code=1008)
        return

    await _ACTIVE_SESSION_LOCK.acquire()
    await ws.accept()

    # ✅ Latest expression-only camera signal (per websocket session)
    last_vision_state: dict | None = None
    _last_expr_log_ts = 0.0

    # --- barge-in debounce / TTS echo guard ---
    last_tts_sent_ts = 0.0
    last_barge_ts = 0.0

    def should_barge_in(partial: str) -> bool:
        """Prevent false barge-in triggered by the assistant TTS leaking into the mic.

        Heuristics:
        - Ignore very short partials.
        - Ignore partials right after we sent TTS.
        - Rate-limit barge_in.
        """
        nonlocal last_tts_sent_ts, last_barge_ts
        now = time.time()

        # 1) Protect window right after we sent TTS (echo/AGC settling)
        if now - last_tts_sent_ts < 0.8:
            return False

        # 2) Ignore too-short partials (often noise/echo)
        if len(partial.strip()) < 4:
            return False

        # 3) Rate-limit barge-in
        if now - last_barge_ts < 0.5:
            return False

        last_barge_ts = now
        return True

    # Azure STT: 브라우저 PCM16(16k mono) 스트림 수신
    stream_format = speechsdk.audio.AudioStreamFormat(samples_per_second=16000, bits_per_sample=16, channels=1)
    push_stream = speechsdk.audio.PushAudioInputStream(stream_format)
    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language = SPEECH_LANG
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    loop = asyncio.get_running_loop()
    final_queue: asyncio.Queue[str] = asyncio.Queue()

    # ----------------------------
    # Turn cancel mechanism
    # ----------------------------
    turn_id = 0
    turn_lock = asyncio.Lock()

    async def send_json(payload: dict):
        try:
            await ws.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            pass

    def bump_turn_and_barge_in():
        """User started speaking -> cancel current turn + notify frontend."""
        nonlocal turn_id
        turn_id += 1
        loop.call_soon_threadsafe(lambda: asyncio.create_task(send_json({"type": "barge_in"})))

    def on_recognizing(evt):
        text = (evt.result.text or "").strip()
        if not text:
            return

        async def _send_partial():
            # Only trigger barge-in when we are confident it's the user speaking,
            # not the assistant TTS echo leaking into the mic.
            if should_barge_in(text):
                bump_turn_and_barge_in()
            await send_json({"type": "partial", "text": text})

        loop.call_soon_threadsafe(lambda: asyncio.create_task(_send_partial()))

    def on_recognized(evt):
        if evt.result.reason != speechsdk.ResultReason.RecognizedSpeech:
            return
        text = (evt.result.text or "").strip()
        if not text:
            return
        loop.call_soon_threadsafe(final_queue.put_nowait, text)

    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)
    recognizer.start_continuous_recognition()

    async def final_consumer():
        nonlocal turn_id, last_vision_state
        while True:
            user_text = await final_queue.get()

            if len(user_text) <= 1:
                continue

            my_turn = turn_id

            async with turn_lock:
                if my_turn != turn_id:
                    continue

                await send_json({"type": "final", "text": user_text})

                bot = await asyncio.to_thread(llm_reply_sync, user_text, last_vision_state)
                if not bot:
                    bot = "죄송합니다. 다시 한 번 말씀해 주세요."

                if my_turn != turn_id:
                    continue

                await send_json({"type": "bot_text", "text": bot})

                wav = await asyncio.to_thread(synth_wav_sync, bot)

                if my_turn != turn_id:
                    continue

                if wav:
                    try:
                        # mark TTS send time (used by should_barge_in)
                        last_tts_sent_ts = time.time()
                        await ws.send_bytes(wav)
                    except Exception:
                        pass

    consumer_task = asyncio.create_task(final_consumer())

    try:
        while True:
            msg = await ws.receive()

            # control
            if "text" in msg and msg["text"]:
                try:
                    ctrl = json.loads(msg["text"])
                    ctype = ctrl.get("type")

                    if ctype == "stop":
                        break

                    if ctype == "barge_in":
                        bump_turn_and_barge_in()
                        continue

                    # ✅ 카메라 표정 신호 수신
                    if ctype == "vision_expression":
                        last_vision_state = ctrl

                        now = time.time()
                        if now - _last_expr_log_ts > 2.0:
                            _last_expr_log_ts = now
                            e = ctrl.get("expression") or {}
                            print(
                                f"[VISION_EXPR] label={e.get('label')} "
                                f"conf={e.get('confidence')} v={e.get('valence')} a={e.get('arousal')}"
                            )
                        continue

                except Exception:
                    pass

            # audio bytes
            if "bytes" in msg and msg["bytes"]:
                push_stream.write(msg["bytes"])

    except WebSocketDisconnect:
        pass
    finally:
        # cleanup
        try:
            push_stream.close()
        except Exception:
            pass

        try:
            recognizer.stop_continuous_recognition()
        except Exception:
            pass

        try:
            consumer_task.cancel()
        except Exception:
            pass

        try:
            if _ACTIVE_SESSION_LOCK.locked():
                _ACTIVE_SESSION_LOCK.release()
        except Exception:
            pass


@app.get("/health")
def health():
    return {"ok": True}
