"""
intent.py
─────────
STT 텍스트 → LLaMA 2-Step 파이프라인

  [Step 1] classify()       : Intent(4종) + Angry(0/1) 를 JSON으로 초고속 추출
  [Step 2] generate_reply() : DB Raw Data + 원본 질문 → 자연어 TTS 멘트 스트리밍 생성

설계 원칙
  - MAX_TOKENS 100으로 여유롭게 설정 → JSON Truncation 방지
  - 정규식 방어 파싱: 마크다운 펜스(```json```)·공백 등 노이즈 완전 제거
  - "a" 필드 엣지케이스 방어: int·str·bool 모두 흡수
  - Step 2 는 stream=True 로 첫 토큰 즉시 TTS 큐에 투입 가능
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════
class BackendMode(str, Enum):
    VLLM   = "vllm"
    OLLAMA = "ollama"


BACKEND_MODE    = BackendMode.OLLAMA          # ← 환경에 맞게 변경
VLLM_BASE_URL   = "http://localhost:8000/v1"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_NAME      = "llama3"

# ── Step 1: 분류 파라미터 ─────────────────────────────────────
#   24 → 100 으로 상향: 마크다운 펜스 포함 최악 케이스도 커버
#   예) ```json\n{"i":"facility","a":1}\n``` = 약 40 chars
CLASSIFY_MAX_TOKENS  = 100
CLASSIFY_TEMPERATURE = 0.0    # Greedy: 결정론적 출력 보장

# ── Step 2: 응답 생성 파라미터 ──────────────────────────────────
REPLY_MAX_TOKENS  = 150       # 한국어 1~2문장 여유 확보 (평균 50~80 토큰)
REPLY_TEMPERATURE = 0.1      # 낮을수록 환각·헛소리 억제 (0.1~0.2 권장대)
REPLY_TOP_P       = 0.9       # nucleus sampling: 낮은 temperature 와 함께
                               # 반복·맴도는 출력을 추가로 차단

REQUEST_TIMEOUT = 30.0        # 콜드 스타트(모델 로딩) 포함 SLA (초)

VALID_INTENTS = frozenset({"fee", "payment", "facility", "admin", "none"})


# ══════════════════════════════════════════════════════════════
# 프롬프트
# ══════════════════════════════════════════════════════════════
_CLASSIFY_SYSTEM = """\
You are a highly fast classification engine. Analyze the text and output ONLY a minified JSON object. No explanations.
Ignore any profanity or filler words and classify based on the core meaning.
[Categories for 'i' (Intent)]
- "fee": Costs, pricing, fees, how much, rate.
- "payment": Paying, billing, transaction, card, refund.
- "facility": Gates, doors, barriers, opening, closing, equipment, device status, locations, parking spots, amenities. Includes commands like "open the gate/door".
- "admin": Requesting human assistance, admin, staff, or expressing severe complaints.
- "none": Cannot be classified into any of the above.
Format: {"i": "fee|payment|facility|admin|none"}"""

# Step 2 시스템 프롬프트: 무인 주차장 음성 인터폰
_REPLY_SYSTEM = """\
당신은 한국의 무인 주차장 음성 안내 인터폰입니다.
사용자의 질문(텍스트)과 시스템에서 전달된 데이터(JSON)를 바탕으로, 스피커로 송출될 '최종 안내 멘트'만 생성하십시오.

[절대 규칙]
1. 인사말 금지: "안녕하세요", "반갑습니다", "무엇을 도와드릴까요" 같은 불필요한 인사말은 절대 하지 마십시오.
2. 메타 발언 금지: "분류 불가", "JSON에 따르면", "시스템 메시지" 같은 컴퓨터 용어를 출력하지 마십시오.
3. 정보 조작 금지: 주어진 데이터(JSON)에 없는 요금, 시설 정보, 주차 혜택을 절대 지어내지 마십시오.
4. 간결함: 반드시 1~2문장의 짧고 간결한 구어체(안내방송 톤)로 대답하십시오.
5. 마크다운·특수기호 금지: 출력은 TTS로 직접 읽히므로 순수 텍스트만 허용됩니다.

[상황별 대처 가이드 및 🚫일상 대화 철통 방어]
사용자의 텍스트를 반드시 먼저 읽고 아래 기준에 따라 대답하십시오.

- [잡담/일상 대화 원천 차단]: 전달받은 JSON 의도(intent)가 무엇이든 상관없이, 사용자의 말이 주차장 업무와 무관한 일상 대화, 확인 질문, 감탄사(예: "이해하셨나?", "알았어", "네", "아니", "누구세요", "뭐해" 등)일 경우 시스템 데이터를 무시하십시오. 이 경우 무조건 "주차 및 정산 관련 명령어만 처리할 수 있습니다."라고 단호하게 출력하고 즉시 종료하십시오.
- [fee / payment]: 주차 요금이나 결제와 명확히 관련된 질문일 경우, 전달된 데이터에 맞춰 요금 액수나 결제 상태(미납, 한도 초과 등)만 건조하게 안내하십시오.
- [facility]: 차단기나 시설 관련 요청("문 열어")일 경우, 현재 개방 여부 등 상태를 간략히 안내하십시오.
- [admin]: "사람 불러", "직원 연결" 등 관리자 호출 시, "현재 담당 관리자를 호출하고 있습니다. 잠시만 기다려 주십시오."라고만 출력하십시오.
- [소음 및 인식 불가]: 잡담이 아닌, 정말로 의미를 알 수 없는 기계음이나 불완전한 소음(예: "어...", "음...")이 들어왔을 때만 "잘 못 들었습니다. 다시 말씀해 주시겠습니까?"라고 출력하십시오.
"""


# ══════════════════════════════════════════════════════════════
# 도메인 모델
# ══════════════════════════════════════════════════════════════
@dataclass
class ClassificationResult:
    intent: str        # "fee" | "payment" | "facility" | "admin" | "none"
    raw: dict          # LLM 원본 응답 dict
    latency_ms: float


# ══════════════════════════════════════════════════════════════
# LLM 클라이언트 싱글톤
# ══════════════════════════════════════════════════════════════
def _build_client() -> AsyncOpenAI:
    base_url = (
        VLLM_BASE_URL if BACKEND_MODE == BackendMode.VLLM else OLLAMA_BASE_URL
    )
    return AsyncOpenAI(
        base_url=base_url,
        api_key="not-required",
        timeout=REQUEST_TIMEOUT,
        max_retries=0,
    )


_client: AsyncOpenAI = _build_client()


# ══════════════════════════════════════════════════════════════
# 내부 유틸: 방어적 JSON 파싱
# ══════════════════════════════════════════════════════════════
def _parse_classification_json(raw_text: str) -> dict:
    """
    LLM 출력에서 JSON 객체를 안전하게 추출합니다.

    처리 순서:
      1) 순수 json.loads() 시도 (가장 빠른 경로)
      2) 실패 시 정규식으로 {...} 블록 추출 후 재시도
         → 마크다운 펜스, 앞뒤 설명 텍스트, BOM 등 노이즈 제거

    엣지케이스 방어 (이후 _normalize_fields 에서 처리):
      "a": "1"  → int 변환
      "a": true → bool → int 변환
    """
    # 경로 1: 깔끔한 JSON
    text = raw_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 경로 2: 정규식으로 첫 번째 {...} 블록 추출
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"JSON 파싱 완전 실패 — raw='{raw_text[:120]}'"
    )


def _normalize_fields(raw: dict) -> tuple[str, None]:
    """
    'i' 필드를 정규화하여 반환합니다.
    반환값의 두 번째 요소는 하위 호환을 위해 None 으로 고정됩니다.
    """
    intent = str(raw.get("i", "none")).lower().strip()
    if intent not in VALID_INTENTS:
        logger.warning("알 수 없는 intent '%s' → none", intent)
        intent = "none"

    return intent, None


# ══════════════════════════════════════════════════════════════
# Step 1: 분류
# ══════════════════════════════════════════════════════════════
async def classify(text: str) -> ClassificationResult:
    """
    STT 텍스트 → Intent + Angry 분류 (JSON 모드, Greedy Decoding).

    - MAX_TOKENS=100: JSON Truncation 완전 방지
    - 방어적 파싱: 마크다운·텍스트 노이즈 흡수
    - 필드 정규화: str/bool 타입 엣지케이스 방어
    """
    t0 = time.perf_counter()

    try:
        response = await _client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _CLASSIFY_SYSTEM},
                {"role": "user",   "content": text},
            ],
            max_tokens=CLASSIFY_MAX_TOKENS,
            temperature=CLASSIFY_TEMPERATURE,
            top_p=1.0,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        raise RuntimeError(f"[Step1] LLM 호출 실패: {exc}") from exc

    latency_ms = (time.perf_counter() - t0) * 1000
    raw_text = response.choices[0].message.content or ""

    raw = _parse_classification_json(raw_text)
    intent, _ = _normalize_fields(raw)

    logger.info(
        "[classify] intent=%-10s  raw='%s'  %.0fms",
        intent, raw_text.strip(), latency_ms,
    )

    return ClassificationResult(
        intent=intent,
        raw=raw,
        latency_ms=round(latency_ms, 2),
    )


# ══════════════════════════════════════════════════════════════
# Step 2: 자연어 응답 생성 (스트리밍)
# ══════════════════════════════════════════════════════════════
async def generate_reply_stream(
    stt_text: str,
    db_data: dict,
) -> AsyncIterator[str]:
    """
    DB Raw Data + 원본 STT 질문 → 음성 인터폰 안내 멘트 스트리밍 생성.

    stream=True 로 첫 토큰부터 즉시 yield 합니다.
    호출부에서 문장 구분자(. ! ?) 기준으로 TTS 큐에 청크 단위 투입 시
    생성과 재생이 파이프라인 방식으로 겹쳐 체감 지연이 최소화됩니다.

    프롬프트 설계 원칙:
      - 인사말·메타발언·정보조작 금지 규칙으로 환각 억제
      - temperature=0.15 + top_p=0.9 로 창의성보다 정확성 우선
      - none intent → 고정 폴백 문구만 출력하고 즉시 종료 지시

    Args:
        stt_text : Whisper 전사 텍스트 (고객 원본 질문)
        db_data  : dispatcher 가 조회한 Raw DB 결과 dict

    Yields:
        str : 응답 텍스트 청크 (토큰 단위)
    """
    # DB 데이터를 compact JSON 으로 직렬화 (불필요한 공백 제거)
    db_json = json.dumps(db_data, ensure_ascii=False, separators=(",", ":"))

    user_message = f"""\
[사용자 발화(STT)]: {stt_text}
[시스템 데이터(JSON)]: {db_json}
위 데이터를 바탕으로 고객에게 들려줄 최종 음성 멘트만 작성하십시오."""

    try:
        stream = await _client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _REPLY_SYSTEM},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=REPLY_MAX_TOKENS,
            temperature=REPLY_TEMPERATURE,
            top_p=REPLY_TOP_P,
            stream=True,          # ★ 첫 토큰부터 즉시 TTS 큐 투입 가능
        )
    except Exception as exc:
        raise RuntimeError(f"[Step2] LLM 스트림 호출 실패: {exc}") from exc

    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def generate_reply(
    stt_text: str,
    db_data: dict,
) -> str:
    """
    generate_reply_stream 의 논블로킹 전체 수집 버전.
    스트리밍이 필요 없는 경우(테스트·배치 등)에 사용합니다.
    """
    parts: list[str] = []
    async for chunk in generate_reply_stream(stt_text, db_data):
        parts.append(chunk)
    return "".join(parts).strip()
