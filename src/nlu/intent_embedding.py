from __future__ import annotations

"""
임베딩 기반 1차 의도 분류 모듈 (운영 최종본)

이 모듈의 역할은 "정답을 맞히는 것"이 아니다.

설계 목적
--------------------------------------------------
- 1차 의도 분류를 '빠른 라우팅 단계'로 사용한다
- 확신 있는 경우에만 intent를 확정한다
- 애매한 경우에는 절대 억지로 분류하지 않는다
- LLM은 여기서 절대 호출하지 않는다
- 애매한 케이스는 Decision Layer(AppEngine)에서
  2차 대화형 모델로 자연스럽게 넘긴다

핵심 철학
--------------------------------------------------
❌ 이 단계에서 모든 걸 맞히려 하지 않는다
✅ 이 단계에서 "자동화해도 안전한 것"만 처리한다
"""

import numpy as np
from typing import Dict, List

from sentence_transformers import SentenceTransformer
from src.nlu.intent_schema import Intent, IntentResult


# ==================================================
# Embedding Model
#
# - 의미 유사도 계산 전용
# - 분류기 ❌ / 추론기 ❌
# - 빠르고 안정적인 CPU 모델
# ==================================================
_EMBEDDING_MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ==================================================
# Intent Prototype 문장
#
# - 실제 STT 발화 스타일
# - 규칙 설명 ❌
# - 의미 anchor 용도
# ==================================================
INTENT_PROTOTYPES: Dict[Intent, List[str]] = {
    Intent.EXIT: [
        "출차하려는데 안 열려요",
        "차를 빼려고 합니다",
        "출구에서 멈췄어요",
        "차단기가 안 올라가요",
    ],
    Intent.ENTRY: [
        "입차하려는데 안 열려요",
        "차가 인식이 안 돼요",
        "들어가려고 하는데 막혔어요",
    ],
    Intent.PAYMENT: [
        "주차비 결제가 안 돼요",
        "요금이 이상해요",
        "정산을 못 했어요",
        "결제 어디서 해요",
    ],
    Intent.REGISTRATION: [
        "차량 등록해야 하나요",
        "방문자 등록 어디서 해요",
    ],
    Intent.TIME_PRICE: [
        "주차 시간 얼마나 됐어요",
        "요금 기준이 어떻게 돼요",
        "얼마 나왔는지 알고 싶어요",
    ],
    Intent.FACILITY: [
        "기계가 고장난 것 같아요",
        "차단기가 멈췄어요",
        "기기가 안 돼요",
    ],
    Intent.COMPLAINT: [
        "왜 안 되는 거죠",
        "너무 불편해요",
        "짜증나요",
        "이상해요",
    ],
}


# ==================================================
# Intent별 자동 확정 기준
#
# NONE 증가 = 실패 ❌
# NONE 증가 = 안전성 ⭕
# ==================================================
INTENT_THRESHOLDS: Dict[Intent, float] = {
    Intent.EXIT: 0.72,
    Intent.ENTRY: 0.72,
    Intent.PAYMENT: 0.72,
    Intent.TIME_PRICE: 0.68,
    Intent.REGISTRATION: 0.68,
    Intent.FACILITY: 0.65,
    Intent.COMPLAINT: 0.60,
}

# top1 / top2 유사도 차이 기준
GAP_THRESHOLD = 0.015


# ==================================================
# 도메인 키워드 보정
#
# - 임베딩을 뒤엎지 않음
# - 상식 수준의 미세 보정만
# ==================================================
KEYWORD_BOOST: Dict[Intent, List[str]] = {
    Intent.PAYMENT: ["결제", "카드", "정산", "요금", "주차비"],
    Intent.EXIT: ["출차", "출구", "나가"],
    Intent.ENTRY: ["입차", "들어가"],
    Intent.REGISTRATION: ["등록", "번호판", "방문"],
}

KEYWORD_BOOST_SCORE = 0.06  # 절대 키우지 말 것


# ==================================================
# Prototype Embedding 사전 계산
# ==================================================
_INTENT_EMBEDDINGS: Dict[Intent, np.ndarray] = {}

for intent, sentences in INTENT_PROTOTYPES.items():
    vecs = _EMBEDDING_MODEL.encode(
        sentences,
        normalize_embeddings=True,
    )
    _INTENT_EMBEDDINGS[intent] = np.mean(vecs, axis=0)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """normalize_embeddings=True → dot = cosine"""
    return float(np.dot(a, b))


# ==================================================
# 최종 1차 의도 분류
# ==================================================
def detect_intent_embedding(text: str) -> IntentResult:
    """
    임베딩 + 도메인 바이어스 기반 1차 의도 분류

    ❌ 틀릴 바엔 확정하지 않는다
    ⭕ 확실할 때만 자동화한다
    """

    print("\n" + "=" * 60)
    print(f"[INTENT-EMBEDDING] Input text: {text}")

    if not text or not text.strip():
        print("[INTENT-EMBEDDING] Empty input → NONE")
        return IntentResult(intent=Intent.NONE, confidence=0.0)

    # 1️⃣ 사용자 발화 임베딩
    user_vec = _EMBEDDING_MODEL.encode(
        text,
        normalize_embeddings=True,
    )

    # 2️⃣ 유사도 계산
    scores: Dict[Intent, float] = {}
    for intent, proto_vec in _INTENT_EMBEDDINGS.items():
        scores[intent] = _cosine(user_vec, proto_vec)

    # 3️⃣ 키워드 보정
    for intent, keywords in KEYWORD_BOOST.items():
        if any(k in text for k in keywords):
            scores[intent] += KEYWORD_BOOST_SCORE
            print(
                f"[INTENT-EMBEDDING] 🔑 Keyword boost → "
                f"{intent.value} (+{KEYWORD_BOOST_SCORE})"
            )

    # 4️⃣ 정렬
    sorted_scores = sorted(
        scores.items(), key=lambda x: x[1], reverse=True
    )

    for intent, score in sorted_scores:
        print(f"  - {intent.value:<15} : {score:.4f}")

    top_intent, top_score = sorted_scores[0]
    second_score = sorted_scores[1][1]
    gap = top_score - second_score
    threshold = INTENT_THRESHOLDS.get(top_intent, 0.7)

    print(
        f"[INTENT-EMBEDDING] Top={top_intent.value}, "
        f"Score={top_score:.4f}, Gap={gap:.4f}"
    )

    conf = round(float(top_score), 2)

    # 5️⃣ 자동 확정 판단
    if top_score >= threshold and gap >= GAP_THRESHOLD:
        print(f"[INTENT-EMBEDDING] ✅ CONFIRMED → {top_intent.value}")
        return IntentResult(
            intent=top_intent,
            confidence=conf,
        )

    print("[INTENT-EMBEDDING] ⚠️ AMBIGUOUS → NONE")
    return IntentResult(
        intent=Intent.NONE,
        confidence=conf,
    )
