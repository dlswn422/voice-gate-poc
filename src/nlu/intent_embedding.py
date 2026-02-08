from __future__ import annotations

"""
임베딩 기반 1차 의도 분류 모듈 (운영 최종본 v2)

이 모듈의 목적은 "의도를 정확히 맞히는 것"이 아니다.

────────────────────────────────────────
설계 목적
────────────────────────────────────────
- 자동 처리 이후, 사용자의 "문제 발생 발화"를 빠르게 감지한다
- 문제의 '대략적인 영역'만 태깅한다 (출차 / 결제 / 기기 / 불만 등)
- 확신 있는 경우에만 intent를 확정한다
- 애매한 경우에는 NONE으로 넘겨 2차 대화로 위임한다
- LLM은 절대 호출하지 않는다 (CPU / 안정성 / 예측 가능성)

────────────────────────────────────────
핵심 철학
────────────────────────────────────────
❌ 이 단계에서 모든 문제를 이해하려 하지 않는다
❌ 원인을 추론하지 않는다
✅ "문제가 발생했다"는 신호를 놓치지 않는다
✅ 대화로 넘길 수 있을 정도의 힌트만 제공한다
"""

import numpy as np
from typing import Dict, List

from sentence_transformers import SentenceTransformer
from src.nlu.intent_schema import Intent, IntentResult


# ==================================================
# Embedding Model
#
# - 문장 의미 유사도 계산 전용
# - 분류기 / 추론기 역할 ❌
# - CPU 환경에서 빠르고 안정적인 모델
# ==================================================
_EMBEDDING_MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ==================================================
# 🔧 [중요] 입력 정규화 레이어
#
# - 짧고 정보가 부족한 발화를
#   "모델이 이해 가능한 최소 문장"으로 확장
# - 의미는 바꾸지 않고, 신호만 증폭
#
# ▶ 이 레이어가 '짧은 발화 대응'의 핵심
# ==================================================
SHORT_ISSUE_EXPANSION: Dict[str, str] = {
    "안 열려요": "차단기가 안 열려요",
    "안열려요": "차단기가 안 열려요",
    "안 돼요": "기기가 정상적으로 작동하지 않아요",
    "안돼요": "기기가 정상적으로 작동하지 않아요",
    "멈췄어요": "기기가 작동 중 멈췄어요",
    "이상해요": "기기 상태가 이상해요",
    "왜 안돼요": "기기가 왜 작동하지 않는지 모르겠어요",
}

def normalize_issue_text(text: str) -> str:
    """
    짧은 문제 발화를 의미 보존 상태로 확장한다.

    - LLM ❌
    - 룰 기반 ⭕
    - 실패해도 원문 그대로 사용하므로 안전
    """
    t = text.strip()
    return SHORT_ISSUE_EXPANSION.get(t, t)


# ==================================================
# Intent Prototype 문장
#
# - 실제 STT 발화 스타일
# - "현상 보고" 중심
# - 규칙 설명 / 교과서 문장 ❌
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
        "작동을 안 해요",
        "계속 멈춰 있어요",
    ],
    Intent.COMPLAINT: [
        "왜 안 되는 거죠",
        "너무 불편해요",
        "짜증나요",
        "이상해요",
    ],
}


# ==================================================
# Intent별 자동 확정 threshold
#
# - 값이 높을수록: 확정이 어려움 (보수적)
# - 값이 낮을수록: 확정이 쉬움 (공격적)
#
# ▶ 튜닝 포인트:
#   - NONE이 많으면 → 낮춘다
#   - 오분류가 많으면 → 올린다
# ==================================================
INTENT_THRESHOLDS: Dict[Intent, float] = {
    Intent.EXIT: 0.72,
    Intent.ENTRY: 0.72,
    Intent.PAYMENT: 0.72,
    Intent.TIME_PRICE: 0.68,
    Intent.REGISTRATION: 0.68,
    Intent.FACILITY: 0.63,     # 문제 포착률을 위해 다소 낮춤
    Intent.COMPLAINT: 0.58,    # 감정/불만은 넓게 허용
}


# ==================================================
# Top-1 / Top-2 score 차이 기준
#
# - 클수록: 확정이 어려워짐
# - 작을수록: 빠른 확정
#
# ▶ 튜닝 포인트:
#   - 짧은 발화가 많으면 줄이는 게 일반적
# ==================================================
GAP_THRESHOLD = 0.015


# ==================================================
# 키워드 기반 미세 보정
#
# - 임베딩 결과를 "뒤엎지 않음"
# - 사람이 봐도 당연한 방향으로만 살짝 밀어줌
# ==================================================
KEYWORD_BOOST: Dict[Intent, List[str]] = {
    Intent.PAYMENT: ["결제", "카드", "정산", "요금", "주차비"],
    Intent.EXIT: ["출차", "출구", "나가"],
    Intent.ENTRY: ["입차", "들어가"],
    Intent.REGISTRATION: ["등록", "번호판", "방문"],
}

# ▶ 너무 키우면 rule-engine이 됨
KEYWORD_BOOST_SCORE = 0.06


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
    """normalize_embeddings=True → dot product = cosine similarity"""
    return float(np.dot(a, b))


# ==================================================
# 🚦 최종 1차 의도 분류
# ==================================================
def detect_intent_embedding(text: str) -> IntentResult:
    """
    임베딩 + 도메인 보정 기반 1차 의도 분류

    반환 규칙
    ----------
    - 확실하면 → Intent 확정
    - 애매하면 → Intent.NONE
    """

    print("\n" + "=" * 60)
    print(f"[INTENT-EMBEDDING] Raw input: {text}")

    if not text or not text.strip():
        return IntentResult(intent=Intent.NONE, confidence=0.0)

    # 1️⃣ 입력 정규화 (짧은 문제 발화 보정)
    normalized_text = normalize_issue_text(text)
    print(f"[INTENT-EMBEDDING] Normalized input: {normalized_text}")

    # 2️⃣ 사용자 발화 임베딩
    user_vec = _EMBEDDING_MODEL.encode(
        normalized_text,
        normalize_embeddings=True,
    )

    # 3️⃣ intent별 유사도 계산
    scores: Dict[Intent, float] = {
        intent: _cosine(user_vec, proto_vec)
        for intent, proto_vec in _INTENT_EMBEDDINGS.items()
    }

    # 4️⃣ 키워드 기반 미세 보정
    for intent, keywords in KEYWORD_BOOST.items():
        if any(k in normalized_text for k in keywords):
            scores[intent] += KEYWORD_BOOST_SCORE

    # 5️⃣ 점수 정렬
    sorted_scores = sorted(
        scores.items(), key=lambda x: x[1], reverse=True
    )

    for intent, score in sorted_scores:
        print(f"  - {intent.value:<15} : {score:.4f}")

    top_intent, top_score = sorted_scores[0]
    second_score = sorted_scores[1][1]
    gap = top_score - second_score
    threshold = INTENT_THRESHOLDS.get(top_intent, 0.7)

    confidence = round(float(top_score), 2)

    # 6️⃣ 자동 확정 판단
    if top_score >= threshold and gap >= GAP_THRESHOLD:
        print(f"[INTENT-EMBEDDING] ✅ CONFIRMED → {top_intent.value}")
        return IntentResult(intent=top_intent, confidence=confidence)

    print("[INTENT-EMBEDDING] ⚠️ AMBIGUOUS → NONE")
    return IntentResult(intent=Intent.NONE, confidence=confidence)
