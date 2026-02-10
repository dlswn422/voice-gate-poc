from __future__ import annotations

"""
임베딩 기반 1차 의도 분류 모듈 (운영 최종본 v3 - Intent 축소/재정의)

이 모듈의 목적은 "의도를 정확히 맞히는 것"이 아니다.

────────────────────────────────────────
설계 목적
────────────────────────────────────────
- 사용자의 "문제 발생 발화"를 빠르게 감지한다
- 문제의 '대략적인 영역'만 태깅한다 (결제 / 등록 / 시설 / 번호판 인식)
- 확신 있는 경우에만 intent를 확정한다
- 애매한 경우에는 NONE으로 넘겨 2차(대화/정책 처리)로 위임한다
- LLM은 절대 호출하지 않는다 (CPU / 안정성 / 예측 가능성)
"""

import numpy as np
from typing import Dict, List

from sentence_transformers import SentenceTransformer
from src.nlu.intent_schema import Intent, IntentResult


# ==================================================
# Embedding Model
# ==================================================
_EMBEDDING_MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ==================================================
# 🔧 입력 정규화 레이어
# - 짧고 정보가 부족한 발화를 "모델이 이해 가능한 최소 문장"으로 확장
# - 의미는 바꾸지 않고, 신호만 증폭
# ==================================================
SHORT_ISSUE_EXPANSION: Dict[str, str] = {
    "안 열려요": "차단기가 안 열려요",
    "안열려요": "차단기가 안 열려요",
    "안 돼요": "기기가 정상적으로 작동하지 않아요",
    "안돼요": "기기가 정상적으로 작동하지 않아요",
    "멈췄어요": "기기가 작동 중 멈췄어요",
    "이상해요": "기기 상태가 이상해요",
    "왜 안돼요": "기기가 왜 작동하지 않는지 모르겠어요",

    # ✅ LPR(번호판) 단문 보정
    "인식이 안돼요": "번호판 인식이 안 돼요",
    "인식이 안 돼요": "번호판 인식이 안 돼요",
    "번호판이 안돼요": "번호판 인식이 안 돼요",
    "번호판 안돼요": "번호판 인식이 안 돼요",
    "번호판 인식 안돼요": "번호판 인식이 안 돼요",
}


def normalize_issue_text(text: str) -> str:
    """
    짧은 문제 발화를 의미 보존 상태로 확장한다.
    - LLM ❌
    - 룰 기반 ⭕
    - 실패해도 원문 그대로 사용하므로 안전
    """
    t = (text or "").strip()
    return SHORT_ISSUE_EXPANSION.get(t, t)


# ==================================================
# Intent Prototype 문장
# - 실제 STT 발화 스타일 / "현상 보고" 중심
# ==================================================
INTENT_PROTOTYPES: Dict[Intent, List[str]] = {
    Intent.PAYMENT: [
        "주차비 결제가 안 돼요",
        "정산이 안 돼요",
        "카드 승인 실패가 떠요",
        "결제 오류가 나요",
        "결제했는데 출차가 안 돼요",
        "결제 완료라고 뜨는데 차단기가 안 열려요",
    ],
    Intent.REGISTRATION: [
        "방문자 등록이 안 돼요",
        "방문자 등록 어디서 해요",
        "정기권 등록이 안 돼요",
        "무료차량 등록이 안 돼요",
        "차량 등록이 안 돼요",
        "등록을 해야 하나요",
    ],
    Intent.FACILITY: [
        "차단기가 고장난 것 같아요",
        "키오스크 화면이 안 켜져요",
        "기기가 멈췄어요",
        "프린터가 안 나와요",
        "통신이 안 되는 것 같아요",
        "기기가 작동을 안 해요",
        "버튼을 눌러도 반응이 없어요",
    ],
    Intent.LPR: [
        "번호판 인식이 안 돼요",
        "번호판이 안 찍혀요",
        "차량번호가 잘못 인식돼요",
        "번호판 인식 오류가 떠요",
        "번호판이 인식되지 않았다고 떠요",
        "입차할 때 번호판 인식이 안 돼요",
        "출차할 때 번호판이 안 읽혀요",
        "번호판 카메라가 인식을 못 해요",
        "번호판이 달라요",
        "번호판이 이상해요",
    ],
}


# ==================================================
# Intent별 자동 확정 threshold
# - 값이 높을수록 보수적(확정 어려움), 낮을수록 공격적(확정 쉬움)
# ==================================================
INTENT_THRESHOLDS: Dict[Intent, float] = {
    Intent.PAYMENT: 0.72,
    Intent.REGISTRATION: 0.70,
    Intent.FACILITY: 0.65,
    Intent.LPR: 0.72,
}


# ==================================================
# Top-1 / Top-2 score 차이 기준 (gap)
# ==================================================
GAP_THRESHOLD = 0.015


# ==================================================
# 키워드 기반 미세 보정 (임베딩 결과를 '뒤엎지 않음')
# ==================================================
KEYWORD_BOOST: Dict[Intent, List[str]] = {
    Intent.PAYMENT: ["결제", "카드", "정산", "요금", "주차비", "승인", "실패", "결제완료"],
    Intent.REGISTRATION: ["등록", "정기권", "무료", "방문", "방문자", "차량등록", "방문등록"],
    Intent.FACILITY: ["기계", "기기", "키오스크", "화면", "프린터", "차단기", "통신", "네트워크", "터미널", "차단봉"],
    Intent.LPR: ["번호판", "인식", "차번호", "카메라", "사진", "찍", "읽", "오인식", "다르", "틀리", "불일치", "바뀌", "다르게", "다른", "맞지"],
}

# 너무 키우면 rule-engine이 되므로 과하지 않게
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

    if not text or not str(text).strip():
        return IntentResult(intent=Intent.NONE, confidence=0.0)

    # 1) 입력 정규화 (짧은 문제 발화 보정)
    normalized_text = normalize_issue_text(str(text))
    print(f"[INTENT-EMBEDDING] Normalized input: {normalized_text}")

    # 2) 사용자 발화 임베딩
    user_vec = _EMBEDDING_MODEL.encode(
        normalized_text,
        normalize_embeddings=True,
    )

    # 3) intent별 유사도 계산
    scores: Dict[Intent, float] = {
        intent: _cosine(user_vec, proto_vec)
        for intent, proto_vec in _INTENT_EMBEDDINGS.items()
    }

    # 4) 키워드 기반 미세 보정
    for intent, keywords in KEYWORD_BOOST.items():
        if any(k in normalized_text for k in keywords):
            scores[intent] += KEYWORD_BOOST_SCORE

    # 5) 점수 정렬
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    for intent, score in sorted_scores:
        print(f"  - {intent.value:<15} : {score:.4f}")

    top_intent, top_score = sorted_scores[0]
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else -1.0
    gap = top_score - second_score
    threshold = INTENT_THRESHOLDS.get(top_intent, 0.7)

    confidence = round(float(top_score), 2)

    # 6) 자동 확정 판단
    if top_score >= threshold and gap >= GAP_THRESHOLD:
        print(f"[INTENT-EMBEDDING] ✅ CONFIRMED → {top_intent.value}")
        return IntentResult(intent=top_intent, confidence=confidence)

    print("[INTENT-EMBEDDING] ⚠️ AMBIGUOUS → NONE")
    return IntentResult(intent=Intent.NONE, confidence=confidence)
