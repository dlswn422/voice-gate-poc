import re


def preprocess_user_text(raw: str) -> str:
    """
    STT 결과를 GPT에 보내기 전에 최소한 정리.
    - 공백 정리
    - 앞 추임새 제거
    - 너무 짧은 입력은 빈 문자열 반환
    """

    if not raw:
        return ""

    text = raw.strip()
    text = re.sub(r"\s+", " ", text)

    # 문장 앞 추임새 제거 (보수적)
    for f in ["어", "음", "그", "저기"]:
        text = re.sub(rf"^{f}\s+", "", text)

    text = text.strip()

    if len(text) < 2:
        return ""

    return text


def should_commit_final(text: str) -> bool:
    """
    FINAL 텍스트를 GPT로 보낼지 결정.
    너무 짧은 문장은 보내지 않음.
    """
    return bool(text) and len(text) >= 2