from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from nlu.intent_schema import IntentResult, Intent

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CACHE_DIR = "models"

_MODEL = None
_TOKENIZER = None


# =========================
# Qwen 모델 로딩 (1회)
# =========================
def load_qwen():
    global _MODEL, _TOKENIZER

    if _MODEL is None:
        print("⏳ Qwen LLM 로딩 중...")

        _TOKENIZER = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )

        _MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="cpu",
            torch_dtype=torch.float32,
            cache_dir=CACHE_DIR,
        )

        _MODEL.eval()
        print("✅ Qwen LLM 로딩 완료")

    return _MODEL, _TOKENIZER


# =========================
# Intent 판별 (LLM 중심, 유사 발음 허용)
# =========================
def detect_intent_llm(text: str) -> IntentResult:
    model, tokenizer = load_qwen()

    messages = [
        {
            "role": "system",
            "content": (
                "너는 '주차장 출입 차단기 제어' 전용 AI다.\n\n"
                "너의 임무는 사용자의 발화가 아래 중 무엇인지 판단하는 것이다:\n"
                "- OPEN_GATE: 사용자가 직접 이동/출입/통과하기 위해 차단기를 열어달라는 요청\n"
                "- CLOSE_GATE: 차단기를 닫거나 막아달라는 요청\n"
                "- NONE: 차단기 제어와 무관하거나, 출입 맥락이 불명확한 발화\n\n"
                "⚠️ 매우 중요 (오타 / 유사 발음 처리 규칙):\n"
                "- 음성 인식 특성상 단어가 틀릴 수 있음을 반드시 고려한다\n"
                "- '차단기'와 발음·철자가 유사한 단어는 문맥상 차단기를 의미할 수 있다\n"
                "  예: 차당기, 하단기, 사단기, 처단기, 챠단기 등\n"
                "- 단어 자체보다 '출입/통과하려는 의도'가 핵심이다\n\n"
                "판단 기준:\n"
                "- 화자가 스스로 이동/출입/통과하려는 맥락이 명확하면 OPEN_GATE\n"
                "- 물리적 출입 장치를 올리거나/열어달라는 의미면 OPEN_GATE\n"
                "- 물리적 출입 장치를 내리거나/막아달라는 의미면 CLOSE_GATE\n"
                "- 내부 공간(방, 방문, 화장실, 집, 사무실 등)은 차단기와 무관하므로 제외한다\n"
                "- 잡담, 설명, 질문, 감정 표현, 욕구 표현은 NONE이다\n"
                "- 조금이라도 확신이 부족하면 반드시 NONE을 선택한다\n\n"
                "출력 규칙:\n"
                "- 반드시 JSON 형식만 출력한다\n"
                "- 형식: {\"intent\":\"OPEN_GATE|CLOSE_GATE|NONE\",\"confidence\":0.0~1.0}\n"
                "- confidence는 판단 확신 정도를 의미한다"
            ),
        },
        {
            "role": "user",
            "content": text,
        },
    ]

    # Qwen Instruct 전용 Chat Template
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=64,
            do_sample=False,  # 결정론적 (안전)
            eos_token_id=tokenizer.eos_token_id,
        )

    # 생성된 부분만 디코딩
    generated_ids = output_ids[0][input_ids.shape[-1]:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # JSON 안전 파싱
    try:
        start = decoded.index("{")
        end = decoded.index("}") + 1
        data = json.loads(decoded[start:end])
        return IntentResult(**data)
    except Exception:
        return IntentResult(intent=Intent.NONE, confidence=0.0)
