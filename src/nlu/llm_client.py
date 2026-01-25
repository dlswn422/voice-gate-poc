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
# Intent 판별 (확장 버전)
# =========================
def detect_intent_llm(text: str, debug: bool = True) -> IntentResult:
    model, tokenizer = load_qwen()

    if debug:
        print(f"📥 [LLM INPUT] {text}")

    messages = [
        {
            "role": "system",
            "content": (
                "너는 '주차장 출입 차단기 제어' 전용 AI다.\n\n"
                "사용자의 발화를 아래 의도 중 하나로 분류하라:\n\n"
                "- OPEN_GATE: 지금 당장 차단기를 열어달라는 명시적 요청\n"
                "- CLOSE_GATE: 차단기를 닫거나 막아달라는 명시적 요청\n"
                "- HELP_REQUEST: 문이 안 열림, 결제 실패, 등록 오류 등 문제 상황 설명\n"
                "- INFO_REQUEST: 방문 등록 방법, 절차, 사용법을 묻는 질문\n"
                "- NONE: 차단기 제어와 무관한 발화\n\n"
                "⚠️ 매우 중요:\n"
                "- OPEN_GATE는 '열어줘', '올려줘', '통과할게요' 등 직접 명령일 때만 선택한다\n"
                "- '문이 안 열려요', '방문등록 했는데 안돼요'는 OPEN_GATE가 아니라 HELP_REQUEST다\n"
                "- 질문형 문장은 INFO_REQUEST로 분류한다\n"
                "- 애매하면 반드시 NONE 또는 HELP_REQUEST를 선택한다\n\n"
                "출력 규칙:\n"
                "- 반드시 JSON만 출력한다\n"
                "- 형식: {\"intent\":\"OPEN_GATE|CLOSE_GATE|HELP_REQUEST|INFO_REQUEST|NONE\",\"confidence\":0.0~1.0}"
            ),
        },
        {
            "role": "user",
            "content": text,
        },
    ]

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
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][input_ids.shape[-1]:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if debug:
        print("🧾 [LLM RAW OUTPUT]")
        print(decoded)

    # =========================
    # JSON 파싱 + Enum 변환
    # =========================
    try:
        start = decoded.find("{")
        end = decoded.rfind("}") + 1
        data = json.loads(decoded[start:end])

        intent_str = data.get("intent", "NONE")
        confidence = float(data.get("confidence", 0.0))

        try:
            intent = Intent(intent_str)
        except ValueError:
            intent = Intent.NONE

        confidence = max(0.0, min(confidence, 1.0))

        if debug:
            print(
                f"📊 [LLM PARSED] intent={intent.name}, "
                f"confidence={confidence:.2f}"
            )

        return IntentResult(intent=intent, confidence=confidence)

    except Exception as e:
        if debug:
            print("❌ [LLM PARSE ERROR]", e)
        return IntentResult(intent=Intent.NONE, confidence=0.0)
