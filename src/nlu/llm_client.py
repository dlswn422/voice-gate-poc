from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

from nlu.intent_schema import IntentResult, Intent


# ==================================================
# 모델 설정
# ==================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CACHE_DIR = "models"

_MODEL = None
_TOKENIZER = None


# ==================================================
# Qwen 모델 로딩 (프로세스 내 1회)
# ==================================================

def load_qwen():
    """
    Qwen LLM 모델과 토크나이저를 프로세스 내에서 1회만 로딩한다.

    목적:
    - 매 발화마다 모델을 다시 로딩하지 않기 위함
    - 1차 의도 분류 지연 시간을 최소화
    """
    global _MODEL, _TOKENIZER

    if _MODEL is None:
        print("[LLM] Loading Qwen model")

        _TOKENIZER = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )

        _MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="cpu",      # 1차 분류용이므로 CPU 고정
            dtype=torch.float32,   # torch_dtype deprecated 대응
            cache_dir=CACHE_DIR,
        )

        _MODEL.eval()
        print("[LLM] Qwen model loaded")

    return _MODEL, _TOKENIZER


# ==================================================
# 1차 의도 분류 (주차장 CX 전용)
# ==================================================

def detect_intent_llm(text: str, debug: bool = True) -> IntentResult:
    """
    주차장 키오스크 환경에서 사용자의 발화를
    '상황 / 문제 / 문의' 관점에서 분류하는 1차 의도 분류 함수

    원칙:
    - 실행 판단은 하지 않는다
    - 제어 명령을 직접 수행하지 않는다
    - 애매한 발화는 confidence를 낮게 반환한다
    """
    model, tokenizer = load_qwen()

    if debug:
        print(f"[LLM] Input text: {text}")

    # ==================================================
    # LLM 프롬프트 (1차 분류 전용)
    # ==================================================
    messages = [
        {
            "role": "system",
            "content": (
                "너는 '주차장 키오스크 CX' 전용 음성 의도 분류 AI다.\n\n"
                "사용자의 발화를 아래 의도 중 하나로 분류하라.\n"
                "이 분류는 제어 명령이 아니라 상황, 문제, 문의를 구분하기 위한 것이다.\n\n"

                "[의도 목록]\n"
                "- ENTRY_FLOW_ISSUE: 입차 상황에서 차단기가 열리지 않음\n"
                "- EXIT_FLOW_ISSUE: 출차 상황에서 차단기가 열리지 않음\n"
                "- PAYMENT_ISSUE: 주차 요금 결제 관련 문제\n"
                "- REGISTRATION_ISSUE: 차량, 방문자, 번호판 등록 문제\n"
                "- TIME_ISSUE: 주차 시간, 무료 시간, 초과 시간 관련 문의 또는 문제\n"
                "- PRICE_INQUIRY: 주차 요금 또는 정산 금액에 대한 단순 문의\n"
                "- HOW_TO_EXIT: 출차 방법에 대한 문의\n"
                "- HOW_TO_REGISTER: 방문자 또는 차량 등록 방법 문의\n"
                "- COMPLAINT: 불만, 혼란, 짜증 등 감정 표현\n"
                "- NONE: 주차장 이용과 무관한 발화\n\n"

                "[분류 규칙]\n"
                "- '문 열어', '나가야 돼요'는 명령이 아니라 상황 설명으로 본다\n"
                "- 문제인지, 방법 문의인지, 감정 표현인지 구분하는 것이 핵심이다\n"
                "- 애매한 경우 가장 가까운 의도를 선택하되 confidence를 낮게 설정한다\n"
                "- 절대 실행 판단을 하지 말고 분류만 수행한다\n\n"

                "[출력 규칙]\n"
                "- 반드시 JSON만 출력한다\n"
                "- 형식: {\"intent\": \"INTENT_NAME\", \"confidence\": 0.0}\n"
            ),
        },
        {
            "role": "user",
            "content": text,
        },
    ]

    # ==================================================
    # 입력 토큰 생성
    # ==================================================
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    # ==================================================
    # LLM 추론 (결정적 출력)
    # ==================================================
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=64,
            do_sample=False,  # 1차 분류는 deterministic
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][input_ids.shape[-1]:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if debug:
        print("[LLM] Raw output:")
        print(decoded)

    # ==================================================
    # JSON 파싱 및 Enum 변환
    # ==================================================
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
                f"[LLM] Parsed result: intent={intent.name}, "
                f"confidence={confidence:.2f}"
            )

        return IntentResult(intent=intent, confidence=confidence)

    except Exception as e:
        if debug:
            print("[LLM] Parse error:", e)

        # 파싱 실패 시 안전한 기본값 반환
        return IntentResult(intent=Intent.NONE, confidence=0.0)
