from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re
import traceback
from typing import Optional

from nlu.intent_schema import IntentResult, Intent

# ==================================================
# 모델 설정
# ==================================================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CACHE_DIR = "models"

_MODEL: Optional[AutoModelForCausalLM] = None
_TOKENIZER: Optional[AutoTokenizer] = None

_JSON_RE = re.compile(r"\{[\s\S]*\}")


# ==================================================
# Qwen 모델 로딩 (프로세스 내 1회)
# ==================================================
def load_qwen():
    global _MODEL, _TOKENIZER

    if _MODEL is None or _TOKENIZER is None:
        print("[LLM] Loading Qwen model")

        _TOKENIZER = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )

        # ✅ pad_token이 없으면 generate에서 문제나는 경우가 있어 eos로 보정
        if _TOKENIZER.pad_token_id is None:
            _TOKENIZER.pad_token = _TOKENIZER.eos_token

        _MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            device_map="cpu",            # 1차 의도 분류용(안정)
            torch_dtype=torch.float32,   # ✅ dtype 금지! torch_dtype 사용
            cache_dir=CACHE_DIR,
            low_cpu_mem_usage=True,
        )

        _MODEL.eval()
        print("[LLM] Qwen model loaded")

    return _MODEL, _TOKENIZER


# ==================================================
# JSON 추출 유틸
# ==================================================
def _extract_json(text: str) -> dict:
    text = (text or "").strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return json.loads(text[start:end])

    m = _JSON_RE.search(text)
    if m:
        return json.loads(m.group(0))

    raise ValueError(f"JSON not found in output: {text}")


# ==================================================
# 1차 의도 분류 (INTENT ONLY)
# ==================================================
def detect_intent_llm(text: str, debug: bool = True) -> IntentResult:
    """
    [역할]
    - 사용자 발화의 '의도(intent)'만 분류한다
    - 실행 판단 ❌
    - confidence 판단 ❌ (AppEngine에서 계산)

    [중요]
    - DB 구조 유지를 위해 confidence는 항상 0.0으로 반환
    """
    model, tokenizer = load_qwen()

    if debug:
        print(f"[LLM] Input text: {text}")

    messages = [
        {
            "role": "system",
            "content": (
                "너는 '주차장 키오스크 CX' 전용 음성 의도 분류 AI다.\n\n"
                "사용자의 발화를 아래 의도 중 하나로 분류하라.\n"
                "이 분류는 실행이나 제어 판단이 아니라,\n"
                "사용자가 처한 상황 / 문제 / 문의 유형을 구분하기 위한 것이다.\n\n"
                "[의도 목록]\n"
                "- ENTRY_FLOW_ISSUE\n"
                "- EXIT_FLOW_ISSUE\n"
                "- PAYMENT_ISSUE\n"
                "- REGISTRATION_ISSUE\n"
                "- TIME_ISSUE\n"
                "- PRICE_INQUIRY\n"
                "- HOW_TO_EXIT\n"
                "- HOW_TO_REGISTER\n"
                "- COMPLAINT\n"
                "- NONE\n\n"
                "[분류 규칙]\n"
                "- '문 열어', '나가야 돼요' 같은 표현은 명령이 아니라 상황 설명으로 본다\n"
                "- 문제 상황인지, 방법 문의인지, 감정 표현인지 구분한다\n"
                "- 입차 문제와 출차 문제를 문맥상 명확히 구분한다\n"
                "- 애매하더라도 반드시 가장 가까운 하나의 의도를 선택한다\n"
                "- 실행 여부나 해결 방법은 절대 판단하지 않는다\n\n"
                "[출력 규칙]\n"
                "- 반드시 JSON만 출력한다\n"
                "- 형식: {\"intent\": \"INTENT_NAME\"}\n"
                "- 다른 문장/설명 금지\n"
            ),
        },
        {"role": "user", "content": text},
    ]

    try:
        # ============================================================
        # ✅ 핵심 수정: tokenize=True로 바로 텐서 만들지 말고
        # 1) prompt 문자열 생성(tokenize=False)
        # 2) tokenizer()로 텐서화(return_tensors="pt") → 항상 Tensor 보장
        # ============================================================
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
        )
        model_inputs = {k: v.to("cpu") for k, v in model_inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **model_inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = model_inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0, input_len:]
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if debug:
            print("[LLM] Raw output:")
            print(decoded)

        data = _extract_json(decoded)
        intent_str = str(data.get("intent", "NONE")).strip()

        try:
            intent = Intent(intent_str)
        except Exception:
            intent = Intent.NONE

        if debug:
            print(f"[LLM] Parsed intent: {intent.name}")

        return IntentResult(intent=intent, confidence=0.0)

    except Exception as e:
        if debug:
            print("[LLM] Inference/Parse error:", repr(e))
            traceback.print_exc()

        return IntentResult(intent=Intent.NONE, confidence=0.0)
