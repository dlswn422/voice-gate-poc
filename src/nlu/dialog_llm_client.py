from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Tuple

import requests

from src.rag.manual_rag import ManualRAG


# ==================================================
# 환경/정책
# ==================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("DIALOG_MODEL", os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

DEFAULT_HARD_TURN_LIMIT = int(os.getenv("SECOND_STAGE_HARD_TURN_LIMIT", "6") or 6)
FAREWELL_TEXT = "이용해 주셔서 감사합니다. 안녕히 가세요."


# ==================================================
# 슬롯 정책
# ==================================================
SLOT_KEYS = ["symptom"]

# ✅ Intent 축소: PAYMENT / REGISTRATION / FACILITY / NONE
REQUIRED_SLOTS_BY_INTENT: Dict[str, List[str]] = {
    "PAYMENT": ["symptom"],
    "REGISTRATION": ["symptom"],
    "FACILITY": ["symptom"],
    "NONE": ["symptom"],
}


# ==================================================
# intent -> 메뉴얼 파일 매핑
# ==================================================
MANUAL_FILES_BY_INTENT: Dict[str, List[str]] = {
    "PAYMENT": ["payment_card_fail.md"],
    "REGISTRATION": ["visit_registration_fail.md"],
    "FACILITY": ["facility_issue.md"],
    "NONE": ["none_issue.md"],
}


# ==================================================
# 유틸
# ==================================================
def _norm_intent_name(x: Any) -> str:
    if not x:
        return "NONE"
    s = str(x).strip().upper()
    if s.startswith("INTENT."):
        s = s.split(".", 1)[-1]
    return s if s in REQUIRED_SLOTS_BY_INTENT else "NONE"


DONE_KEYWORDS = [
    "됐어요", "되었습니다", "해결", "괜찮아요",
    "그만", "종료", "끝", "마칠게",
    "고마워", "감사", "안녕",
]


def _normalize(text: str) -> str:
    return re.sub(r"[\s\.\,\!\?]+", "", (text or "").strip().lower())


def _is_done_utterance(text: str) -> bool:
    t = _normalize(text)
    return any(_normalize(k) in t for k in DONE_KEYWORDS)


def _manuals_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "manuals"))


def _read_manual(filename: str) -> str:
    path = os.path.join(_manuals_dir(), filename)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _ollama_chat(messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return ((data.get("message") or {}).get("content", "") or "").strip()


def _json_extract(text: str) -> Optional[dict]:
    if not text:
        return None
    s = text.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and j > i:
        snippet = s[i:j + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


# ==================================================
# 메뉴얼 후보 추출
# ==================================================
@dataclass
class ManualSolution:
    manual_file: str
    solution_text: str


_CASE_RE = re.compile(r"^###\s*CASE\s*([A-Z]{1,3}-\d+)\s*$", re.IGNORECASE | re.MULTILINE)


def _extract_case_solutions(md: str) -> List[str]:
    if not md:
        return []
    headers = [(m.group(1).strip().upper(), m.start()) for m in _CASE_RE.finditer(md)]
    if not headers:
        return []

    sols: List[str] = []
    for idx, (_cid, start) in enumerate(headers):
        end = headers[idx + 1][1] if idx + 1 < len(headers) else len(md)
        block = md[start:end].strip()

        m_solve = re.search(r"\*\*안내\s*문장\*\*\s*(.*)$", block, re.IGNORECASE | re.DOTALL)
        if not m_solve:
            continue
        solve = (m_solve.group(1) or "").strip()
        solve = "\n".join([ln.rstrip() for ln in solve.splitlines() if ln.strip()])
        if solve:
            sols.append(solve)
    return sols


def _extract_solve_template_solutions(md: str) -> List[str]:
    if not md:
        return []

    header_patterns = [
        r"##\s*해결\s*안내\s*문장\s*템플릿.*?$",
        r"##\s*SOLVE_TEMPLATE.*?$",
        r"##\s*해결\s*안내.*?$",
        r"##\s*조치\s*안내.*?$",
    ]

    start_idx = -1
    for hp in header_patterns:
        m = re.search(hp, md, re.IGNORECASE | re.MULTILINE)
        if m:
            start_idx = m.end()
            break
    if start_idx == -1:
        return []

    rest = md[start_idx:]
    m_next = re.search(r"^\s*##\s+", rest, re.MULTILINE)
    section = rest[: m_next.start()] if m_next else rest

    bullets: List[str] = []
    buf: List[str] = []
    in_bullet = False

    for ln in section.splitlines():
        s = ln.rstrip().strip()

        if re.match(r"^-\s+", s):
            if buf:
                bullets.append("\n".join(buf).strip())
                buf = []
            in_bullet = True
            buf.append(s[2:].strip())
            continue

        if in_bullet:
            if s == "":
                if buf:
                    bullets.append("\n".join(buf).strip())
                buf = []
                in_bullet = False
                continue
            buf.append(s)
            continue

    if buf:
        bullets.append("\n".join(buf).strip())

    bullets = [b for b in bullets if len(b) >= 6]
    return bullets


def _gather_solutions_for_intent(intent_name: str) -> List[ManualSolution]:
    intent_name = _norm_intent_name(intent_name)
    files = MANUAL_FILES_BY_INTENT.get(intent_name, [])
    out: List[ManualSolution] = []

    for fn in files:
        md = _read_manual(fn)
        if not md:
            continue

        case_sols = _extract_case_solutions(md)
        if case_sols:
            for s in case_sols:
                out.append(ManualSolution(manual_file=fn, solution_text=s))
            continue

        tmpl_sols = _extract_solve_template_solutions(md)
        for s in tmpl_sols:
            out.append(ManualSolution(manual_file=fn, solution_text=s))

    uniq: List[ManualSolution] = []
    seen = set()
    for it in out:
        key = (it.manual_file, it.solution_text)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)
    return uniq


# ==================================================
# 솔루션 선택기 (LLM은 선택만)
# ==================================================
SYSTEM_PROMPT_SELECT_SOLUTION = """
너는 '솔루션 선택기'다.
사용자 symptom(raw 문장)과 후보 솔루션들을 비교해서,
가장 적합한 솔루션 1개를 '인덱스'로 선택한다.

절대 규칙:
- 솔루션 문장을 새로 만들거나 바꾸지 마라.
- 후보에 없는 문장은 절대 출력하지 마라.
- 출력은 JSON 1개만.
형식:
{"action":"SOLVE","index":0,"confidence":0.0~1.0}

선택 기준:
- symptom 문장에 들어있는 핵심 단어/표현과 가장 잘 맞는 후보
- intent와 문맥이 맞는 후보(결제/등록/시설 등)
""".strip()


def _llm_select_solution_index(intent_name: str, symptom: str, sols: List[ManualSolution]) -> Tuple[Optional[int], float]:
    if len(sols) == 1:
        return 0, 0.99

    candidates = [{"i": i, "manual": s.manual_file, "text": s.solution_text} for i, s in enumerate(sols)]
    prompt = (
        f"[intent]={intent_name}\n"
        f"[symptom_raw]={symptom}\n"
        f"[candidates]={json.dumps(candidates, ensure_ascii=False)}\n\n"
        "가장 적합한 후보의 i 값을 선택해라.\n"
    )
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT_SELECT_SOLUTION},
        {"role": "user", "content": prompt},
    ]
    content = _ollama_chat(msgs, temperature=0.0)
    obj = _json_extract(content) or {}

    action = str(obj.get("action", "")).strip().upper()
    conf = float(obj.get("confidence", 0.8) or 0.8)

    if action != "SOLVE":
        return None, conf

    try:
        idx = int(obj.get("index"))
    except Exception:
        return None, conf

    if 0 <= idx < len(sols):
        return idx, conf
    return None, conf


# ==================================================
# 결과 타입
# ==================================================
DialogAction = Literal["SOLVE", "DONE", "ESCALATE_DONE"]


@dataclass
class DialogResult:
    reply: str
    action: DialogAction = "SOLVE"
    slots: Optional[Dict[str, Any]] = None
    pending_slot: Optional[str] = None
    new_intent: Optional[str] = None
    confidence: float = 0.75


# ==================================================
# 메인
# ==================================================
def dialog_llm_chat(
    user_text: str,
    *,
    history: Optional[List[Dict[str, str]]] = None,
    context: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> DialogResult:
    if _is_done_utterance(user_text):
        return DialogResult(
            reply=FAREWELL_TEXT,
            action="DONE",
            confidence=1.0,
            slots=(context or {}).get("slots") or {},
        )

    ctx = context or {}
    hard_limit = int(ctx.get("hard_turn_limit", DEFAULT_HARD_TURN_LIMIT) or DEFAULT_HARD_TURN_LIMIT)
    turn_count_user = int(ctx.get("turn_count_user", 0) or 0)

    current_intent = _norm_intent_name(ctx.get("intent"))
    slots = ctx.get("slots") if isinstance(ctx.get("slots"), dict) else {}

    if turn_count_user >= hard_limit:
        return DialogResult(
            reply="여러 번 확인했지만 현재 정보로는 문제 상황을 정확히 특정하기 어렵습니다. 관리자를 호출해 도움을 받아주세요. " + FAREWELL_TEXT,
            action="ESCALATE_DONE",
            confidence=1.0,
            slots=slots,
            new_intent=current_intent,
        )

    # ✅ symptom 매 턴 덮어쓰기
    slots["symptom"] = user_text.strip()
    symptom = str(slots.get("symptom") or "").strip()

    # ✅ PAYMENT면 payment_ctx 힌트를 symptom에만 부착(선택 품질 개선용)
    symptom_for_select = symptom
    if current_intent == "PAYMENT":
        pctx = ctx.get("payment_ctx") if isinstance(ctx.get("payment_ctx"), dict) else {}
        if pctx:
            symptom_for_select = (
                f"{symptom}\n"
                f"(결제로그: result={pctx.get('log_result')}, reason={pctx.get('log_reason')}, payment_status={pctx.get('payment_status')})"
            )

    sols = _gather_solutions_for_intent(current_intent)

    if debug:
        print(f"[SOLVE] intent={current_intent} solutions={len(sols)} symptom={symptom_for_select}")

    if not sols:
        rag = ManualRAG()
        query = f"{current_intent} | symptom: {symptom_for_select}"
        try:
            hits = rag.retrieve(query=query, preferred_docs=None, hard_filter=False, debug=debug)
        except Exception as e:
            if debug:
                print(f"[RAG] failed: {e}")
            hits = []

        if hits:
            top = (getattr(hits[0], "text", "") or "").strip()
            if top:
                return DialogResult(reply=top, action="SOLVE", slots=slots, confidence=0.85, new_intent=current_intent)

        return DialogResult(
            reply="현재 메뉴얼에서 해당 상황의 해결 안내를 찾지 못했습니다. 관리자를 호출해 도움을 받아주세요. " + FAREWELL_TEXT,
            action="ESCALATE_DONE",
            slots=slots,
            confidence=1.0,
            new_intent=current_intent,
        )

    idx, conf = _llm_select_solution_index(current_intent, symptom_for_select, sols)

    if idx is None:
        return DialogResult(
            reply="현재 입력 정보로는 문제 유형을 정확히 특정하기 어렵습니다. 관리자를 호출해 도움을 받아주세요. " + FAREWELL_TEXT,
            action="ESCALATE_DONE",
            slots=slots,
            confidence=1.0,
            new_intent=current_intent,
        )

    chosen = sols[idx]
    return DialogResult(
        reply=chosen.solution_text.strip(),
        action="SOLVE",
        slots=slots,
        new_intent=current_intent,
        confidence=min(0.99, max(0.8, conf)),
    )
