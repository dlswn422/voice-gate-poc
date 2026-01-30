from nlu.llm_client import detect_intent_llm
from nlu.intent_schema import Intent
from engine.intent_logger import log_intent, log_dialog  # ??(異붽?) 2李????濡쒓렇 ???from nlu.dialog_llm_client import dialog_llm_chat        # ??(異붽?) 2李?Llama ?몄텧

import uuid
import time


# --------------------------------------------------
# ?뺤콉 ?ㅼ젙
# --------------------------------------------------

CONFIDENCE_THRESHOLD = 0.75
SITE_ID = "parkassist_local"


class AppEngine:
    """
    二쇱감???ㅼ삤?ㅽ겕 CX??App Engine

    ?곹깭:
    - FIRST_STAGE  : 1李??섎룄 遺꾨쪟 ?④퀎
    - SECOND_STAGE : 2李??곷떞(?쇰쭏) ?④퀎
    """

    def __init__(self):
        # ?뵎 ?곹깭 愿由?(?먮낯 ?좎?)
        self.state = "FIRST_STAGE"
        self.session_id = None

        # ??(異붽?) 2李????濡쒓렇 ?곌껐???꾪븳 媛믩뱾
        self.intent_log_id = None          # 1李?intent_logs PK (dialog_logs FK)
        self.dialog_turn_index = 0         # dialog_logs turn_index (1遺??利앷?)

    # ==================================================
    # ?뵩 confidence 怨꾩궛 濡쒖쭅 (?먮낯 ?좎?)
    # ==================================================

    def calculate_confidence(self, text: str, intent: Intent) -> float:
        score = 0.0
        text = text.strip()

        KEYWORDS = {
            Intent.EXIT_FLOW_ISSUE: ["異쒖감", "?섍?", "李⑤떒湲?, "???대젮"],
            Intent.ENTRY_FLOW_ISSUE: ["?낆감", "?ㅼ뼱媛", "李⑤떒湲?, "???대젮"],
            Intent.PAYMENT_ISSUE: ["寃곗젣", "?붽툑", "移대뱶", "?뺤궛"],
            Intent.TIME_ISSUE: ["?쒓컙", "臾대즺", "珥덇낵"],
            Intent.PRICE_INQUIRY: ["?쇰쭏", "?붽툑", "媛寃?],
            Intent.HOW_TO_EXIT: ["?대뼸寃?, "異쒖감", "?섍?"],
            Intent.HOW_TO_REGISTER: ["?깅줉", "?대뵒", "諛⑸쾿"],
        }

        hits = sum(1 for k in KEYWORDS.get(intent, []) if k in text)

        if hits >= 2:
            score += 0.45
        elif hits == 1:
            score += 0.30
        else:
            score += 0.10

        if len(text) < 3:
            score += 0.05
        elif any(f in text for f in ["??, "??, "..."]):
            score += 0.10
        else:
            score += 0.25

        INTENT_RISK_WEIGHT = {
            Intent.HOW_TO_EXIT: 1.0,
            Intent.PRICE_INQUIRY: 1.0,
            Intent.TIME_ISSUE: 0.9,
            Intent.EXIT_FLOW_ISSUE: 0.7,
            Intent.ENTRY_FLOW_ISSUE: 0.7,
            Intent.PAYMENT_ISSUE: 0.7,
            Intent.REGISTRATION_ISSUE: 0.6,
            Intent.COMPLAINT: 0.5,
        }

        score *= INTENT_RISK_WEIGHT.get(intent, 0.6)
        return round(min(score, 1.0), 2)

    # ==================================================
    # ?럺截?STT ?띿뒪??泥섎━ ?뷀듃由ы룷?명듃
    # ==================================================

    def handle_text(self, text: str):
        if not text or not text.strip():
            return

        print("=" * 50)
        print(f"[ENGINE] State={self.state}")
        print(f"[ENGINE] Text={text}")

        # ==================================================
        # ?윟 2李??곷떞 ?④퀎 (?곹깭 ?좎?, ??붾뒗 怨꾩냽 ?쇰쭏濡?
        # ==================================================
        if self.state == "SECOND_STAGE":
            try:
                # ??(異붽?) ?ъ슜??諛쒗솕 濡쒓렇
                self.dialog_turn_index += 1
                log_dialog(
                    intent_log_id=self.intent_log_id,
                    session_id=self.session_id,
                    role="user",
                    content=text,
                    model="stt",
                    turn_index=self.dialog_turn_index,
                )

                # ??(異붽?) ?쇰쭏 ?몄텧
                res = dialog_llm_chat(text, history=None, context={"session_id": self.session_id}, debug=True)
                llama_response = res.reply if hasattr(res, "reply") else str(res)

                # ??(異붽?) ?쇰쭏 ?묐떟 濡쒓렇
                self.dialog_turn_index += 1
                log_dialog(
                    intent_log_id=self.intent_log_id,
                    session_id=self.session_id,
                    role="assistant",
                    content=llama_response,
                    model="llama-3.1-8b",
                    turn_index=self.dialog_turn_index,
                )

                print(f"[DIALOG] {llama_response}")

            except Exception as e:
                # ??(異붽?) STT 肄쒕갚??二쎌? ?딅룄濡?2李??덉쇅???붿쭊?먯꽌 泥섎━
                print(f"[ENGINE] 2nd-stage failed: {repr(e)}")

            print("=" * 50)
            return

        # ==================================================
        # ?뵷 1李??섎룄 遺꾨쪟 ?④퀎 (?먮낯 ?좎?)
        # ==================================================
        try:
            result = detect_intent_llm(text)
        except Exception as e:
            print("[ENGINE] LLM inference failed:", e)
            print("=" * 50)
            return

        result.confidence = self.calculate_confidence(text=text, intent=result.intent)

        print(f"[ENGINE] Intent={result.intent.name}, confidence={result.confidence:.2f}")

        # ??(蹂寃? 1李?intent 濡쒓렇 ?곸옱 + PK 諛섑솚媛????(2李?dialog_logs FK濡??ъ슜)
        self.intent_log_id = log_intent(
            utterance=text,
            predicted_intent=result.intent.value,
            predicted_confidence=result.confidence,
            source="kiosk",
            site_id=SITE_ID,
        )
        print(f"[ENGINE] intent_log_id={self.intent_log_id}")  # ??(異붽?) NULL ?щ? ?뺤씤??
        # ??(異붽?) intent_log_id媛 ?놁쑝硫?dialog_logs NOT NULL 源⑥?誘濡?2李⑤줈 紐?媛?        if self.intent_log_id is None:
            print("[ENGINE] intent_log_id is None ??skip llama fallback")
            print("=" * 50)
            return

        if result.intent == Intent.NONE:
            print("[ENGINE] Decision: irrelevant utterance")
            print("=" * 50)
            return

        # ==================================================
        # confidence 湲곗? 遺꾧린: ?ш린留??쇰쭏 遺숈엫 (?붽뎄?ы빆 ?듭떖)
        # ==================================================
        if result.confidence < CONFIDENCE_THRESHOLD:
            print("[ENGINE] Decision: low confidence ??llama fallback")

            # ??(異붽?) ?몄뀡 id ?앹꽦(?ъ슜?먮퀎 ???異붿쟻)
            self.state = "SECOND_STAGE"
            self.session_id = str(uuid.uuid4())   # ???붽뎄?ы빆: uuid 湲곕컲 session_id
            self.dialog_turn_index = 0            # ??(異붽?) ??珥덇린??
            print(f"[ENGINE] Session started: {self.session_id}")
            print("[ENGINE] Llama will handle this utterance (logging dialog)")

            # ??(異붽?) ?쒖껀 諛쒗솕?앸룄 諛붾줈 2李⑤줈 泥섎━(濡쒓렇 + ?쇰쭏?묐떟)
            self.handle_text(text)

            print("=" * 50)
            return

        print("[ENGINE] Decision: passed 1st-stage classification")
        print("[ENGINE] Action: defer execution to next stage")
        print("=" * 50)

    # ==================================================
    # ?뵚 ?곷떞 醫낅즺 ???몄텧
    # ==================================================

    def end_second_stage(self):
        print(f"[ENGINE] Session ended: {self.session_id}")
        self.state = "FIRST_STAGE"
        self.session_id = None
        self.intent_log_id = None          # ??(異붽?) ?몄뀡 醫낅즺 ??珥덇린??        self.dialog_turn_index = 0         # ??(異붽?) 珥덇린??
