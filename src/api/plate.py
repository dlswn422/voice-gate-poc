from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np
import easyocr
import re
from datetime import datetime

from src.db.postgres import get_conn
from src.speech.tts import synthesize

router = APIRouter()

# =========================
# OCR 설정 (CPU ONLY)
# =========================
print("[PLATE] Initializing EasyOCR (CPU)")
reader = easyocr.Reader(["ko", "en"], gpu=False)

PLATE_REGEX = re.compile(r"\d{2,3}[가-힣]\d{4}")

COMMON_FIX = {
    "히": "허",
    "기": "가",
    "리": "라",
    "미": "마",
    "비": "바",
    "시": "사",
    "지": "자",
    "오": "호",
}

def normalize_plate(text: str) -> str:
    for wrong, right in COMMON_FIX.items():
        text = text.replace(wrong, right)
    return text


def extract_plate(image: np.ndarray) -> str | None:
    results = reader.readtext(image)

    for _, text, conf in results:
        cleaned = text.replace(" ", "")
        normalized = normalize_plate(cleaned)

        print(f"[PLATE] Detected='{cleaned}' → '{normalized}', conf={conf}")

        if PLATE_REGEX.match(normalized):
            print(f"[PLATE] ✅ Plate matched: {normalized}")
            return normalized

    print("[PLATE] ❌ No valid plate found")
    return None


# =========================
# DB 기반 입출차 판별 + 정책 처리
# =========================
def resolve_direction_and_process(plate: str):
    conn = get_conn()
    cur = conn.cursor()

    # 1️⃣ vehicle 조회 or 생성
    cur.execute("""
        SELECT id
        FROM vehicle
        WHERE plate_number = %s
        LIMIT 1
    """, (plate,))
    vehicle = cur.fetchone()

    if not vehicle:
        cur.execute("""
            INSERT INTO vehicle (
                plate_number,
                vehicle_type,
                created_at
            )
            VALUES (%s, %s, now())
            RETURNING id
        """, (plate, "NORMAL"))
        vehicle_id = cur.fetchone()["id"]
        conn.commit()
    else:
        vehicle_id = vehicle["id"]

    # 2️⃣ 활성 parking_session 조회
    cur.execute("""
        SELECT id
        FROM parking_session
        WHERE vehicle_id = %s
          AND exit_time IS NULL
        ORDER BY entry_time DESC
        LIMIT 1
    """, (vehicle_id,))
    session = cur.fetchone()

    # ==================================================
    # ENTRY
    # ==================================================
    if not session:
        cur.execute("""
            SELECT COUNT(*) AS count
            FROM parking_session
            WHERE exit_time IS NULL
        """)
        active_count = cur.fetchone()["count"]

        cur.execute("SELECT capacity FROM parking_lot LIMIT 1")
        capacity = cur.fetchone()["capacity"]

        is_full = active_count >= capacity

        if is_full:
            message = (
                "현재 주차장이 만차입니다.\n"
                "근처 이용 가능한 주차장을 안내해드릴게요."
            )
            barrier_open = False
        else:
            cur.execute("""
                INSERT INTO parking_session (
                    vehicle_id,
                    entry_time,
                    status,
                    created_at
                )
                VALUES (%s, %s, 'PARKED', now())
            """, (vehicle_id, datetime.utcnow()))
            conn.commit()

            message = "입차가 확인되었습니다.\n차단기가 열립니다."
            barrier_open = True

        conn.close()
        return {
            "direction": "ENTRY",
            "barrier_open": barrier_open,
            "message": message,
            "tts_url": synthesize(message),
            "end_session": True,
        }

    # ==================================================
    # EXIT
    # ==================================================
    session_id = session["id"]

    cur.execute("""
        SELECT id
        FROM payment
        WHERE parking_session_id = %s
          AND payment_status = 'PAID'
        LIMIT 1
    """, (session_id,))
    payment = cur.fetchone()

    # ✅ 결제 완료 → 즉시 출차
    if payment:
        cur.execute("""
            UPDATE parking_session
            SET exit_time = now(),
                status = 'EXITED'
            WHERE id = %s
        """, (session_id,))
        conn.commit()
        conn.close()

        message = "결제가 확인되었습니다.\n안전하게 출차하세요."
        return {
            "direction": "EXIT",
            "paid": True,
            "barrier_open": True,
            "message": message,
            "tts_url": synthesize(message),
            "end_session": True,
        }

    # ❌ 결제 미완료 → 상담 유도
    conn.close()

    message = (
        "아직 결제가 확인되지 않았어요.\n"
        "문제가 있다면 말씀해 주세요."
    )
    return {
        "direction": "EXIT",
        "paid": False,
        "barrier_open": False,
        "message": message,
        "tts_url": synthesize(message),
        "end_session": False,
    }


# =========================
# API Endpoint
# =========================
@router.post("/api/plate/recognize")
async def recognize_plate(image: UploadFile = File(...)):
    contents = await image.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return {"success": False, "error": "INVALID_IMAGE"}

    plate = extract_plate(img)
    if not plate:
        return {"success": False, "error": "PLATE_NOT_FOUND"}

    result = resolve_direction_and_process(plate)

    return {
        "success": True,
        "plate": plate,
        **result
    }