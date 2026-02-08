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
    print("[PLATE] Running OCR...")
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
# DB 기반 입출차 판별 + 입차 INSERT
# =========================
def resolve_direction_and_insert(plate: str):
    print(f"[LOGIC] Resolve direction for plate={plate}")

    conn = get_conn()
    cur = conn.cursor()

    # 1️⃣ vehicle 조회
    cur.execute("""
        SELECT id
        FROM vehicle
        WHERE plate_number = %s
        LIMIT 1
    """, (plate,))
    vehicle = cur.fetchone()

    # vehicle 없으면 생성
    if not vehicle:
        print("[DB] Vehicle not found → creating new vehicle")
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

    # =========================
    # ENTRY
    # =========================
    if not session:
        print("[LOGIC] ENTRY 판단")

        cur.execute("""
            SELECT COUNT(*) AS count
            FROM parking_session
            WHERE exit_time IS NULL
        """)
        active_count = cur.fetchone()["count"]

        cur.execute("SELECT capacity FROM parking_lot LIMIT 1")
        capacity = cur.fetchone()["capacity"]

        is_full = active_count >= capacity
        print(f"[LOGIC] active={active_count}, capacity={capacity}, is_full={is_full}")

        if is_full:
            message = (
                "현재 주차장이 만차입니다.\n"
                "근처 이용 가능한 주차장을 안내해드릴게요."
            )
        else:
            print("[DB] INSERT parking_session (ENTRY)")
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

        conn.close()

        tts_url = synthesize(message)

        return {
            "direction": "ENTRY",
            "is_full": is_full,
            "message": message,
            "tts_url": tts_url,
            "session": {
                "exists": False,
                "paid": False
            }
        }

    # =========================
    # EXIT
    # =========================
    session_id = session["id"]
    print(f"[LOGIC] EXIT 판단, session_id={session_id}")

    cur.execute("""
        SELECT id
        FROM payment
        WHERE parking_session_id = %s
          AND status = 'PAID'
        LIMIT 1
    """, (session_id,))
    payment = cur.fetchone()

    conn.close()

    message = "출차 차량으로 확인되었습니다.\n문제가 있으시면 말씀해 주세요."
    tts_url = synthesize(message)

    return {
        "direction": "EXIT",
        "is_full": False,
        "message": message,
        "tts_url": tts_url,
        "session": {
            "exists": True,
            "paid": bool(payment)
        }
    }


# =========================
# API Endpoint
# =========================
@router.post("/api/plate/recognize")
async def recognize_plate(image: UploadFile = File(...)):
    print("\n==============================")
    print("[API] /api/plate/recognize called")

    contents = await image.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        print("[API] ❌ Image decode failed")
        return {"success": False, "error": "INVALID_IMAGE"}

    plate = extract_plate(img)

    if not plate:
        return {"success": False, "error": "PLATE_NOT_FOUND"}

    result = resolve_direction_and_insert(plate)

    response = {
        "success": True,
        "plate": plate,
        **result
    }

    print(f"[API] ✅ Response: {response}")
    print("==============================\n")

    return response
