from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np
import easyocr
import re
from datetime import datetime
import requests

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
    "히": "허", "기": "가", "리": "라", "미": "마",
    "비": "바", "시": "사", "지": "자", "오": "호",
}

def normalize_plate(text: str) -> str:
    for wrong, right in COMMON_FIX.items():
        text = text.replace(wrong, right)
    return text


def extract_plate(image: np.ndarray) -> str | None:
    results = reader.readtext(image)
    for _, text, _ in results:
        cleaned = text.replace(" ", "")
        normalized = normalize_plate(cleaned)
        if PLATE_REGEX.match(normalized):
            print(f"[PLATE] ✅ Plate matched: {normalized}")
            return normalized
    return None


# =========================
# Kakao Local API
# =========================
KAKAO_REST_KEY = "ed8389b7bbe2ae8a2b8b3496e4919ecc"

def search_nearby_parking(lat: float, lng: float):
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {
        "Authorization": f"KakaoAK {KAKAO_REST_KEY}"
    }
    params = {
        "query": "주차장",
        "x": lng,
        "y": lat,
        "radius": 500,
        "size": 1,          # ✅ 1개만 조회
        "sort": "distance"
    }

    try:
        res = requests.get(url, headers=headers, params=params, timeout=3)
        res.raise_for_status()
        docs = res.json().get("documents", [])
        return docs[0] if docs else None
    except Exception as e:
        print("[KAKAO] ❌ parking search failed:", e)
        return None


# =========================
# 입출차 + 결제 정책 처리
# ALWAYS VOICE READY
# =========================
def resolve_direction_and_process(plate: str):
    conn = get_conn()
    cur = conn.cursor()

    # 1️⃣ vehicle 조회 or 생성
    cur.execute("""
        SELECT id, vehicle_type
        FROM vehicle
        WHERE plate_number = %s
        LIMIT 1
    """, (plate,))
    vehicle = cur.fetchone()

    if not vehicle:
        cur.execute("""
            INSERT INTO vehicle (plate_number, vehicle_type, created_at)
            VALUES (%s, 'NORMAL', now())
            RETURNING id, vehicle_type
        """, (plate,))
        vehicle = cur.fetchone()
        conn.commit()

    vehicle_id = vehicle["id"]
    vehicle_type = vehicle["vehicle_type"]

    # 2️⃣ 활성 세션 조회
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
    # 🚗 ENTRY
    # ==================================================
    if not session:
        cur.execute("""
            SELECT COUNT(*) AS count
            FROM parking_session
            WHERE exit_time IS NULL
        """)
        active_count = cur.fetchone()["count"]

        cur.execute("""
            SELECT capacity, latitude, longitude
            FROM parking_lot
            LIMIT 1
        """)
        lot = cur.fetchone()
        capacity = lot["capacity"]
        lat = lot["latitude"]
        lng = lot["longitude"]

        # 🚫 만차
        if active_count >= capacity:
            conn.close()

            parking = None
            if lat and lng:
                parking = search_nearby_parking(lat, lng)

            if parking:
                message = (
                    "현재 주차장이 만차입니다.\n"
                    f"근처 {parking['place_name']} 주차장을 추천드려요.\n"
                    f"도보 약 {parking['distance']}미터 거리입니다.\n"
                    "혹시 문제가 있으시면 말씀해주세요."
                )
            else:
                message = (
                    "현재 주차장이 만차입니다.\n"
                    "근처 주차장을 찾지 못했어요.\n"
                    "다른 문제가 있으시면 말씀해주세요."
                )

            return {
                "direction": "ENTRY",
                "can_pay": False,
                "barrier_open": False,
                "message": message,
                "tts_url": synthesize(message),
                "end_session": False,
            }

        # ✅ 입차 처리
        cur.execute("""
            INSERT INTO parking_session (
                vehicle_id,
                entry_time,
                status,
                created_at
            )
            VALUES (%s, %s, 'PARKED', now())
            RETURNING id
        """, (vehicle_id, datetime.utcnow()))
        session_id = cur.fetchone()["id"]

        payment_status = "FREE" if vehicle_type != "NORMAL" else "UNPAID"

        cur.execute("""
            INSERT INTO payment (
                parking_session_id,
                amount,
                payment_status,
                created_at
            )
            VALUES (%s, %s, %s, now())
            RETURNING id, payment_status
        """, (session_id, 0, payment_status))
        payment = cur.fetchone()

        conn.commit()
        conn.close()

        message = (
            "입차가 확인되었습니다.\n"
            "다른 문제가 있으시면 말씀해주세요."
        )

        return {
            "direction": "ENTRY",
            "parking_session_id": session_id,
            "payment_id": payment["id"],
            "payment_status": payment["payment_status"],
            "can_pay": False,   # 입차 시 결제 불가
            "barrier_open": True,
            "message": message,
            "tts_url": synthesize(message),
            "end_session": False,
        }

    # ==================================================
    # 🚙 EXIT
    # ==================================================
    session_id = session["id"]

    cur.execute("""
        SELECT id, payment_status
        FROM payment
        WHERE parking_session_id = %s
        LIMIT 1
    """, (session_id,))
    payment = cur.fetchone()

    # ✅ 이미 결제 완료 → 출차 허용
    if payment and payment["payment_status"] in ("PAID", "FREE"):
        cur.execute("""
            UPDATE parking_session
            SET exit_time = now(),
                status = 'EXITED'
            WHERE id = %s
        """, (session_id,))
        conn.commit()
        conn.close()

        message = (
            "출차가 확인되었습니다.\n"
            "다른 문제가 있으시면 말씀해주세요."
        )

        return {
            "direction": "EXIT",
            "parking_session_id": session_id,
            "payment_id": payment["id"],
            "payment_status": payment["payment_status"],
            "can_pay": False,   # ✅ 이미 결제됨
            "paid": True,
            "barrier_open": True,
            "message": message,
            "tts_url": synthesize(message),
            "end_session": False,
        }

    # ❗ 결제 안 된 출차 시도
    conn.close()
    message = (
        "아직 결제가 완료되지 않았어요.\n"
        "결제를 진행해주세요."
    )

    return {
        "direction": "EXIT",
        "parking_session_id": session_id,
        "payment_id": payment["id"],
        "payment_status": payment["payment_status"],
        "can_pay": True,    # ✅ 여기서만 결제 가능
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
