from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np
import easyocr
import re
from datetime import datetime
import requests
import uuid
import os

from src.db.postgres import get_conn
from src.speech.tts import synthesize
from src.storage import upload_image   # ✅ 정상 사용

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


# =========================
# 번호판 영역 탐지
# =========================
def detect_plate_region(img: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 30, 200)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ratio = w / float(h)
        if 2.0 < ratio < 6.0 and w > 120:
            candidates.append((x, y, w, h))

    if not candidates:
        return None

    x, y, w, h = max(candidates, key=lambda b: b[2] * b[3])
    return img[y:y+h, x:x+w]


# =========================
# OCR 추출
# =========================
def extract_plate(image: np.ndarray) -> tuple[str | None, float]:
    results = reader.readtext(image)
    best = None

    for _, text, conf in results:
        cleaned = text.replace(" ", "")
        normalized = normalize_plate(cleaned)
        if PLATE_REGEX.fullmatch(normalized):
            if not best or conf > best[1]:
                best = (normalized, conf)

    return best if best else (None, 0.0)


# =========================
# Kakao Local API (유지)
# =========================
KAKAO_REST_KEY = os.environ.get(
    "KAKAO_REST_KEY",
    "ed8389b7bbe2ae8a2b8b3496e4919ecc"
)

def search_nearby_parking(lat: float, lng: float):
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_KEY}"}
    params = {
        "query": "주차장",
        "x": lng,
        "y": lat,
        "radius": 500,
        "size": 1,
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
# 입출차 + 결제 처리 (DB 스키마 기준)
# =========================
def resolve_direction_and_process(
    plate: str,
    confidence: float,
    image_url: str,
):
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

    # =========================
    # ENTRY
    # =========================
    if not session:
        cur.execute("""
            INSERT INTO parking_session (
                vehicle_id,
                entry_time,
                status,
                entry_image_url,
                entry_image_captured_at,
                created_at
            )
            VALUES (%s, now(), 'PARKED', %s, now(), now())
            RETURNING id
        """, (vehicle_id, image_url))
        session_id = cur.fetchone()["id"]

        payment_status = "FREE" if vehicle_type != "NORMAL" else "UNPAID"

        cur.execute("""
            INSERT INTO payment (
                parking_session_id,
                amount,
                payment_status,
                created_at
            )
            VALUES (%s, 0, %s, now())
            RETURNING id, payment_status
        """, (session_id, payment_status))
        payment = cur.fetchone()

        conn.commit()
        conn.close()

        message = "입차가 확인되었습니다.\n다른 문제가 있으시면 말씀해주세요."

        return {
            "direction": "ENTRY",
            "parking_session_id": session_id,
            "payment_id": payment["id"],
            "payment_status": payment["payment_status"],
            "barrier_open": True,
            "message": message,
            "tts_url": synthesize(message),
            "end_session": False,
        }

    # =========================
    # EXIT
    # =========================
    session_id = session["id"]

    cur.execute("""
        SELECT id, payment_status
        FROM payment
        WHERE parking_session_id = %s
        LIMIT 1
    """, (session_id,))
    payment = cur.fetchone()

    if payment["payment_status"] not in ("PAID", "FREE"):
        conn.close()
        message = "아직 결제가 완료되지 않았어요.\n결제를 진행해주세요."
        return {
            "direction": "EXIT",
            "can_pay": True,
            "barrier_open": False,
            "message": message,
            "tts_url": synthesize(message),
            "end_session": False,
        }

    cur.execute("""
        UPDATE parking_session
        SET exit_time = now(),
            status = 'EXITED',
            exit_image_url = %s,
            exit_image_captured_at = now()
        WHERE id = %s
    """, (image_url, session_id))

    conn.commit()
    conn.close()

    message = "출차가 확인되었습니다.\n안전 운행하세요."

    return {
        "direction": "EXIT",
        "parking_session_id": session_id,
        "barrier_open": True,
        "message": message,
        "tts_url": synthesize(message),
        "end_session": False,
    }


# =========================
# API Endpoint
# =========================
@router.post("/api/plate/recognize")
async def recognize_plate(
    image: UploadFile = File(...)
):
    contents = await image.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return {"success": False, "error": "INVALID_IMAGE"}

    # 🔥 이미지 업로드 (고정 경로)
    image_key = f"parking/raw/{uuid.uuid4()}.jpg"
    image_url = upload_image(contents, image_key)

    plate_img = detect_plate_region(img)
    if plate_img is None:
        return {
            "success": False,
            "error": "PLATE_NOT_FOUND",
            "image_url": image_url
        }

    plate, confidence = extract_plate(plate_img)
    if not plate or confidence < 0.6:
        return {
            "success": False,
            "error": "LOW_CONFIDENCE",
            "confidence": confidence,
            "image_url": image_url
        }

    result = resolve_direction_and_process(
        plate=plate,
        confidence=confidence,
        image_url=image_url,
    )

    return {
        "success": True,
        "plate": plate,
        "confidence": confidence,
        "image_url": image_url,
        **result
    }
