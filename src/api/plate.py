from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np
import easyocr
import re
from datetime import datetime
import uuid

from psycopg2.extras import RealDictCursor

from src.db.postgres import get_conn
from src.storage import upload_image

router = APIRouter()

# =========================
# OCR 설정
# =========================
reader = easyocr.Reader(["ko", "en"], gpu=False)

PLATE_REGEX = re.compile(r"\d{2,3}[가-힣]\d{4}")

COMMON_FIX = {
    "히": "허", "기": "가", "리": "라", "미": "마",
    "비": "바", "시": "사", "지": "자", "오": "호",
}

VEHICLE_TYPE_LABEL = {
    "NORMAL": "일반 차량",
    "VISITOR": "방문 차량",
    "MEMBER": "정기권 차량",
}


def normalize_plate(text: str) -> str:
    for wrong, right in COMMON_FIX.items():
        text = text.replace(wrong, right)
    return text


# =========================
# 번호판 영역 탐지
# =========================
def detect_plate_region(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 30, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ratio = w / float(h)
        if 2.0 < ratio < 6.0 and w > 120:
            candidates.append((x, y, w, h))

    if not candidates:
        return None

    x, y, w, h = max(candidates, key=lambda b: b[2] * b[3])
    return img[y:y + h, x:x + w]


# =========================
# OCR 추출
# =========================
def extract_plate(image: np.ndarray):
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
# 입출차 + 만차 처리 (🔥 FINAL)
# =========================
def resolve_direction_and_process(
    plate: str,
    image_url: str,
    parking_lot_id: str,
):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # 🔒 주차장 row lock
    cur.execute(
        """
        SELECT capacity
        FROM parking_lot
        WHERE id = %s
        FOR UPDATE
        """,
        (parking_lot_id,),
    )
    lot = cur.fetchone()
    capacity = lot["capacity"]

    # 차량 조회 / 생성
    cur.execute(
        """
        SELECT id, vehicle_type
        FROM vehicle
        WHERE plate_number = %s
        LIMIT 1
        """,
        (plate,),
    )
    vehicle = cur.fetchone()

    if not vehicle:
        cur.execute(
            """
            INSERT INTO vehicle (plate_number, vehicle_type, created_at)
            VALUES (%s, 'NORMAL', now())
            RETURNING id, vehicle_type
            """,
            (plate,),
        )
        vehicle = cur.fetchone()

    vehicle_id = vehicle["id"]
    vehicle_type = vehicle["vehicle_type"]

    # 활성 세션 확인
    cur.execute(
        """
        SELECT id, entry_time, entry_image_url
        FROM parking_session
        WHERE vehicle_id = %s
          AND exit_time IS NULL
        LIMIT 1
        """,
        (vehicle_id,),
    )
    session = cur.fetchone()

    # =========================
    # ENTRY
    # =========================
    if not session:
        cur.execute(
            """
            SELECT COUNT(*) AS occupied
            FROM parking_session
            WHERE parking_lot_id = %s
              AND exit_time IS NULL
            """,
            (parking_lot_id,),
        )
        occupied = cur.fetchone()["occupied"]

        if occupied >= capacity:
            conn.rollback()
            conn.close()
            return {
                "direction": "ENTRY_DENIED",
                "reason": "FULL",
                "parking_lot": {
                    "occupied": occupied,
                    "capacity": capacity,
                },
            }

        cur.execute(
            """
            INSERT INTO parking_session (
                vehicle_id,
                parking_lot_id,
                entry_time,
                status,
                entry_image_url,
                created_at
            )
            VALUES (%s, %s, now(), 'PARKED', %s, now())
            RETURNING id
            """,
            (vehicle_id, parking_lot_id, image_url),
        )
        session_id = cur.fetchone()["id"]

        payment_status = "FREE" if vehicle_type != "NORMAL" else "UNPAID"

        cur.execute(
            """
            INSERT INTO payment (
                parking_session_id,
                amount,
                payment_status,
                created_at
            )
            VALUES (%s, 0, %s, now())
            """,
            (session_id, payment_status),
        )

        conn.commit()
        conn.close()

        return {
            "direction": "ENTRY",
            "parking_session_id": session_id,
            "card": {
                "plate": plate,
                "vehicle_type": vehicle_type,
                "vehicle_type_label": VEHICLE_TYPE_LABEL.get(vehicle_type),
                "entry_image_url": image_url,
                "payment_status": payment_status,
            },
            "payment_status": payment_status,
        }

    # =========================
    # EXIT (🔥 구조 통일)
    # =========================
    session_id = session["id"]

    cur.execute(
        """
        SELECT payment_status
        FROM payment
        WHERE parking_session_id = %s
        """,
        (session_id,),
    )
    payment_status = cur.fetchone()["payment_status"]

    # 공통 카드 구성 (⭐ 중요)
    card = {
        "plate": plate,
        "vehicle_type": vehicle_type,
        "vehicle_type_label": VEHICLE_TYPE_LABEL.get(vehicle_type),
        "entry_image_url": session["entry_image_url"],
        "exit_image_url": image_url,
        "payment_status": payment_status,
    }

    # 미결제
    if payment_status not in ("PAID", "FREE"):
        conn.close()
        return {
            "direction": "EXIT",
            "parking_session_id": session_id,
            "card": card,
            "payment_status": payment_status,
        }

    # 결제 완료 → 출차 처리
    cur.execute(
        """
        UPDATE parking_session
        SET exit_time = now(),
            status = 'EXITED',
            exit_image_url = %s
        WHERE id = %s
          AND exit_time IS NULL
        """,
        (image_url, session_id),
    )

    conn.commit()
    conn.close()

    return {
        "direction": "EXIT",
        "parking_session_id": session_id,
        "card": card,              # ✅ 항상 포함
        "payment_status": payment_status,
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

    image_url = upload_image(
        contents,
        f"parking/raw/{uuid.uuid4()}.jpg"
    )

    plate_img = detect_plate_region(img)
    if plate_img is None:
        return {"success": False, "error": "PLATE_NOT_FOUND"}

    plate, confidence = extract_plate(plate_img)
    if not plate or confidence < 0.45:
        return {
            "success": False,
            "error": "LOW_CONFIDENCE",
            "confidence": confidence,
        }

    parking_lot_id = "33e088ea-66d8-4ef9-aa0b-dd533cdb885b"

    result = resolve_direction_and_process(
        plate=plate,
        image_url=image_url,
        parking_lot_id=parking_lot_id,
    )

    return {
        "success": True,
        "plate": plate,
        "confidence": confidence,
        **result,
    }