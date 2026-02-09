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
import src.app_state as app_state

router = APIRouter()

# =========================
# OCR 설정 (CPU ONLY)
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
def detect_plate_region(img: np.ndarray) -> np.ndarray | None:
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
# 입출차 처리 (DB 전용)
# =========================
def resolve_direction_and_process(
    plate: str,
    image_url: str,
):
    now = datetime.utcnow()

    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # 1️⃣ vehicle 조회 or 생성
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
        conn.commit()

    vehicle_id = vehicle["id"]
    vehicle_type = vehicle["vehicle_type"]

    # 2️⃣ 활성 세션 조회
    cur.execute(
        """
        SELECT id, entry_time, entry_image_url
        FROM parking_session
        WHERE vehicle_id = %s
          AND exit_time IS NULL
        ORDER BY entry_time DESC
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
            INSERT INTO parking_session (
                vehicle_id,
                entry_time,
                status,
                entry_image_url,
                created_at
            )
            VALUES (%s, now(), 'PARKED', %s, now())
            RETURNING id, entry_time
            """,
            (vehicle_id, image_url),
        )
        new_session = cur.fetchone()
        session_id = new_session["id"]

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
            },
            "payment_status": payment_status,
        }

    # =========================
    # EXIT
    # =========================
    session_id = session["id"]

    cur.execute(
        """
        SELECT payment_status
        FROM payment
        WHERE parking_session_id = %s
        LIMIT 1
        """,
        (session_id,),
    )
    payment = cur.fetchone()
    payment_status = payment["payment_status"]

    exit_time = now.isoformat()

    # 미결제 → 출차 불가
    if payment_status not in ("PAID", "FREE"):
        conn.close()
        return {
            "direction": "EXIT",
            "parking_session_id": session_id,
            "card": {
                "plate": plate,
                "vehicle_type": vehicle_type,
                "vehicle_type_label": VEHICLE_TYPE_LABEL.get(vehicle_type),
                "entry_image_url": session["entry_image_url"],
                "exit_image_url": image_url,
                "payment_status": payment_status,
            },
            "time_info": {
                "entry_time": session["entry_time"].isoformat(),
                "exit_time": exit_time,
            },
            "payment_status": payment_status,
        }

    # 결제 완료 출차
    cur.execute(
        """
        UPDATE parking_session
        SET exit_time = now(),
            status = 'EXITED',
            exit_image_url = %s
        WHERE id = %s
        """,
        (image_url, session_id),
    )

    conn.commit()
    conn.close()

    return {
        "direction": "EXIT",
        "parking_session_id": session_id,
        "card": {
            "plate": plate,
            "vehicle_type": vehicle_type,
            "vehicle_type_label": VEHICLE_TYPE_LABEL.get(vehicle_type),
            "entry_image_url": session["entry_image_url"],
            "exit_image_url": image_url,
            "payment_status": payment_status,
        },
        "time_info": {
            "entry_time": session["entry_time"].isoformat(),
            "exit_time": exit_time,
        },
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

    image_key = f"parking/raw/{uuid.uuid4()}.jpg"
    image_url = upload_image(contents, image_key)

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

    db_result = resolve_direction_and_process(
        plate=plate,
        image_url=image_url,
    )

    # 🔔 중앙 세션 엔진에 이벤트 전달
    engine_result = app_state.session_engine.handle_event({
        "type": "VEHICLE_DETECTED",
        "direction": db_result["direction"],
        "payment_status": db_result.get("payment_status"),
    })

    return {
        "success": True,
        "plate": plate,
        "confidence": confidence,
        **db_result,
        **engine_result,
    }
