# 모델 / 엔진 전역 상태
whisper_model = None
app_engine = None

# ==================================================
# ✅ [추가] 데모(단일 세션)용 현재 컨텍스트
# - plate 업로드 시점에 현재 parking_session_id를 저장해두고
# - voice/app_engine에서 이 값을 기준으로 payment/payment_log를 조회한다
# ==================================================
current_parking_session_id = None  # str | None
current_plate_number = None        # str | None
current_direction = None           # "ENTRY" | "EXIT" | None
current_vehicle_type = None        # 예: "NORMAL"/"MEMBER"/"VISITOR" | None
current_payment_status = None      # 예: "PAID"/"UNPAID"/"FREE" | None


def set_current_context(
    *,
    parking_session_id: str,
    plate: str,
    direction: str,
    vehicle_type: str | None = None,
    payment_status: str | None = None,
) -> None:
    global current_parking_session_id, current_plate_number, current_direction
    global current_vehicle_type, current_payment_status

    current_parking_session_id = parking_session_id
    current_plate_number = plate
    current_direction = direction
    current_vehicle_type = vehicle_type
    current_payment_status = payment_status