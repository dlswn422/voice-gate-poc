# events/entry_event.py

from datetime import datetime, timezone


# 순서를 나타내는 고유어(순우리말) 수사 변환
# "n번째"에 사용. 한자어 수사(일, 이, 삼...)로 읽히면 어색하므로
# 고유어 수사(첫, 두, 세...)로 변환합니다.
_NATIVE_ONES  = ["", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"]
_NATIVE_TENS  = ["", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"]


def _ordinal_kor(n: int) -> str:
    """
    양의 정수 n을 한국어 고유어 서수 표현으로 변환합니다.
    예) 1 → "첫 번째", 2 → "두 번째", 11 → "열한 번째"
    99 초과는 숫자 그대로 "n번째" 로 fallback 합니다.
    """
    if n <= 0:
        return f"{n}번째"
    if n == 1:
        return "첫 번째"  # '한 번째'보다 '첫 번째'가 더 자연스러움
    if n <= 9:
        return f"{_NATIVE_ONES[n]} 번째"
    if n <= 99:
        tens, ones = divmod(n, 10)
        if ones == 0:
            return f"{_NATIVE_TENS[tens]} 번째"
        return f"{_NATIVE_TENS[tens]}{_NATIVE_ONES[ones]} 번째"
    # 100 이상: 한자어 혼합 표현 대신 숫자 그대로 사용
    return f"{n}번째"


def handle_entry_event(supabase, plate_number):
    """
    정상 입차 처리:
    1. 방문 횟수 증가 (신규 차량이면 먼저 row 생성)
    2. v1_parking_session 에 새 세션 row INSERT (session_id 는 DB Auto-increment 위임)
    3. 여유 주차 구역 추천
    4. TTS 멘트 생성
    """
    try:
        # -------------------------------------------------------
        # [Step 1] 방문 횟수 조회 및 업데이트
        # -------------------------------------------------------
        user_res = supabase.table("v1_vehicle_master") \
            .select("visit_count") \
            .eq("plate_number", plate_number) \
            .execute()

        if user_res.data:
            # 기존 차량: visit_count +1 update
            current_count = user_res.data[0]["visit_count"] or 0
            new_count = current_count + 1
            supabase.table("v1_vehicle_master") \
                .update({"visit_count": new_count}) \
                .eq("plate_number", plate_number) \
                .execute()
        else:
            # 신규 차량: row 자체가 없으므로 insert 로 생성
            new_count = 1
            supabase.table("v1_vehicle_master") \
                .insert({
                    "plate_number": plate_number,
                    "is_monthly":   False,
                    "visit_count":  new_count,
                }) \
                .execute()
        # -------------------------------------------------------
        # [Step 1.5] 중복 입차 검증 (방어 코드)
        # -------------------------------------------------------
        active_session = supabase.table("v1_parking_session") \
            .select("session_id") \
            .eq("plate_number", plate_number) \
            .eq("status", "PARKED") \
            .execute()

        if active_session.data:
            # 이미 주차 중인 차량이면 데이터베이스에 넣지 않고 즉시 반환
            return {
                "status": "success", # 시스템 에러는 아니므로 success 처리
                "type": "entry_duplicate",
                "tts_message": "이미 입차 처리된 차량입니다."
            }
        # -------------------------------------------------------
        # [Step 2] 주차 세션 INSERT
        # session_id 는 DB(Integer, Auto-increment)에 위임 — 페이로드에서 제외
        # -------------------------------------------------------
        entry_time = datetime.now(timezone.utc)

        insert_res = supabase.table("v1_parking_session") \
            .insert({
                "plate_number": plate_number,
                "entry_time":   entry_time.isoformat(),
                "exit_time":    None,
                "status":       "PARKED",
            }) \
            .execute()

        # DB 가 자동 생성한 정수형 session_id 추출
        session_id = insert_res.data[0]["session_id"]

        # -------------------------------------------------------
        # [Step 3] 주차 구역 추천 (여유 공간이 많은 구역 1개)
        # -------------------------------------------------------
        zone_res = supabase.table("v1_parking_zone_status") \
            .select("zone_id") \
            .order("total_spots", desc=True) \
            .limit(1) \
            .execute()

        recommend_zone = zone_res.data[0]["zone_id"] if zone_res.data else "A구역"

        # -------------------------------------------------------
        # [Step 4] TTS 멘트 생성
        # -------------------------------------------------------
        ordinal = _ordinal_kor(new_count)
        tts_message = (
            f"{ordinal} 방문이시네요. "
            f"현재 {recommend_zone}에 자리가 여유롭습니다. 편안한 주차 되십시오."
        )

        return {
            "status": "success",
            "type":   "entry",
            "data": {
                "plate_number":   plate_number,
                "session_id":     session_id,
                "visit_count":    new_count,
                "recommend_zone": recommend_zone,
            },
            "tts_message": tts_message,
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}