# events/exit_event.py
from datetime import datetime, timezone
import math


def handle_exit_event(supabase, plate_number):
    """
    정상 출차 처리: 요금 계산 및 고정 멘트 반환

    [버그 수정 내역]
    - is_monthly(정기권) 여부를 확인하지 않아 정기권 차량에도
      일반 요금이 계산되던 문제 → fee_logic.py 와 동일한 정기권 분기 추가
    """
    try:
        # -------------------------------------------------------
        # [Step 1] 활성 세션(PARKED) 조회
        # -------------------------------------------------------
        session_res = supabase.table("v1_parking_session") \
            .select("*") \
            .eq("plate_number", plate_number) \
            .eq("status", "PARKED") \
            .execute()

        if not session_res.data:
            return {"status": "error", "message": "주차 차량 아님"}

        session = session_res.data[0]

        # -------------------------------------------------------
        # [Step 2] 정기권 여부 조회
        # -------------------------------------------------------
        vehicle_res = supabase.table("v1_vehicle_master") \
            .select("is_monthly") \
            .eq("plate_number", plate_number) \
            .execute()

        is_monthly = vehicle_res.data[0]["is_monthly"] if vehicle_res.data else False

        # -------------------------------------------------------
        # [Step 3] 이용 시간 계산
        # -------------------------------------------------------
        entry_dt   = datetime.fromisoformat(session["entry_time"].replace("Z", "+00:00"))
        current_dt = datetime.now(timezone.utc)
        total_minutes = int((current_dt - entry_dt).total_seconds() / 60)

        # -------------------------------------------------------
        # [Step 4] 요금 계산 (정기권이면 0원, 일반이면 정책 적용)
        # -------------------------------------------------------
        calculated_fee = 0

        if not is_monthly:
            policy_res = supabase.table("v1_fee_policy") \
                .select("*") \
                .eq("policy_ver", 1) \
                .execute()

            if not policy_res.data:
                return {"status": "error", "message": "요금 정책 데이터가 없습니다"}

            policy = policy_res.data[0]

            if total_minutes > policy["free_grace_min"]:
                if total_minutes <= policy["base_time_min"]:
                    calculated_fee = policy["base_fee"]
                else:
                    extra       = total_minutes - policy["base_time_min"]
                    extra_units = math.ceil(extra / policy["unit_time_min"])
                    calculated_fee = policy["base_fee"] + (extra_units * policy["unit_fee"])

            if calculated_fee > policy["max_daily_fee"]:
                calculated_fee = policy["max_daily_fee"]

        # -------------------------------------------------------
        # [Step 5] 세션 종료 업데이트  (status → COMPLETED)
        # -------------------------------------------------------
        supabase.table("v1_parking_session") \
            .update({
                "status":    "COMPLETED",
                "exit_time": current_dt.isoformat(),
            }) \
            .eq("session_id", session["session_id"]) \
            .execute()

        # -------------------------------------------------------
        # [Step 6] TTS 멘트 생성
        # -------------------------------------------------------
        hours    = total_minutes // 60
        mins     = total_minutes % 60
        time_str = (f"{hours}시간 " if hours > 0 else "") + f"{mins}분"

        if is_monthly:
            tts_message = (
                f"총 {time_str} 이용하셨습니다. "
                "정기권 차량으로 요금이 없습니다. 안녕히 가십시오."
            )
        else:
            tts_message = (
                f"총 {time_str} 이용하셨습니다. "
                f"요금은 {calculated_fee:,}원입니다. 이용해 주셔서 감사합니다."
            )

        return {
            "status": "success",
            "type":   "exit",
            "tts_message": tts_message,
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}