# intents/fee_logic.py
from datetime import datetime, timezone
import math

def get_fee_info(supabase, plate_number):
    try:
        # 1. 차량 정보 조회 (정기권 여부 확인)
        vehicle_res = supabase.table("v1_vehicle_master")\
            .select("is_monthly")\
            .eq("plate_number", plate_number)\
            .execute()
            
        if not vehicle_res.data:
            return {"status": "error", "message": "등록되지 않은 차량입니다."}
            
        is_monthly = vehicle_res.data[0]['is_monthly']

        # 2. 입차 세션 조회
        session_res = supabase.table("v1_parking_session")\
            .select("*")\
            .eq("plate_number", plate_number)\
            .order("entry_time", desc=True)\
            .limit(1)\
            .execute()

        if not session_res.data:
            return {"status": "error", "message": "입차 기록이 없는 차량입니다."}

        session = session_res.data[0]
        entry_time_str = session["entry_time"]
        
        # 3. 시간 계산
        entry_dt = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00'))
        current_dt = datetime.now(timezone.utc)
        duration = current_dt - entry_dt
        total_minutes = int(duration.total_seconds() / 60)

        # =====================================================
        #  정기권이면 요금 계산 로직 스킵 
        # =====================================================
        calculated_fee = 0
        policy_desc = ""

        if is_monthly:
            # 정기권일 때
            calculated_fee = 0
            policy_desc = "정기권 차량 (무료)"
        else:
            # 일반 차량일 때 
            policy_res = supabase.table("v1_fee_policy").select("*").eq("policy_ver", 1).execute()
            policy = policy_res.data[0]
            
            if total_minutes <= policy["free_grace_min"]:
                calculated_fee = 0
            elif total_minutes <= policy["base_time_min"]:
                calculated_fee = policy["base_fee"]
            else:
                extra = total_minutes - policy["base_time_min"]
                extra_units = math.ceil(extra / policy["unit_time_min"])
                calculated_fee = policy["base_fee"] + (extra_units * policy["unit_fee"])

            if calculated_fee > policy["max_daily_fee"]:
                calculated_fee = policy["max_daily_fee"]
            
            policy_desc = f"일반 요금 (30분 {policy['base_fee']}원, 10분당 {policy['unit_fee']}원)"

        # 4. 결과 리턴
        return {
            "status": "success",
            "intent": "fee",
            "plate_number": plate_number,
            "is_monthly": is_monthly, 
            "total_minutes": total_minutes,
            "calculated_fee": calculated_fee,
            "description": f"차량 유형: {'정기권' if is_monthly else '일반'}, 이용시간: {total_minutes}분, 최종요금: {calculated_fee}원."
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}