# intents/payment_logic.py

def get_payment_status(supabase, plate_number):
    """
    해당 차량의 가장 최근 결제 시도 내역을 조회합니다.
    """
    try:
        # 1. 차량 번호로 session_id 먼저 찾기
        session_res = supabase.table("v1_parking_session")\
            .select("session_id")\
            .eq("plate_number", plate_number)\
            .order("entry_time", desc=True)\
            .limit(1)\
            .execute()

        if not session_res.data:
            return {"status": "error", "message": "차량 조회 불가"}
            
        session_id = session_res.data[0]['session_id']

        # 2. session_id로 결제 로그 조회 (최신순)
        log_res = supabase.table("v1_payment_log")\
            .select("*")\
            .eq("session_id", session_id)\
            .order("paid_at", desc=True)\
            .limit(1)\
            .execute()

        # 결제 기록이 아예 없는 경우
        if not log_res.data:
            return {
                "status": "info",
                "payment_status": "NONE",
                "message": "결제 시도 이력이 없습니다. 정산 먼저 부탁드립니다."
            }

        last_log = log_res.data[0]

        # 3. AI용 데이터 구성
        return {
            "status": "success",
            "intent": "payment",
            "plate_number": plate_number,
            "payment_status": last_log['status'],  # SUCCESS or FAIL
            "fail_reason": last_log.get('err_msg', '알 수 없음'), # 잔액부족 등
            "tried_at": last_log['paid_at'],
            "description": f"최근 결제 상태는 {last_log['status']}이며, 사유는 {last_log.get('err_msg', '없음')} 입니다."
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}