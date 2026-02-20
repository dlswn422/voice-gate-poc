# intents/facility_logic.py

def get_device_status(supabase, device_id="GATE_OUT_01"):
    """
    기기 상태 조회 로직
    - OPEN/CLOSED/NORMAL : 정상 상태로 판단
    - CONN_ERR/PAPER_EMPTY : 고장(에러) 상태로 판단 -> 관리자 호출 필요
    """
    try:
        # 1. 기기 상태 테이블 조회
        device_res = supabase.table("v1_device_status")\
            .select("*")\
            .eq("device_id", device_id)\
            .execute()

        if not device_res.data:
            return {"status": "error", "message": "등록되지 않은 기기입니다."}

        device = device_res.data[0]
        status_code = device['status']

        # 2. 상태 해석 로직 
        status_desc = ""
        is_error = False # 기본값: 정상

        if status_code == 'NORMAL':
            status_desc = "정상 작동 중"
        elif status_code == 'BREAKER_OPEN':
            status_desc = "차단기 열림 (진입 가능)"
        elif status_code == 'BREAKER_CLOSED':
            status_desc = "차단기 닫힘 (대기 중)"
        elif status_code == 'CONN_ERR':
            status_desc = "통신 장애 발생"
            is_error = True  # 관리자 호출 대상
        elif status_code == 'PAPER_EMPTY':
            status_desc = "용지 부족"
            is_error = True  # 관리자 호출 대상
        else:
            status_desc = f"알 수 없는 상태 ({status_code})"
            is_error = True

       
        return {
            "status": "success",
            "intent": "facility",
            "device_id": device_id,
            "device_status": status_code,
            "status_description": status_desc,
            "need_admin": is_error,  # True면 main.py에서 관리자 호출
            "description": f"현재 {device_id} 기기는 {status_desc} 상태입니다."
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}