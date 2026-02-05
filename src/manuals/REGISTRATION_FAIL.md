# REGISTRATION_FAIL

### CASE 1. 방문자 등록이 되어 있지 않음
조건:
- visit_registration_expected == true
- visit_not_registered == true
ASK_QUESTION:
- "방문자 등록을 미리 하신 상태인가요?"
- "방문자 등록이 되어 있지 않다는 안내가 나오나요?"
결론:
- 방문자 등록을 먼저 진행해 주세요.
- 등록이 완료된 뒤 다시 입차를 시도해 주세요.
- 등록 방법을 모르겠으면 관제에 연결해 주세요.

### CASE 2. 방문자 등록은 했으나 차량 번호 불일치
조건:
- visit_registered == true
- vehicle_number_mismatch == true
ASK_QUESTION:
- "방문자 등록은 되어 있는데 차량 번호가 다르다는 안내가 나오나요?"
결론:
- 등록된 차량 번호와 실제 차량 번호가 동일한지 확인해 주세요.
- 번호가 다르면 방문자 등록을 다시 수정해 주세요.
- 문제가 계속되면 관제에 연결해 주세요.

### CASE 3. 방문자 등록 시간 미도래 또는 만료
조건:
- visit_registered == true
- visit_time_invalid == true
ASK_QUESTION:
- "방문 시간 전이거나 방문 시간이 이미 지난 상태인가요?"
결론:
- 방문 가능한 시간인지 다시 확인해 주세요.
- 필요하면 방문자 등록 시간을 수정해 주세요.
- 수정이 어렵다면 관제에 연결해 주세요.

### CASE 4. 방문자 등록 시스템 오류 (관리자 이관)
조건:
- visit_registration_error_message == true
ASK_QUESTION:
- "방문자 등록 관련 오류 메시지가 표시되나요?"
결론:
- 시스템 문제로 자동 해결이 불가능합니다.
- 즉시 관리자 또는 관제에 연결해 주세요.

### CASE 5. 방문자 등록은 했으나 입차 권한 미부여
조건:
- visit_registered == true
- entry_permission_denied == true
ASK_QUESTION:
- "방문자 등록은 되어 있는데 입차 권한이 없다는 안내가 나오나요?"
결론:
- 방문자 등록 정보가 정상 승인되었는지 확인해 주세요.
- 승인 상태가 아니라면 승인 후 다시 입차를 시도해 주세요.
- 확인이 어렵다면 관제에 연결해 주세요.

### CASE 6. 차량 번호 입력 오타 또는 형식 오류
조건:
- registration_input_done == true
- plate_input_error == true
ASK_QUESTION:
- "차량 번호 입력에 오류가 있을 수 있나요?"
결론:
- 차량 번호를 다시 확인한 뒤 정확히 입력해 주세요.
- 수정 후 다시 등록을 시도해 주세요.

### CASE 7. 차량 번호가 이미 등록됨
조건:
- plate_already_registered_message == true
ASK_QUESTION:
- "이미 등록된 차량 번호라는 안내가 나오나요?"
결론:
- 중복 등록으로 자동 처리하기 어렵습니다.
- 관제에 연결해 등록 상태 확인을 요청해 주세요.

### CASE 8. 방문자 등록 승인 대기 상태
조건:
- visit_registered == true
- visit_approval_pending == true
ASK_QUESTION:
- "방문자 등록이 승인 대기 상태로 표시되나요?"
결론:
- 승인 완료 후 입차가 가능합니다.
- 급한 경우 관제에 연결해 승인을 요청해 주세요.

### CASE 9. 방문자 등록이 반려됨
조건:
- visit_registered == true
- visit_approval_rejected == true
ASK_QUESTION:
- "방문자 등록이 반려되었다는 안내가 나오나요?"
결론:
- 등록 정보를 다시 확인해 재등록해 주세요.
- 사유 확인이 필요하면 관제에 연결해 주세요.

### CASE 10. 필수 정보 누락으로 등록 실패
조건:
- required_field_missing_message == true
ASK_QUESTION:
- "필수 입력 항목이 누락되었다는 안내가 나오나요?"
결론:
- 누락된 항목을 모두 입력한 뒤 다시 등록해 주세요.

### CASE 11. 인증번호(SMS) 확인 실패 또는 미수신
조건:
- sms_verification_issue == true
ASK_QUESTION:
- "인증번호 확인이 되지 않았거나 문자를 받지 못하셨나요?"
결론:
- 인증번호를 다시 요청한 뒤 정확히 입력해 주세요.
- 계속 실패하면 관제에 연결해 주세요.

### CASE 12. 여러 차량 등록으로 혼란
조건:
- multiple_vehicles_registered == true
ASK_QUESTION:
- "등록된 차량이 여러 대로 표시되나요?"
결론:
- 실제 입차 차량 번호가 등록 목록에 포함되어 있는지 확인해 주세요.
- 확인이 어렵다면 관제에 연결해 주세요.

### CASE 13. 임시/대차/렌트 차량 등록 필요
조건:
- temporary_vehicle_expected == true
- not_registered_message == true
ASK_QUESTION:
- "임시 차량(대차/렌트) 등록이 필요한 상황인가요?"
결론:
- 임시 차량 등록 후 다시 입차를 시도해 주세요.
- 등록 방법 안내가 필요하면 관제에 연결해 주세요.

### CASE 14. 정기권/입주/직원 차량 등록 만료
조건:
- membership_expected == true
- membership_expired == true
ASK_QUESTION:
- "정기권이나 등록이 만료되었다는 안내가 나오나요?"
결론:
- 갱신이 필요한 상태입니다.
- 관제에 연결해 갱신 여부를 확인해 주세요.

### CASE 15. 정기권/입주/직원 차량 등록 비활성화
조건:
- membership_expected == true
- membership_disabled == true
ASK_QUESTION:
- "등록 정보가 비활성화되었다는 안내가 나오나요?"
결론:
- 자동으로 해결하기 어렵습니다.
- 관제에 연결해 상태 확인을 요청해 주세요.

### CASE 16. 등록 저장 실패
조건:
- registration_save_failed == true
ASK_QUESTION:
- "등록 저장이 되지 않는다는 안내가 나오나요?"
결론:
- 입력 정보를 다시 확인한 뒤 재시도해 주세요.
- 계속 실패하면 관제에 연결해 주세요.

### CASE 17. 등록 완료 후 입차 불가
조건:
- registration_completed == true
- entry_permission_denied == true
ASK_QUESTION:
- "등록은 완료했는데 입차가 되지 않나요?"
결론:
- 반영 지연일 수 있으니 잠시 후 다시 시도해 주세요.
- 계속 안 되면 관제에 연결해 확인을 요청해 주세요.

### CASE 18. 개인정보/약관 동의 미완료
조건:
- consent_required_message == true
ASK_QUESTION:
- "약관이나 개인정보 동의가 필요하다는 안내가 나오나요?"
결론:
- 필수 동의를 완료한 뒤 다시 등록해 주세요.

### CASE 19. 등록 횟수 또는 정원 초과
조건:
- registration_quota_exceeded == true
ASK_QUESTION:
- "등록 가능 횟수나 정원이 초과되었다는 안내가 나오나요?"
결론:
- 자동으로 처리하기 어렵습니다.
- 관제에 연결해 확인을 요청해 주세요.

### CASE 20. 동일 방문자 중복 등록
조건:
- duplicate_visitor_detected == true
ASK_QUESTION:
- "이미 등록된 방문자 정보라는 안내가 나오나요?"
결론:
- 중복 등록일 수 있습니다.
- 관제에 연결해 등록 상태 확인을 요청해 주세요.

### CASE 21. 등록 유형 선택 오류
조건:
- registration_type_mismatch == true
ASK_QUESTION:
- "방문자/직원/입주 등 등록 유형을 잘못 선택하신 것 같나요?"
결론:
- 올바른 등록 유형으로 다시 등록해 주세요.
- 구분이 어렵다면 관제에 연결해 주세요.

### CASE 22. 등록 관련 오류 메시지 표시 (관리자 이관)
조건:
- registration_error_message == true
ASK_QUESTION:
- "등록 중 오류 메시지가 표시되나요?"
결론:
- 시스템 문제로 자동 해결이 불가능합니다.
- 즉시 관리자 또는 관제에 연결해 주세요.

### CASE 23. 차량/방문자 등록 방법을 모름
조건:
- registration_method_unknown == true
ASK_QUESTION:
- "차량이나 방문자 등록 방법을 잘 모르시겠나요?"
결론:
- 등록은 사전에 완료되어야 입차가 가능합니다.
- 안내가 필요하면 관제에 연결해 주세요.
