# EXIT_FLOW_ISSUE

### CASE 1. 결제 완료 후 차단기 미개방
조건:
- paid_done == true
- barrier_not_open == true
ASK_QUESTION:
- "방금 결제(정산) 완료 화면을 보셨나요?"
- "지금 출구에서 차단기가 올라가지 않나요?"
결론:
- 출구에 있는 호출 버튼을 눌러 관제와 연결해 주세요.

### CASE 2. 출구 카드 결제 실패
조건:
- exit_payment_tried == true
- card_payment_failed == true
ASK_QUESTION:
- "출구에서 카드로 결제를 시도하셨나요?"
- "카드가 승인되지 않거나 반응이 없었나요?"
결론:
- 다른 결제 수단이나 무인정산기(키오스크)를 이용해 주세요.
- 같은 문제가 반복되면 관리자 또는 관제에 연결해 주세요.

### CASE 3. 번호판 인식 실패로 출차 불가
조건:
- exit_mode == "LPR"
- lpr_recognition_failed == true
ASK_QUESTION:
- "출차가 번호판 인식 방식인가요?"
- "번호판 인식이 되지 않는다는 안내가 나오나요?"
결론:
- 번호판이 가려지거나 오염되지 않았는지 확인해 주세요.
- 천천히 후진한 뒤 다시 출차를 시도해 주세요.
- 해결되지 않으면 관리자 또는 관제에 연결해 주세요.

### CASE 4. 입차 기록 없음 표시 (관리자 이관)
조건:
- no_entry_record_message == true
ASK_QUESTION:
- "입차 기록이 없다는 메시지가 표시되나요?"
결론:
- 자동으로 해결할 수 없는 상황입니다.
- 즉시 관리자 또는 관제에 연결해 주세요.

### CASE 5. 할인 또는 무료시간 미적용
조건:
- discount_expected == true
- discount_not_applied == true
ASK_QUESTION:
- "할인이나 무료주차 대상이 맞으신가요?"
- "요금에 할인이 적용되지 않았나요?"
결론:
- 할인이나 무료주차가 정상 등록되었는지 다시 확인해 주세요.
- 필요하면 관리자 또는 관제에 연결해 주세요.

### CASE 6. 요금이 비정상적으로 과다함
조건:
- fee_excessive == true
ASK_QUESTION:
- "예상보다 요금이 많이 나온 것 같으신가요?"
결론:
- 요금 내역을 한 번 더 확인해 주세요.
- 요금이 납득되지 않으면 관리자 또는 관제에 연결해 주세요.

### CASE 7. 출구 키오스크 또는 단말기 먹통 (관리자 이관)
조건:
- exit_terminal_unresponsive == true
ASK_QUESTION:
- "출구 화면이 멈추거나 터치가 되지 않나요?"
결론:
- 자동으로 해결할 수 없는 상황입니다.
- 즉시 관리자 또는 관제에 연결해 주세요.

### CASE 8. 네트워크 또는 서버 오류 (관리자 이관)
조건:
- network_error_message == true
- exit_process_blocked == true
ASK_QUESTION:
- "서버 오류나 통신 오류 메시지가 표시되나요?"
결론:
- 시스템 문제로 자동 해결이 불가능합니다.
- 관리자 또는 관제에 연결해 주세요.

### CASE 9. 앞차 출차 후 차단기 미개방
조건:
- front_car_exited == true
- barrier_not_open == true
ASK_QUESTION:
- "앞차는 정상적으로 출차했나요?"
결론:
- 차량을 앞뒤로 조금 이동한 뒤 다시 정차해 주세요.
- 같은 문제가 반복되면 관리자 또는 관제에 연결해 주세요.

### CASE 10. 출차 방법 자체를 모름
조건:
- exit_method_unknown == true
ASK_QUESTION:
- "출차 방법을 잘 모르시겠나요?"
결론:
- 출구로 이동해 정차하면 차량 인식 후 차단기가 자동으로 열립니다.
- 결제가 필요한 경우 정산 후 다시 출차를 시도해 주세요.

### CASE 11. 출구에서 차량 위치/거리 문제로 인식 실패
조건:
- vehicle_position_misaligned == true
- barrier_not_open == true
ASK_QUESTION:
- "차량이 정지선보다 너무 앞이나 뒤에 있지는 않나요?"
결론:
- 차량을 정지선에 맞춰 다시 정차해 주세요.
- 천천히 다시 출차를 시도해 주세요.

### CASE 12. 출구 접근 속도가 너무 빠름
조건:
- vehicle_approach_too_fast == true
ASK_QUESTION:
- "출구로 빠르게 접근하셨나요?"
결론:
- 천천히 접근해 정지선에 정차해 주세요.
- 다시 출차를 시도해 주세요.

### CASE 13. 출구 차로 잘못 진입
조건:
- wrong_exit_lane == true
ASK_QUESTION:
- "차량 종류에 맞지 않은 출구 차로로 나오신 건가요?"
결론:
- 안내 표지에 맞는 출구 차로로 이동해 주세요.
- 필요하면 관제에 연결해 주세요.

### CASE 14. 번호판이 가려지거나 오염됨
조건:
- exit_mode == "LPR"
- license_plate_obstructed == true
ASK_QUESTION:
- "번호판이 가려지거나 오염되어 있지는 않나요?"
결론:
- 번호판을 깨끗이 한 뒤 다시 출차를 시도해 주세요.
- 해결되지 않으면 관제에 연결해 주세요.

### CASE 15. 야간/역광 등 환경 문제로 번호판 인식 실패
조건:
- exit_mode == "LPR"
- poor_lighting_condition == true
ASK_QUESTION:
- "주변이 어둡거나 역광 상태인가요?"
결론:
- 전조등을 끄거나 차량 각도를 조금 조정한 뒤 다시 시도해 주세요.
- 계속 실패하면 관제에 연결해 주세요.

### CASE 16. 비·눈 등 날씨 영향으로 출차 인식 실패
조건:
- bad_weather_condition == true
- barrier_not_open == true
ASK_QUESTION:
- "비나 눈 때문에 시야가 좋지 않나요?"
결론:
- 잠시 정차한 뒤 다시 출차를 시도해 주세요.
- 인식이 되지 않으면 관제에 연결해 주세요.

### CASE 17. 이전 차량 출차 처리 지연
조건:
- previous_exit_processing == true
ASK_QUESTION:
- "앞차 출차 직후 바로 따라 나오셨나요?"
결론:
- 잠시 기다린 뒤 다시 출차를 시도해 주세요.
- 반복되면 관제에 연결해 주세요.

### CASE 18. 결제는 완료됐으나 반영 지연
조건:
- paid_done == true
- payment_reflection_delayed == true
ASK_QUESTION:
- "결제 후 바로 출차를 시도하셨나요?"
결론:
- 잠시 기다린 뒤 다시 출차를 시도해 주세요.
- 계속 안 되면 관제에 연결해 주세요.

### CASE 19. 무료주차/할인 적용 후 반응 지연
조건:
- discount_applied == true
- barrier_not_open == true
ASK_QUESTION:
- "할인이나 무료주차 적용 후 바로 출차하셨나요?"
결론:
- 잠시 대기한 뒤 다시 출차를 시도해 주세요.
- 해결되지 않으면 관제에 연결해 주세요.

### CASE 20. 출구 경고음만 울리고 진행 안 됨
조건:
- exit_warning_sound_only == true
ASK_QUESTION:
- "경고음만 울리고 출차가 되지 않나요?"
결론:
- 차량을 정차한 상태로 유지해 주세요.
- 관제에 연결해 안내를 받아 주세요.

### CASE 21. 출구 안전 차단 상태
조건:
- safety_lock_active == true
ASK_QUESTION:
- "안전 차단으로 출차가 제한된다는 안내가 나오나요?"
결론:
- 잠시 대기해 주세요.
- 안내가 없으면 관제에 연결해 주세요.

### CASE 22. 출구 설비 점검 중
조건:
- exit_maintenance_mode == true
ASK_QUESTION:
- "출구 설비 점검 중이라는 안내가 보이나요?"
결론:
- 다른 출구를 이용해 주세요.
- 안내가 없으면 관제에 연결해 주세요.

### CASE 23. 차량 번호 일부만 인식됨
조건:
- partial_plate_match == true
ASK_QUESTION:
- "번호판 일부만 인식된다는 안내가 나오나요?"
결론:
- 등록된 차량 번호와 실제 번호판이 일치하는지 확인해 주세요.
- 수정이 필요하면 관제에 연결해 주세요.

### CASE 24. 중복 차량 번호로 출차 제한
조건:
- duplicate_plate_detected == true
ASK_QUESTION:
- "차량 번호 중복 안내가 나오나요?"
결론:
- 자동으로 해결하기 어렵습니다.
- 관제에 연결해 확인을 요청해 주세요.

### CASE 25. 출차 인증은 되었으나 차단기 반응 없음
조건:
- exit_auth_completed == true
- barrier_not_open == true
ASK_QUESTION:
- "출차 인증 완료 안내는 나왔나요?"
결론:
- 잠시 기다린 뒤 다시 시도해 주세요.
- 반응이 없으면 관제에 연결해 주세요.

### CASE 26. 출차 가능 시간 외 시도
조건:
- exit_time_restricted == true
ASK_QUESTION:
- "현재 출차 가능 시간이 아닌가요?"
결론:
- 운영 시간 내에 출차를 시도해 주세요.
- 긴급 상황이면 관제에 연결해 주세요.

### CASE 27. 출차 예약은 있으나 아직 유효 시간 아님
조건:
- exit_reservation_exists == true
- exit_time_not_started == true
ASK_QUESTION:
- "출차 예약 시간이 아직 시작되지 않았나요?"
결론:
- 예약 시간 이후에 출차를 시도해 주세요.
- 조정이 필요하면 관제에 연결해 주세요.

### CASE 28. 출차 예약 시간이 초과됨
조건:
- exit_reservation_exists == true
- exit_time_expired == true
ASK_QUESTION:
- "출차 예약 시간이 이미 지난 상태인가요?"
결론:
- 관제에 연결해 출차 처리를 요청해 주세요.

### CASE 29. 출구 혼잡으로 일시 정지
조건:
- exit_lane_congested == true
ASK_QUESTION:
- "출구가 혼잡해 대기 중이라는 안내가 나오나요?"
결론:
- 앞차가 완전히 빠져나갈 때까지 잠시 기다려 주세요.

### CASE 30. 출차 절차를 중간에 취소함
조건:
- exit_process_cancelled == true
ASK_QUESTION:
- "출차를 시도하다가 중간에 취소하셨나요?"
결론:
- 다시 출구로 이동해 출차를 처음부터 시도해 주세요.
- 문제가 반복되면 관제에 연결해 주세요.
