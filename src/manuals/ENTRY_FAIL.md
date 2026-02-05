# ENTRY_FLOW_ISSUE

### CASE 1. 입구에서 차단기 미개방 (기본)
조건:
- entry_gate_not_open == true
ASK_QUESTION:
- "지금 입구에서 차단기가 올라가지 않나요?"
결론:
- 차량을 정지선에 맞춰 정확히 정차한 뒤 잠시 기다려 주세요.
- 그래도 열리지 않으면 출입구 호출 버튼을 눌러 관제와 연결해 주세요.

### CASE 2. 앞차 통과 직후 내 차에서 차단기 미개방
조건:
- front_car_entered == true
- entry_gate_not_open == true
ASK_QUESTION:
- "앞차는 방금 정상적으로 입차했나요?"
결론:
- 차량을 앞뒤로 조금 이동한 뒤 다시 정차해 주세요.
- 같은 문제가 반복되면 관제에 연결해 주세요.

### CASE 3. 차량 위치/거리 문제로 인식 실패
조건:
- vehicle_position_misaligned == true
ASK_QUESTION:
- "차량을 정지선보다 너무 앞/뒤로 세우신 상태인가요?"
결론:
- 차량을 정지선에 맞춰 다시 정차해 주세요.
- 조금 뒤로 물러난 뒤 천천히 다시 접근해 주세요.

### CASE 4. 번호판 인식 방식인데 인식이 안 됨
조건:
- entry_mode == "LPR"
- lpr_recognition_failed == true
ASK_QUESTION:
- "입차가 번호판 인식 방식인가요?"
- "번호판 인식이 되지 않는다는 안내가 나오나요?"
결론:
- 번호판이 가려지거나 오염되지 않았는지 확인해 주세요.
- 천천히 후진한 뒤 다시 입차를 시도해 주세요.
- 해결되지 않으면 관제에 연결해 주세요.

### CASE 5. 등록되지 않은 차량/방문자라 입차 불가
조건:
- not_registered_message == true
ASK_QUESTION:
- "등록되지 않은 차량/방문자라는 안내가 나오나요?"
결론:
- 방문자 등록 또는 차량 등록이 필요한 경우 등록 후 다시 입차해 주세요.
- 등록 방법을 모르겠으면 관제에 연결해 주세요.

### CASE 6. 정기권/입주/직원 차량인데 인증이 안 됨
조건:
- membership_expected == true
- membership_not_recognized == true
ASK_QUESTION:
- "정기권/입주/직원 차량인데 인증이 안 된다는 안내가 나오나요?"
결론:
- 차량 번호가 등록된 번호와 동일한지 확인해 주세요.
- 문제가 계속되면 관제에 연결해 주세요.

### CASE 7. 입구에서 카드/태그 인증을 시도했으나 실패
조건:
- entry_card_tag_tried == true
- entry_card_tag_failed == true
ASK_QUESTION:
- "입구에서 카드나 태그로 인증을 시도하셨나요?"
- "인증이 실패하거나 반응이 없었나요?"
결론:
- 카드/태그 방향을 바꿔 다시 시도해 주세요.
- 다른 인증 수단이 있으면 이용해 주세요.
- 반복되면 관제에 연결해 주세요.

### CASE 8. 입구 발권기(티켓) 관련 문제
조건:
- ticket_issue == true
ASK_QUESTION:
- "입구에서 티켓 발권이 되지 않거나 티켓이 나오지 않나요?"
결론:
- 발권 버튼이 있다면 한 번만 눌러 잠시 기다려 주세요.
- 계속 발권이 되지 않으면 관제에 연결해 주세요.

### CASE 9. 입구 단말기/화면 먹통 (관리자 이관)
조건:
- entry_terminal_unresponsive == true
ASK_QUESTION:
- "입구 화면이 멈추거나 터치가 되지 않나요?"
결론:
- 자동으로 해결할 수 없는 상황입니다.
- 즉시 관리자 또는 관제에 연결해 주세요.

### CASE 10. 네트워크/서버 오류로 입차 진행 불가 (관리자 이관)
조건:
- network_error_message == true
- entry_process_blocked == true
ASK_QUESTION:
- "서버 오류나 통신 오류 메시지가 표시되나요?"
결론:
- 시스템 문제로 자동 해결이 불가능합니다.
- 관리자 또는 관제에 연결해 주세요.

### CASE 11. 입차 방법 자체를 모름
조건:
- entry_method_unknown == true
ASK_QUESTION:
- "입차 방법을 잘 모르시겠나요?"
결론:
- 입구로 이동해 정지선에 정차하면 차량 인식 후 차단기가 자동으로 열립니다.
- 인증(번호판/카드/티켓)이 필요한 경우 안내에 따라 인증 후 입차해 주세요.

### CASE 12. 번호판이 임시 가림/훼손 상태
조건:
- entry_mode == "LPR"
- license_plate_obstructed == true
ASK_QUESTION:
- "번호판이 가려지거나 임시로 덮여 있지는 않나요?"
결론:
- 번호판을 깨끗이 노출한 뒤 다시 입차를 시도해 주세요.
- 해결되지 않으면 관제에 연결해 주세요.

### CASE 13. 야간/역광 등 환경 문제로 번호판 인식 실패
조건:
- entry_mode == "LPR"
- poor_lighting_condition == true
ASK_QUESTION:
- "주변이 어둡거나 역광 상태인가요?"
결론:
- 전조등을 끄거나 차량 각도를 조금 조정한 뒤 다시 시도해 주세요.
- 계속 실패하면 관제에 연결해 주세요.

### CASE 14. 비·눈 등 날씨 영향으로 인식 실패
조건:
- bad_weather_condition == true
- entry_gate_not_open == true
ASK_QUESTION:
- "비나 눈 때문에 시야가 좋지 않나요?"
결론:
- 잠시 정차한 뒤 다시 시도해 주세요.
- 인식이 되지 않으면 관제에 연결해 주세요.

### CASE 15. 차종/번호판 규격 문제
조건:
- unusual_vehicle_plate == true
ASK_QUESTION:
- "번호판이 일반 차량과 다른 형태인가요?"
결론:
- 자동 인식이 어려울 수 있습니다.
- 관제에 연결해 수동 처리를 요청해 주세요.

### CASE 16. 이전 차량 입차 처리 지연
조건:
- previous_entry_processing == true
ASK_QUESTION:
- "앞차 처리 후 바로 입차를 시도하셨나요?"
결론:
- 잠시 기다린 뒤 다시 입차를 시도해 주세요.
- 반복되면 관제에 연결해 주세요.

### CASE 17. 차량이 너무 빠르게 접근함
조건:
- vehicle_approach_too_fast == true
ASK_QUESTION:
- "입구로 빠르게 접근하셨나요?"
결론:
- 천천히 접근해 정지선에 정차해 주세요.
- 다시 시도해 주세요.

### CASE 18. 차량이 너무 멀리 정차함
조건:
- vehicle_too_far_from_sensor == true
ASK_QUESTION:
- "정지선보다 많이 뒤에 정차하셨나요?"
결론:
- 차량을 조금 앞으로 이동해 다시 정차해 주세요.

### CASE 19. 입구 차로 잘못 진입
조건:
- wrong_entry_lane == true
ASK_QUESTION:
- "차량 종류에 맞는 입구 차로로 진입하셨나요?"
결론:
- 안내 표지에 맞는 입구로 이동해 주세요.
- 필요하면 관제에 연결해 주세요.

### CASE 20. 입차 가능 시간 외 이용 시도
조건:
- entry_time_restricted == true
ASK_QUESTION:
- "현재 입차 가능 시간이 아닌가요?"
결론:
- 운영 시간 내에 입차를 시도해 주세요.
- 긴급 상황이면 관제에 연결해 주세요.

### CASE 21. 방문 예약은 했으나 아직 유효 시간이 아님
조건:
- visit_registered == true
- visit_time_not_started == true
ASK_QUESTION:
- "방문 예약 시간이 아직 시작되지 않았나요?"
결론:
- 예약 시간 이후에 입차를 시도해 주세요.
- 조정이 필요하면 관제에 연결해 주세요.

### CASE 22. 방문 예약 시간이 초과됨
조건:
- visit_registered == true
- visit_time_expired == true
ASK_QUESTION:
- "방문 예약 시간이 이미 지난 상태인가요?"
결론:
- 재등록 후 입차를 시도해 주세요.
- 필요하면 관제에 연결해 주세요.

### CASE 23. 차량 번호 일부만 일치
조건:
- partial_plate_match == true
ASK_QUESTION:
- "번호판 일부만 일치한다는 안내가 나오나요?"
결론:
- 등록된 차량 번호를 다시 확인해 주세요.
- 수정이 필요하면 관제에 연결해 주세요.

### CASE 24. 동일 번호판 중복 등록 문제
조건:
- duplicate_plate_detected == true
ASK_QUESTION:
- "번호판 중복 등록 안내가 나오나요?"
결론:
- 자동으로 해결하기 어렵습니다.
- 관제에 연결해 확인을 요청해 주세요.

### CASE 25. 입차 인증 후 반응 지연
조건:
- entry_auth_delayed == true
ASK_QUESTION:
- "인증은 했는데 반응이 늦게 나오나요?"
결론:
- 잠시 기다려 주세요.
- 계속 반응이 없으면 관제에 연결해 주세요.

### CASE 26. 입구 경고음만 울리고 진행 안 됨
조건:
- entry_warning_sound_only == true
ASK_QUESTION:
- "경고음만 울리고 진행이 안 되나요?"
결론:
- 차량을 정차한 상태로 유지해 주세요.
- 관제에 연결해 안내를 받아 주세요.

### CASE 27. 입구 안전 차단으로 일시 중단
조건:
- safety_lock_active == true
ASK_QUESTION:
- "안전 차단으로 입차가 제한된다는 안내가 나오나요?"
결론:
- 잠시 대기해 주세요.
- 안내가 없으면 관제에 연결해 주세요.

### CASE 28. 입구 설비 점검 중
조건:
- entry_maintenance_mode == true
ASK_QUESTION:
- "입구 설비 점검 중이라는 안내가 보이나요?"
결론:
- 다른 입구를 이용해 주세요.
- 안내가 없으면 관제에 연결해 주세요.
