# PAYMENT_FAIL

### CASE 1. 카드 승인 실패
조건:
- card_payment_tried == true
- card_approval_failed == true
ASK_QUESTION:
- "카드로 결제를 시도하셨나요?"
- "카드 승인 실패 또는 거절 메시지가 나왔나요?"
결론:
- 다른 카드로 다시 결제를 시도해 주세요.
- 다른 결제 수단이나 무인정산기(키오스크)를 이용해 주세요.
- 같은 문제가 반복되면 관리자 또는 관제에 연결해 주세요.

### CASE 2. 카드 인식은 되었으나 결제 진행 안 됨
조건:
- card_tagged == true
- payment_not_started == true
ASK_QUESTION:
- "카드를 단말기에 인식시키셨나요?"
- "결제 화면으로 넘어가지 않았나요?"
결론:
- 카드를 다시 한 번 인식 위치에 정확히 대고 시도해 주세요.
- 다른 카드로 재시도해 주세요.
- 계속 진행되지 않으면 관제에 연결해 주세요.

### CASE 3. 카드 인식 자체가 되지 않음
조건:
- card_tagged == true
- card_not_detected == true
ASK_QUESTION:
- "카드를 단말기에 대도 아무 반응이 없었나요?"
결론:
- 카드 방향을 바꿔 다시 인식시켜 주세요.
- 카드 인식부에 정확히 밀착해 주세요.
- 다른 카드로 시도해 보시고, 문제가 지속되면 관제에 연결해 주세요.

### CASE 4. 카드 결제 중 오류 메시지 표시 (관리자 이관)
조건:
- card_error_message == true
ASK_QUESTION:
- "결제 중 오류 메시지가 화면에 표시되나요?"
결론:
- 시스템 오류로 자동 해결이 불가능합니다.
- 즉시 관리자 또는 관제에 연결해 주세요.

### CASE 5. 결제 도중 화면 멈춤 또는 먹통 (관리자 이관)
조건:
- payment_terminal_unresponsive == true
ASK_QUESTION:
- "결제 중 화면이 멈추거나 터치가 되지 않나요?"
결론:
- 자동으로 해결할 수 없는 상황입니다.
- 즉시 관리자 또는 관제에 연결해 주세요.

### CASE 6. 결제는 되었으나 완료 화면이 나오지 않음
조건:
- card_approved == true
- payment_completion_not_displayed == true
ASK_QUESTION:
- "카드 승인은 되었는데 결제 완료 화면이 나오지 않았나요?"
결론:
- 잠시 기다린 뒤 화면 상태를 확인해 주세요.
- 출차가 되지 않으면 관제에 연결해 결제 상태를 확인받아 주세요.

### CASE 7. 카드 결제 방법 자체를 모름
조건:
- payment_method_unknown == true
ASK_QUESTION:
- "카드 결제 방법을 잘 모르시겠나요?"
결론:
- 화면 안내에 따라 카드를 결제 단말기에 인식시켜 주세요.
- 결제가 완료되면 출차를 다시 시도해 주세요.

### CASE 8. 카드 승인 지연
조건:
- card_payment_tried == true
- card_approval_delayed == true
ASK_QUESTION:
- "카드 결제 후 승인 대기 상태가 오래 지속되나요?"
결론:
- 잠시 기다린 뒤 결제 상태를 확인해 주세요.
- 화면이 변하지 않으면 관제에 연결해 주세요.

### CASE 9. 카드 승인 후 중복 결제 우려
조건:
- card_approved == true
- duplicate_payment_warning == true
ASK_QUESTION:
- "결제 후 중복 결제 안내가 나오나요?"
결론:
- 추가 결제를 시도하지 말고 잠시 기다려 주세요.
- 관제에 연결해 결제 상태를 확인받아 주세요.

### CASE 10. 카드 결제 취소 후 재결제 실패
조건:
- payment_cancelled == true
- card_payment_failed == true
ASK_QUESTION:
- "결제를 취소한 뒤 다시 시도하셨나요?"
결론:
- 잠시 기다린 뒤 다시 결제를 시도해 주세요.
- 계속 실패하면 관제에 연결해 주세요.

### CASE 11. 카드 한도 초과로 결제 실패
조건:
- card_limit_exceeded_message == true
ASK_QUESTION:
- "카드 한도 초과 안내가 표시되나요?"
결론:
- 다른 카드로 결제를 시도해 주세요.
- 다른 결제 수단을 이용해 주세요.

### CASE 12. 카드 사용 불가(정지/만료)
조건:
- card_invalid_message == true
ASK_QUESTION:
- "카드 사용 불가 또는 만료 안내가 나오나요?"
결론:
- 사용 가능한 다른 카드로 결제를 시도해 주세요.
- 다른 결제 수단을 이용해 주세요.

### CASE 13. 카드 비밀번호 입력 오류
조건:
- pin_input_failed == true
ASK_QUESTION:
- "카드 비밀번호 오류 안내가 나오나요?"
결론:
- 비밀번호를 다시 정확히 입력해 주세요.
- 계속 오류가 나면 다른 결제 수단을 이용해 주세요.

### CASE 14. 서명/확인 단계에서 진행 안 됨
조건:
- payment_confirmation_blocked == true
ASK_QUESTION:
- "결제 확인 단계에서 진행이 멈췄나요?"
결론:
- 화면 안내에 따라 확인을 완료해 주세요.
- 진행되지 않으면 관제에 연결해 주세요.

### CASE 15. 카드 결제 도중 네트워크 불안정
조건:
- network_error_message == true
- card_payment_tried == true
ASK_QUESTION:
- "결제 중 통신 오류 안내가 표시되나요?"
결론:
- 시스템 문제로 자동 해결이 어렵습니다.
- 관제에 연결해 결제 상태를 확인받아 주세요.

### CASE 16. 카드 결제 후 영수증/완료 표시 없음
조건:
- receipt_not_displayed == true
ASK_QUESTION:
- "결제 후 영수증이나 완료 표시가 보이지 않나요?"
결론:
- 결제는 정상 처리되었을 수 있습니다.
- 출차가 되지 않으면 관제에 연결해 확인해 주세요.

### CASE 17. 결제 금액 확인 단계에서 중단됨
조건:
- payment_amount_confirmation_blocked == true
ASK_QUESTION:
- "결제 금액 확인 화면에서 멈췄나요?"
결론:
- 금액을 확인한 뒤 결제를 다시 진행해 주세요.
- 화면이 넘어가지 않으면 관제에 연결해 주세요.

### CASE 18. 카드 결제는 되었으나 출차 불가
조건:
- card_approved == true
- exit_blocked_after_payment == true
ASK_QUESTION:
- "결제는 완료됐는데 출차가 되지 않나요?"
결론:
- 잠시 기다린 뒤 다시 출차를 시도해 주세요.
- 계속 안 되면 관제에 연결해 주세요.

### CASE 19. 카드 결제 중 사용자가 중단
조건:
- payment_interrupted_by_user == true
ASK_QUESTION:
- "결제 도중 취소하거나 자리를 이동하셨나요?"
결론:
- 결제를 처음부터 다시 시도해 주세요.
- 문제가 반복되면 관제에 연결해 주세요.

### CASE 20. 카드 결제 시도 횟수 초과
조건:
- payment_attempt_limit_exceeded == true
ASK_QUESTION:
- "결제를 여러 번 시도하셨나요?"
결론:
- 잠시 기다린 뒤 다시 시도해 주세요.
- 다른 결제 수단을 이용하거나 관제에 연결해 주세요.

### CASE 21. 카드 결제 단말기 점검 중
조건:
- payment_maintenance_mode == true
ASK_QUESTION:
- "결제 단말기 점검 중이라는 안내가 나오나요?"
결론:
- 다른 결제 수단이나 무인정산기를 이용해 주세요.
- 안내가 없으면 관제에 연결해 주세요.

### CASE 22. 결제 수단 선택 화면에서 진행 안 됨
조건:
- payment_method_selection_blocked == true
ASK_QUESTION:
- "결제 수단 선택 화면에서 진행이 안 되나요?"
결론:
- 화면을 다시 확인한 뒤 결제 수단을 선택해 주세요.
- 계속 진행되지 않으면 관제에 연결해 주세요.

### CASE 23. 결제 완료 후 금액 재표시 오류
조건:
- payment_amount_display_error == true
ASK_QUESTION:
- "결제 후 금액 표시가 이상하게 나오나요?"
결론:
- 중복 결제를 시도하지 마세요.
- 관제에 연결해 결제 상태를 확인받아 주세요.

### CASE 24. 카드 결제 대신 다른 수단을 원함
조건:
- user_wants_other_payment_method == true
ASK_QUESTION:
- "다른 결제 수단을 이용하고 싶으신가요?"
결론:
- 무인정산기나 다른 결제 수단을 이용해 주세요.
- 안내가 필요하면 관제에 연결해 주세요.
