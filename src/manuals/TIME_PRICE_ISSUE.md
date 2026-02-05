# TIME_PRICE_FAIL

### CASE 1. 무료주차 대상인데 요금이 발생함
조건:
- free_time_expected == true
- free_time_not_applied == true
ASK_QUESTION:
- "무료주차 대상이신가요?"
- "요금이 발생한 것으로 표시되나요?"
결론:
- 무료주차 대상 등록이 정상적으로 되었는지 확인해 주세요.
- 등록이 되어 있다면 요금 내역을 다시 확인해 주세요.
- 문제가 해결되지 않으면 관제에 연결해 주세요.

### CASE 2. 할인 대상인데 할인이 적용되지 않음
조건:
- discount_expected == true
- discount_not_applied == true
ASK_QUESTION:
- "할인 대상이신가요?"
- "요금에 할인이 적용되지 않았나요?"
결론:
- 할인 등록이 정상적으로 되었는지 다시 확인해 주세요.
- 할인 적용 후 다시 정산을 시도해 주세요.
- 적용이 되지 않으면 관제에 연결해 주세요.

### CASE 3. 주차 시간이 실제보다 길게 계산됨
조건:
- parking_time_excessive == true
ASK_QUESTION:
- "실제 주차 시간보다 더 길게 계산된 것 같으신가요?"
결론:
- 입차·출차 시간을 한 번 더 확인해 주세요.
- 시간이 맞지 않으면 관제에 연결해 확인을 요청해 주세요.

### CASE 4. 요금이 예상보다 과다함
조건:
- fee_excessive == true
ASK_QUESTION:
- "예상하신 요금보다 많이 나온 것 같으신가요?"
결론:
- 요금 내역을 다시 확인해 주세요.
- 요금이 납득되지 않으면 관제에 연결해 주세요.

### CASE 5. 정산 금액이 갑자기 변경됨
조건:
- fee_changed_unexpectedly == true
ASK_QUESTION:
- "정산 금액이 갑자기 달라졌나요?"
결론:
- 할인 또는 무료시간 적용 여부를 다시 확인해 주세요.
- 변경 사유가 확인되지 않으면 관제에 연결해 주세요.

### CASE 6. 주차 요금 계산 기준을 모름
조건:
- pricing_rule_unknown == true
ASK_QUESTION:
- "주차 요금이 어떻게 계산되는지 모르시겠나요?"
결론:
- 주차 요금은 이용 시간에 따라 자동으로 계산됩니다.
- 상세 요금 기준은 안내판 또는 관제를 통해 확인하실 수 있습니다.

### CASE 7. 무료시간/할인 적용 방법을 모름
조건:
- discount_application_method_unknown == true
ASK_QUESTION:
- "무료시간이나 할인 적용 방법을 잘 모르시겠나요?"
결론:
- 무료주차나 할인은 사전에 등록되거나 인증이 필요합니다.
- 적용 방법 안내가 필요하면 관제에 연결해 주세요.

### CASE 8. 무료시간은 끝났는데 아직 남은 줄 앎
조건:
- free_time_expected == true
- free_time_expired == true
ASK_QUESTION:
- "무료주차 시간이 아직 남아 있다고 생각하셨나요?"
결론:
- 무료주차 시간은 정해진 시간까지만 적용됩니다.
- 현재 요금 내역을 확인해 주세요.

### CASE 9. 할인은 적용됐으나 기대한 금액보다 적음
조건:
- discount_applied == true
- discount_amount_less_than_expected == true
ASK_QUESTION:
- "할인은 적용됐지만 금액이 생각보다 적나요?"
결론:
- 할인 조건과 적용 시간을 다시 확인해 주세요.
- 할인 기준이 다를 수 있으니 요금 내역을 확인해 주세요.

### CASE 10. 여러 할인 중 일부만 적용됨
조건:
- multiple_discounts_expected == true
- partial_discount_applied == true
ASK_QUESTION:
- "여러 할인을 기대하셨는데 일부만 적용됐나요?"
결론:
- 할인은 중복 적용이 제한될 수 있습니다.
- 적용된 할인 내역을 확인해 주세요.
- 필요하면 관제에 연결해 주세요.

### CASE 11. 할인 적용 후 요금이 다시 증가함
조건:
- discount_applied == true
- fee_increased_after_discount == true
ASK_QUESTION:
- "할인 적용 후 다시 요금이 늘어난 것처럼 보이나요?"
결론:
- 할인 이후 추가 시간이 발생했을 수 있습니다.
- 현재 주차 시간을 다시 확인해 주세요.

### CASE 12. 입차 시각이 실제보다 늦게 기록됨
조건:
- entry_time_recorded_late == true
ASK_QUESTION:
- "입차 시간이 실제보다 늦게 찍힌 것 같으신가요?"
결론:
- 입차 기록 시각을 확인해 주세요.
- 차이가 크면 관제에 연결해 확인을 요청해 주세요.

### CASE 13. 출차 시각이 실제보다 빠르게 기록됨
조건:
- exit_time_recorded_early == true
ASK_QUESTION:
- "출차 시간이 실제보다 빠르게 기록된 것 같으신가요?"
결론:
- 출차 기록 시각을 확인해 주세요.
- 문제가 있으면 관제에 연결해 주세요.

### CASE 14. 입차 기록 누락으로 요금이 과다 계산됨
조건:
- entry_record_missing == true
ASK_QUESTION:
- "입차 기록이 없거나 이상하다고 나오나요?"
결론:
- 자동으로 확인이 어렵습니다.
- 관제에 연결해 입차 기록 확인을 요청해 주세요.

### CASE 15. 정산 직후 요금이 다시 표시됨
조건:
- payment_completed == true
- fee_displayed_again == true
ASK_QUESTION:
- "정산을 했는데 다시 요금이 표시되나요?"
결론:
- 반영 지연일 수 있으니 잠시 기다려 주세요.
- 출차가 되지 않으면 관제에 연결해 주세요.

### CASE 16. 시간 단위 요금 경계에서 요금 증가
조건:
- pricing_boundary_crossed == true
ASK_QUESTION:
- "시간 단위가 바뀌면서 요금이 오른 것 같나요?"
결론:
- 요금은 시간 단위 기준으로 계산됩니다.
- 경계 시점에 요금이 증가할 수 있습니다.

### CASE 17. 요금이 갑자기 초기화되거나 0원으로 표시됨
조건:
- fee_reset_or_zero_displayed == true
ASK_QUESTION:
- "요금이 갑자기 0원이나 초기값으로 표시되나요?"
결론:
- 표시 오류일 수 있습니다.
- 정산 전 관제에 연결해 요금 상태를 확인해 주세요.

### CASE 18. 하루 최대 요금(상한) 적용이 안 된 것처럼 보임
조건:
- daily_cap_expected == true
- daily_cap_not_applied == true
ASK_QUESTION:
- "하루 최대 요금이 적용될 거라고 생각하셨나요?"
결론:
- 최대 요금 적용 조건을 다시 확인해 주세요.
- 조건에 해당하면 관제에 연결해 확인을 요청해 주세요.

### CASE 19. 장시간 주차로 요금이 예상보다 큼
조건:
- long_term_parking == true
ASK_QUESTION:
- "장시간 주차로 요금이 많이 나온 것 같나요?"
결론:
- 주차 시간에 따라 요금이 누적 계산됩니다.
- 요금 내역을 확인해 주세요.

### CASE 20. 요금 내역 자체를 이해하지 못함
조건:
- fee_breakdown_unknown == true
ASK_QUESTION:
- "요금이 어떻게 구성됐는지 이해하기 어려우신가요?"
결론:
- 요금은 주차 시간과 할인 여부에 따라 계산됩니다.
- 상세 내역은 관제를 통해 확인하실 수 있습니다.
