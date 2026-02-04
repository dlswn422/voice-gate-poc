# 매뉴얼 모음 v2 (RAG/슬롯 수집 최적화) - MANUALS_V2

## 사용 목적
- 2차 상담 모델이 슬롯(symptom/where/...)을 채우며 문제를 특정하고, 충분히 특정되면 해결책을 제시하도록 돕는 매뉴얼입니다.
- 각 문서는 required_slots / optional_slots / ASK_1 / SOLVE_TEMPLATE / 에스컬레이션 조건을 포함합니다.
## 문서 목록
- barrier_physical_fault.md
- discount_free_time_issue.md
- entry_gate_not_open.md
- exit_gate_not_open.md
- failsafe_done.md
- kiosk_ui_device_issue.md
- lpr_mismatch_or_no_entry_record.md
- mobile_payment_qr_issue.md
- network_terminal_down.md
- payment_card_fail.md
- price_inquiry.md
- visit_registration_fail.md
## 작성 규칙
- 조치 문장은 가능한 한 '~하세요/~해 주세요' 형태로 작성(모델이 ALLOWED_ACTIONS로 쉽게 추출)
- 오류 문구는 사용자가 보는 그대로 예시로 포함(검색 성능 향상)
- 헤더/태그는 답변에 그대로 복사하지 않도록 주의(프롬프트에서도 금지)
