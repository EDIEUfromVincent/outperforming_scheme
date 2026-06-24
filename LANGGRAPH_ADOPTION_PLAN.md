# LangGraph Adoption Plan

이 문서는 현재 Streamlit/FastAPI/LangChain 앱을 세컨드 브레인형 임용 학습 시스템으로 확장하기 위한 LangGraph 도입 기준과 순서를 정리한다.

## 결론

LangGraph는 모든 LLM 호출을 대체하는 계층이 아니다. 이 프로젝트에서 LangGraph가 맡아야 할 역할은 다음과 같다.

- 여러 단계가 있는 장기 작업의 상태 전이 관리
- 실패한 node 단위 재시도와 작업 복구
- OCR, 성취기준 매핑, 메일 발송처럼 사람 검토가 필요한 지점의 interrupt/review/resume
- 업로드 문서, 교육과정, 기출, Obsidian note, 복습 이력을 연결하는 장기 workflow
- 각 답변의 근거, 페이지, 성취기준, 불확실성 라벨을 검증하는 auditor pipeline

LangChain은 여전히 retrieval, prompt, parser, model wrapper에 사용한다. FastAPI는 HTTP 경계와 권한/파일 처리에 사용한다. LangGraph는 이 둘을 묶는 workflow runtime이 된다.

공식 문서 기준으로도 이 판단이 맞다. LangGraph persistence는 checkpointer로 thread state를, store로 장기 데이터를 저장한다. interrupt는 실행을 멈추고 같은 thread에서 사람 입력으로 재개하는 구조다.

- https://docs.langchain.com/oss/python/langgraph/persistence
- https://docs.langchain.com/oss/python/langgraph/add-memory
- https://docs.langchain.com/oss/python/langgraph/interrupts
- https://docs.langchain.com/oss/python/langgraph/streaming

## 현재 반영 상태

현재 코드에는 1단계 기반이 들어갔다.

- `langgraph_runtime.py`: 공통 runtime, in-memory checkpointer/store, thread id, health status
- `document_compare_graph.py`: 다문서 비교를 retrieve -> summarize -> synthesize node로 분리
- `/documents/compare`: `thread_id`를 받을 수 있고, 응답에 `thread_id`, `trace`를 반환
- Streamlit: 비교 결과에 workflow/thread/step trace 표시

아직 production-grade persistence는 아니다. 로컬 개발용 in-memory checkpointer라서 서버 재시작 후 복구는 되지 않는다. 상용화 단계에서는 Postgres/Redis/MongoDB 계열 checkpointer로 바꿔야 한다.

## 도입 우선순위

### P0: 바로 LangGraph화해야 하는 영역

1. 검색 전략 라우팅 그래프

현재 질문은 업로드 문서, 교육과정 비교, 기출, 평가기준, 웹 공지, Project 3 메일 브리핑 중 어디로 가야 하는지 단일 함수/버튼 흐름에 의존한다. LangGraph로 `classify_intent -> route -> retrieve -> answer -> verify` 구조를 만들어야 한다.

필요 node:

- classify_intent
- route_uploaded_docs
- route_curriculum
- route_exam_bank
- route_assessment_standard
- route_project3_notice
- merge_context
- answer
- verify_evidence

2. 다문서 비교 그래프

이미 1차 반영했다. 다음 단계에서는 문서별 요약을 fan-out subgraph로 병렬화하고, 근거표/auditor node를 추가한다.

목표 node:

- select_documents
- retrieve_per_document
- summarize_each_document
- extract_issues
- align_commonalities
- align_differences
- evidence_table
- final_synthesis
- evidence_auditor

3. 문서 업로드/색인 그래프

현재 업로드는 한 요청 안에서 저장, OCR, split, vector index까지 이어진다. 대용량 PDF나 OCR 실패가 생기면 요청 전체가 불안정해진다. 이 영역이 LangGraph persistence의 가장 큰 수혜자다.

목표 node:

- save_original
- extract_text
- ocr_if_needed
- quality_check
- classify_document
- split_sections
- build_summary_index
- build_vector_index
- write_manifest
- export_obsidian_candidate

4. OCR 품질 검수 그래프

OCR confidence가 낮은 페이지는 자동 답변에 섞이면 전체 지식베이스를 오염시킨다. 낮은 confidence 페이지에서 interrupt를 걸고 사용자가 수정/승인한 text로 재개해야 한다.

interrupt 지점:

- low_confidence_page_review
- missing_page_text_review
- noisy_table_review

5. Project 3 메일 발송 승인 그래프

메일 발송은 critical action이다. 공지 수집과 초안 생성은 자동화해도, 실제 발송 전에는 approve/reject interrupt를 둬야 한다.

목표 node:

- collect_notices
- filter_relevance
- draft_email
- interrupt_for_approval
- send_email
- log_delivery

6. 장기 작업 복구 그래프

이건 별도 기능이라기보다 모든 장기 graph에 깔리는 운영 원칙이다. 업로드, 다문서 비교, Obsidian sync, 메일 발송은 모두 thread_id를 발급하고 node trace를 남겨야 한다.

필수 메타데이터:

- thread_id
- workflow_name
- node
- status
- input_digest
- output_digest
- source_files
- created_at
- updated_at

### P1: 학습 품질을 크게 올리는 영역

7. 근거 검증 그래프

상용화하려면 답변보다 검증이 더 중요하다. 답변이 그럴듯해도 source/page/code가 맞지 않으면 세컨드 브레인으로 쓸 수 없다.

목표 node:

- draft_answer
- extract_claims
- retrieve_evidence_for_claims
- verify_page_source_code
- label_uncertainty
- revise_answer

8. 성취기준 대응표 검수 그래프

현재 검수 필요 mapping은 자동 판단 결과와 사람 판단이 섞일 가능성이 있다. LangGraph로 자동 제안, 승인, 수정, 기각, 재색인을 명확히 분리한다.

목표 node:

- load_mapping_candidates
- score_mapping_confidence
- group_review_batch
- interrupt_for_human_decision
- apply_decision
- reindex_mapping

9. 수업-임용 코치 멀티에이전트 그래프

현재 `SupervisorAgent.run()`은 총괄, 교과, 학년 학생, 근거, audit를 순차 호출한다. LangGraph로 agent 역할과 감사 흐름을 명시해야 한다.

목표 node:

- supervisor_route
- subject_agent
- grade_student_agent
- assessment_agent
- curriculum_governance_agent
- exam_linker
- auditor
- final_coach_answer

10. 사용자 답안 첨삭 그래프

사용자 답안을 rubric으로 나누고, 빠진 근거를 찾고, 점수/피드백/복습 예약까지 이어야 한다.

목표 node:

- parse_user_answer
- decompose_rubric
- match_required_elements
- detect_missing_elements
- score_answer
- generate_feedback
- schedule_review

11. 학습 복습 그래프

현재 복습/학습 DB는 기능 단위 API에 가깝다. 실제 학습 루프는 graph로 만들어야 한다.

목표 node:

- select_due_items
- generate_recall_question
- collect_user_answer
- evaluate_answer
- adjust_interval
- create_weakness_note

### P2: 세컨드 브레인 상용화에 필요한 확장

12. Obsidian sync 그래프

Obsidian은 단순 export가 아니라 지식베이스의 편집/탐색 interface가 되어야 한다. DB 변경, Markdown note, wikilink, Base, Canvas, 재색인이 한 workflow로 묶여야 한다.

목표 node:

- detect_changed_records
- render_markdown_note
- validate_wikilinks
- update_base
- update_canvas
- reindex_notes
- report_sync_result

## 구현 원칙

1. graph는 HTTP 요청 하나에 갇히면 안 된다.

긴 작업은 `thread_id`를 반환하고, 프론트는 상태 조회/재개를 호출할 수 있어야 한다.

2. LLM이 하는 일과 시스템이 하는 일을 분리한다.

LLM은 요약, 분류, 초안, 평가를 한다. 저장, 발송, 인덱싱, 승인 반영은 deterministic node가 한다.

3. interrupt는 위험하거나 애매한 지점에만 둔다.

모든 node에 사람 검토를 넣으면 자동화 가치가 사라진다. OCR confidence 낮음, 메일 발송, mapping confidence 낮음, 근거 불충분 답변에 한정한다.

4. store에는 장기 기억을, checkpoint에는 진행 상태를 둔다.

checkpoint에는 현재 workflow state를 둔다. store에는 사용자 선호, 문서별 canonical summary, 검수된 mapping, Obsidian note id 같은 cross-thread 데이터를 둔다.

5. trace를 먼저 표준화한다.

LangGraph 도입의 체감 가치는 UI에서 "어디까지 진행됐고 어디서 실패했는지"를 보여주는 데서 나온다.

## 다음 구현 순서

1. 검색 전략 라우팅 그래프 추가
2. 문서 업로드/색인 그래프 추가
3. OCR confidence 기반 interrupt 설계
4. Project 3 메일 승인 interrupt 그래프 추가
5. 근거 검증 graph를 모든 답변 뒤에 붙이기
6. Postgres/Redis/MongoDB checkpointer 중 하나로 production persistence 전환

## 배포 전 필요 조건

- graph_runs 테이블 또는 LangGraph checkpointer backend
- object storage 기반 원본 파일 저장
- vector index 백업/복원 전략
- node별 idempotency 보장
- 발송/삭제/재색인 같은 critical action의 approval log
- thread_id 기반 상태 조회 API
- Streamlit 또는 별도 admin UI에서 interrupt 승인/수정/기각 가능
