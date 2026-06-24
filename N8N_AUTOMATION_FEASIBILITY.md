# Outperforming × n8n 자동화 적용 가능성 점검

현재 프로젝트는 교육과정 PDF OCR, 2015↔2022 성취기준 매핑, 기출문제 파싱, FAISS 기반 비교 검색, Streamlit/FastAPI 화면 검증까지 완료되어 있다. 따라서 n8n은 새 기능을 직접 대체하기보다, 백그라운드 자동화·스케줄링·외부 알림·데이터 파이프라인 역할로 붙이는 것이 가장 적합하다.

## 1. 성취기준 데이터 자동 인제스션

적합도: 매우 높음

현재 이미 존재하는 로컬 처리 모듈:

- `ocr_service.py`
- `batch_ocr.py`
- `curriculum_mapper.py`
- `exam_parser.py`
- `parse_2015_ocr.py`
- `rebuild_vector_db.py`

n8n에서는 새 PDF 감지 → FastAPI ingestion endpoint 호출 → 매핑 재생성 → Vector DB upsert 순서로 자동화할 수 있다.

단, 현재 벡터DB는 로컬 FAISS다. Supabase Vector로 전환하려면 `documents`, `curriculum_standards`, `standard_mappings`, `exam_questions` 테이블과 pgvector upsert 코드가 필요하다.

## 2. 에빙하우스 망각곡선 기반 인출 트리거

적합도: 높음

현재 프로젝트에는 성취기준 단위 메타데이터가 있다. 다만 사용자별 학습 이력 테이블은 아직 없다.

필요 테이블:

- `users`
- `learning_events`
- `review_schedules`
- `standard_master`

n8n은 매일 스케줄러로 `review_schedules.next_review_at <= now()` 항목을 조회하고, 오늘 복습할 성취기준 리스트를 생성할 수 있다.

## 3. 이동평균선 계산 및 지표 생성

적합도: 높음

현재는 퀴즈 풀이 기록이 없으므로 데이터 모델이 먼저 필요하다.

필요 테이블:

- `quiz_attempts`
- `daily_learning_metrics`
- `standard_performance_metrics`

n8n은 매일 정답률, 풀이 시간, 성취기준별 약점 점수, 5일/20일 이동평균을 집계해 대시보드용 테이블에 저장할 수 있다.

## 4. BYOK SaaS 백엔드

적합도: 중간

n8n으로 HTTP 게이트웨이를 만들 수는 있지만, 사용자 API 키를 안전하게 보관하고 호출량·실패·권한을 관리해야 하므로 신중해야 한다.

권장 구조:

- API key는 암호화 저장
- n8n은 직접 키를 노출하지 않고 backend service role로만 접근
- 요청 로그와 비용 추정치를 별도 테이블에 저장

MVP에서는 BYOK보다 현재 `.env` 기반 단일 키 운영이 더 단순하다. 다중 사용자 SaaS로 넘어갈 때 도입하는 것이 좋다.

## 5. 취약 성취기준 기반 변형 문제 자동 생성

적합도: 매우 높음

현재 이미 성취기준 매핑, 관련 기출 검색, 검수 우선순위가 있다. 여기에 `quiz_attempts`와 `weak_standards`만 붙이면 n8n이 약점 성취기준을 감지해 변형 문제 생성을 요청할 수 있다.

권장 흐름:

1. 정답률 하락 또는 이동평균선 하향 돌파 감지
2. 해당 성취기준 원문·2022 변화·관련 기출 검색
3. LLM에 변형 문제 생성 요청
4. `generated_quizzes` 테이블에 저장
5. 사용자의 오늘 인출 리스트에 추가

## 6. 다중 소스 크롤링 및 교육 소식 업데이트

적합도: 중간~높음

기술적으로는 가능하지만, 공지사항 크롤링은 사이트 구조 변경과 저작권·이용약관을 조심해야 한다. 우선은 RSS나 공식 공지 목록 페이지를 대상으로 시작하는 것이 안전하다.

권장 대상:

- 한국교육과정평가원 공지
- 시·도교육청 임용 공고
- 교육부 보도자료

저장 테이블:

- `education_news`
- `news_summaries`
- `notification_events`

## 우선순위 제안

1. Supabase 데이터 모델 설계
2. FastAPI에 n8n용 ingestion endpoint 추가
3. n8n 파일 감지 → ingestion 호출 자동화
4. 학습 로그/퀴즈 시도 테이블 추가
5. 에빙하우스 복습 스케줄러 구현
6. 약점 성취기준 기반 변형 문제 생성
7. BYOK와 외부 공지 크롤링은 SaaS화 단계에서 구현

## 결론

가장 먼저 할 만한 것은 1번, 2번, 5번이다.

- 1번은 현재 코드 자산과 바로 연결된다.
- 2번은 초등임용 수험생의 매일 학습 루프를 만든다.
- 5번은 프로젝트의 핵심 가치인 “약점 성취기준 → 변형 문제 → 인출”로 이어진다.

3번은 학습 기록이 쌓인 뒤 붙이는 것이 좋고, 4번 BYOK와 6번 크롤러는 제품화 후반 단계로 미루는 것이 안전하다.
