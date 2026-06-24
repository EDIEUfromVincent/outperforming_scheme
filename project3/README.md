# Project 3. 초등임용 일일 브리핑 메일러

목표: 기존 앱의 `뉴스 검색`을 임용고시 수험생에게 의미 있는 기능으로 바꾼다.

매일 다음 세 가지를 한 통의 이메일로 묶는다.

1. 기출문제 1문제
2. 성취기준 기반 예비문제 1문제
3. 임용 공지·교육과정 소식

기본 수신자: `ohjinwoo9696@gmail.com`

## 핵심 출처

`sources.json`에 다음 계열을 넣어 두었다.

- 한국교육과정평가원: 기출·시험자료
- 교육부: 교육과정·정책 고시/보도자료
- 국가교육과정정보센터: 2015/2022 교육과정 원문
- 온라인 교직원 채용: 시도교육청 임용 공고 진입점
- 17개 시·도교육청: 지역별 초등 임용 공고

시·도교육청 전체 수집은 시간이 걸릴 수 있으므로 기본값은 꺼져 있다.

## 파일 구조

```text
project3/
├── app.py               # Streamlit UI
├── main.py              # FastAPI API
├── notice_crawler.py    # 공식 사이트 공지 후보 수집
├── daily_digest.py      # 기출 1문제 + 예비문제 + 뉴스 메일 본문 생성
├── email_sender.py      # Gmail SMTP 발송 유틸
├── send_daily_email.py  # CLI 미리보기/발송
├── sources.json         # 핵심 사이트 목록
└── harness.py           # 로컬 검증
```

## 실행

프로젝트 루트(`/Users/vincent/Downloads/project`)에서 실행한다.

```bash
cd project3
python main.py
```

다만 `project3/main.py`는 FastAPI 앱 파일이므로 보통은 아래처럼 실행한다.

```bash
/opt/anaconda3/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8020
```

다른 터미널에서:

```bash
streamlit run app.py --server.port 8502
```

브라우저:

```text
http://localhost:8502
```

## CLI 미리보기

```bash
cd project3
python send_daily_email.py
```

실제 발송:

```bash
python send_daily_email.py --send
```

시·도교육청까지 포함:

```bash
python send_daily_email.py --include-regions
```

## Gmail 설정

루트 `.env` 또는 `project3/.env`에 다음을 넣는다.

```text
EMAIL_ADDRESS=발신Gmail주소
EMAIL_PASSWORD=Gmail앱비밀번호16자리
EMAIL_RECIPIENT=ohjinwoo9696@gmail.com
```

주의:

- `EMAIL_PASSWORD`는 일반 Gmail 로그인 비밀번호가 아니라 앱 비밀번호다.
- Google 계정에서 2단계 인증을 켠 뒤 앱 비밀번호를 생성해야 한다.
- 실제 메일 발송은 Streamlit에서 확인 체크박스를 누르거나 CLI에 `--send`를 붙였을 때만 실행된다.

## 자동화

macOS에서 매일 아침 7시에 보내려면 cron 또는 launchd를 쓸 수 있다.

예시 cron:

```text
0 7 * * * cd /Users/vincent/Downloads/project/project3 && /opt/anaconda3/bin/python send_daily_email.py --send >> daily_email.log 2>&1
```

## 검증

```bash
python harness.py
```

네트워크가 막혀도 메일 본문 생성, 기출 선택, 예비문제 생성은 통과해야 한다.
