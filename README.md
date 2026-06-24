# Elementary Teacher Exam Intelligence

초등임용 교육과정 QA, 2015/2022 교육과정 비교, 수업-임용 통합 코치, 학습 대시보드, 임용 공지 브리핑을 묶은 Streamlit + FastAPI 앱입니다.

## Services

- `app.py`: Streamlit frontend
- `main.py`: main FastAPI backend on port `8010`
- `project3/main.py`: notice and daily digest FastAPI backend on port `8020`

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
pip install -r project3/requirements.txt
```

Run the main backend:

```bash
/opt/anaconda3/bin/python main.py
```

Run the Project 3 backend:

```bash
cd project3
/opt/anaconda3/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8020
```

Run Streamlit:

```bash
API_URL=http://localhost:8010 PROJECT3_API_URL=http://localhost:8020 streamlit run app.py
```

## Environment

Copy `.env.example` to `.env` locally, or set the same keys as deployment secrets.

Do not commit `.env`, uploaded files, OCR cache, FAISS indexes, SQLite files, notebooks, or PDFs.

## External Integration

External tools such as Runway cannot call your laptop's `localhost` directly. Deploy the FastAPI backend to a public host or expose it temporarily with a tunnel, then use the public URL.

See [DEPLOYMENT.md](DEPLOYMENT.md) for GitHub and server setup.
