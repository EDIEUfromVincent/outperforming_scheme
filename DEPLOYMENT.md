# GitHub and Server Deployment

## GitHub

This repository should store source code and lightweight configuration only.
Secrets, uploaded files, vector indexes, OCR cache, SQLite data, notebooks, and PDFs are excluded by `.gitignore`.

```bash
git init
git add .
git commit -m "Initial teacher exam intelligence app"
git branch -M main
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git push -u origin main
```

Before pushing, rotate any API keys that were ever pasted into local notebooks or shell scripts.

## Local Services

Run the main education coach backend:

```bash
cd /Users/vincent/Downloads/project
/opt/anaconda3/bin/python main.py
```

Run the Project 3 notice/digest backend:

```bash
cd /Users/vincent/Downloads/project/project3
/opt/anaconda3/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8020
```

Run Streamlit:

```bash
cd /Users/vincent/Downloads/project
API_URL=http://localhost:8010 PROJECT3_API_URL=http://localhost:8020 streamlit run app.py
```

## External Tools such as Runway

External services cannot call `localhost` on your laptop. Use one of these approaches:

1. Deploy the FastAPI backend to a public host such as Render, Railway, Fly.io, or a VPS.
2. For quick testing, expose the local backend with a tunnel such as ngrok or Cloudflare Tunnel.
3. Put the public backend URL into Streamlit with `API_URL` and `PROJECT3_API_URL`.

Useful API endpoints:

- `GET /health`
- `POST /upload-document`
- `POST /query/stream`
- `POST /documents`
- `GET /notices` on the Project 3 service
- `POST /digest/preview` on the Project 3 service

For a hosted setup, configure secrets in the hosting provider dashboard instead of committing `.env`.
