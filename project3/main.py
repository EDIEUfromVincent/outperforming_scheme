"""Project 3: 초등임용 공지·교육뉴스·일일문제 메일링 API."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from daily_digest import build_daily_digest
from email_sender import email_config_status, send_email
from notice_crawler import collect_notices, load_sources


app = FastAPI(title="초등임용 일일 브리핑 메일러")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DigestRequest(BaseModel):
    recipient: str = "ohjinwoo9696@gmail.com"
    include_regions: bool = False
    send: bool = False
    seed: str | None = None


@app.get("/")
def root():
    return {
        "service": "초등임용 일일 브리핑 메일러",
        "endpoints": {
            "sources": "/sources",
            "notices": "/notices",
            "digest_preview": "/digest/preview",
            "digest_email": "/digest/email",
            "email_status": "/email/status",
        },
    }


@app.get("/sources")
def sources(include_regions: bool = True):
    return {"sources": load_sources(include_regions=include_regions)}


@app.get("/notices")
def notices(include_regions: bool = False):
    return collect_notices(include_regions=include_regions)


@app.get("/email/status")
def email_status():
    return email_config_status()


@app.post("/digest/preview")
def digest_preview(request: DigestRequest):
    return build_daily_digest(
        recipient=request.recipient,
        include_regions=request.include_regions,
        seed=request.seed,
    )


@app.post("/digest/email")
def digest_email(request: DigestRequest):
    digest = build_daily_digest(
        recipient=request.recipient,
        include_regions=request.include_regions,
        seed=request.seed,
    )
    if not request.send:
        return {
            "sent": False,
            "message": "send=false이므로 발송하지 않고 미리보기만 반환합니다.",
            **digest,
        }
    try:
        result = send_email(
            subject=digest["subject"],
            body=digest["body"],
            recipient=request.recipient,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "sent": True,
        "email": result,
        "subject": digest["subject"],
        "recipient": request.recipient,
    }
