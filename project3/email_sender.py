"""Gmail SMTP 이메일 발송 유틸.

필요한 .env 값:
- EMAIL_ADDRESS: 발신 Gmail 주소
- EMAIL_PASSWORD: Gmail 앱 비밀번호 16자리
- EMAIL_RECIPIENT: 기본 수신자. 없으면 ohjinwoo9696@gmail.com

주의:
- 일반 Gmail 비밀번호가 아니라 Google 계정의 '앱 비밀번호'를 사용해야 한다.
- 이 모듈은 호출자가 명시적으로 send 함수를 실행할 때만 발송한다.
"""

from __future__ import annotations

import os
import smtplib
from email.mime.text import MIMEText

from dotenv import load_dotenv


load_dotenv()

DEFAULT_RECIPIENT = "ohjinwoo9696@gmail.com"


def email_config_status() -> dict:
    sender = os.getenv("EMAIL_ADDRESS", "")
    password = os.getenv("EMAIL_PASSWORD", "")
    recipient = os.getenv("EMAIL_RECIPIENT", DEFAULT_RECIPIENT)
    return {
        "sender_configured": bool(sender),
        "password_configured": bool(password),
        "recipient": recipient,
        "ready": bool(sender and password and recipient),
    }


def send_email(subject: str, body: str, recipient: str | None = None) -> dict:
    sender = os.getenv("EMAIL_ADDRESS", "")
    password = os.getenv("EMAIL_PASSWORD", "")
    to_address = recipient or os.getenv("EMAIL_RECIPIENT", DEFAULT_RECIPIENT)
    if not sender:
        raise RuntimeError("EMAIL_ADDRESS가 설정되어 있지 않습니다.")
    if not password:
        raise RuntimeError("EMAIL_PASSWORD, 즉 Gmail 앱 비밀번호가 설정되어 있지 않습니다.")
    if not to_address:
        raise RuntimeError("수신자 이메일이 설정되어 있지 않습니다.")

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_address

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
        server.login(sender, password)
        server.sendmail(sender, [to_address], msg.as_string())

    return {
        "sent": True,
        "from": sender,
        "to": to_address,
        "subject": subject,
    }
