#!/usr/bin/env python3
"""초등임용 일일 브리핑 메일 CLI.

기본은 미리보기만 출력한다. 실제 발송은 --send 옵션이 있어야 한다.
"""

from __future__ import annotations

import argparse

from daily_digest import build_daily_digest
from email_sender import send_email


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipient", default="ohjinwoo9696@gmail.com")
    parser.add_argument("--include-regions", action="store_true")
    parser.add_argument("--seed", default=None)
    parser.add_argument("--send", action="store_true")
    args = parser.parse_args()

    digest = build_daily_digest(
        recipient=args.recipient,
        include_regions=args.include_regions,
        seed=args.seed,
    )
    if not args.send:
        print(digest["subject"])
        print(digest["body"])
        print("\n[미리보기] 실제 발송하려면 --send 옵션을 추가하세요.")
        return 0

    result = send_email(
        subject=digest["subject"],
        body=digest["body"],
        recipient=args.recipient,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
