#!/usr/bin/env python3
"""2015↔2022 초등 성취기준을 추출하고 근거 기반 대응표를 생성한다."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from ocr_service import OCRService


CODE_RE = re.compile(r"\[(?P<code>[246](?:국|도|사|수|과|실|체|음|미|영|바|슬|즐)\d{2}-\d{2})\]")
SUBJECT_MARKERS = {
    "국어": "국어", "도덕": "도덕", "사회": "사회", "수학": "수학",
    "과학": "과학", "실과": "실과", "체육": "체육", "음악": "음악",
    "미술": "미술", "영어": "영어", "바른 생활": "통합교과",
}


@dataclass
class Standard:
    id: str
    version: str
    subject: str
    grade_band: str
    code: str
    text: str
    source: str
    page: int


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def subject_from_filename(name: str) -> str | None:
    name = nfc(name)
    if "별책15" in name:
        return "통합교과"
    for marker, subject in SUBJECT_MARKERS.items():
        if marker in name:
            return subject
    return None


def version_from_filename(name: str) -> str | None:
    name = nfc(name)
    if "2015" in name or "제2015-" in name:
        return "2015"
    if name.startswith("[별책"):
        return "2022"
    return None


def clean_ocr_codes(text: str) -> str:
    # OCR이 성취기준 코드 내부에 넣은 공백만 제거한다.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = text.replace("\uf000", " ").replace("\uf001", " ")
    return re.sub(
        r"\[\s*([246])\s*([가-힣])\s*(\d{2})\s*-\s*(\d{2})\s*\]",
        r"[\1\2\3-\4]",
        text,
    )


def extract_standards(path: Path, service: OCRService) -> list[Standard]:
    version = version_from_filename(path.name)
    subject = subject_from_filename(path.name)
    if not version or not subject:
        return []
    document = service.extract_pdf(path)
    standards = []
    seen = set()
    for page in document.pages:
        text = clean_ocr_codes(page.text)
        matches = list(CODE_RE.finditer(text))
        for index, match in enumerate(matches):
            code = match.group("code")
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else min(len(text), start + 1400)
            block = re.sub(r"\s+", " ", text[start:end]).strip()
            if code in seen or len(block) < len(code) + 8:
                continue
            seen.add(code)
            grade_band = {"2": "1-2", "4": "3-4", "6": "5-6"}[code[0]]
            standards.append(
                Standard(
                    id=f"{version}-{code}", version=version, subject=subject,
                    grade_band=grade_band, code=code, text=block,
                    source=nfc(path.name), page=page.page,
                )
            )
    return standards


def canonical(text: str) -> str:
    return re.sub(r"[^가-힣A-Za-z0-9]", "", text)


def create_mappings(standards: list[Standard]) -> list[dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    mappings = []
    groups = sorted({(s.subject, s.grade_band) for s in standards})
    for subject, grade_band in groups:
        old = [s for s in standards if s.version == "2015" and s.subject == subject and s.grade_band == grade_band]
        new = [s for s in standards if s.version == "2022" and s.subject == subject and s.grade_band == grade_band]
        new_by_code = {s.code: s for s in new}
        used_new = set()

        for source in old:
            target = new_by_code.get(source.code)
            if target:
                used_new.add(target.code)
                same = canonical(source.text) == canonical(target.text)
                mappings.append(mapping_row(source, target, "unchanged" if same else "same_code_modified", 1.0))

        old_remaining = [s for s in old if s.code not in new_by_code]
        new_remaining = [s for s in new if s.code not in used_new]
        if old_remaining and new_remaining:
            corpus = [canonical(s.text) for s in old_remaining + new_remaining]
            vectors = TfidfVectorizer(analyzer="char", ngram_range=(2, 5)).fit_transform(corpus)
            scores = cosine_similarity(vectors[: len(old_remaining)], vectors[len(old_remaining) :])
            candidates = sorted(
                (
                    (float(scores[i, j]), i, j)
                    for i in range(len(old_remaining))
                    for j in range(len(new_remaining))
                    if scores[i, j] >= 0.22
                ),
                reverse=True,
            )
            used_old_indexes, used_new_indexes = set(), set()
            for score, i, j in candidates:
                if i in used_old_indexes or j in used_new_indexes:
                    continue
                used_old_indexes.add(i); used_new_indexes.add(j)
                mappings.append(mapping_row(old_remaining[i], new_remaining[j], "semantic_match", score))
            for i, source in enumerate(old_remaining):
                if i not in used_old_indexes:
                    mappings.append(mapping_row(source, None, "removed_or_unmatched", 0.0))
            for j, target in enumerate(new_remaining):
                if j not in used_new_indexes:
                    mappings.append(mapping_row(None, target, "new_or_unmatched", 0.0))
        else:
            mappings.extend(mapping_row(source, None, "removed_or_unmatched", 0.0) for source in old_remaining)
            mappings.extend(mapping_row(None, target, "new_or_unmatched", 0.0) for target in new_remaining)
    return mappings


def mapping_row(old: Standard | None, new: Standard | None, change_type: str, score: float) -> dict:
    old_domain = re.search(r"[가-힣](\d{2})-", old.code).group(1) if old else None
    new_domain = re.search(r"[가-힣](\d{2})-", new.code).group(1) if new else None
    return {
        "mapping_id": f"MAP-{old.code if old else 'NEW'}-{new.code if new else 'REMOVED'}",
        "subject": (old or new).subject,
        "grade_band": (old or new).grade_band,
        "domain_code_2015": old_domain,
        "domain_code_2022": new_domain,
        "change_type": change_type,
        "similarity": round(score, 4),
        "code_2015": old.code if old else None,
        "text_2015": old.text if old else None,
        "source_2015": old.source if old else None,
        "page_2015": old.page if old else None,
        "code_2022": new.code if new else None,
        "text_2022": new.text if new else None,
        "source_2022": new.source if new else None,
        "page_2022": new.page if new else None,
        "review_required": change_type in {"semantic_match", "removed_or_unmatched", "new_or_unmatched"},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("ocr_cache"))
    parser.add_argument("--output", type=Path, default=Path("curriculum_mapping"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    service = OCRService(args.cache, backend="tesseract", dpi=250)

    standards = []
    seen_hashes = set()
    for path in sorted(args.directory.glob("*.pdf")):
        if subject_from_filename(path.name) is None or version_from_filename(path.name) is None:
            continue
        file_hash = service.file_id(path)
        if file_hash in seen_hashes:
            continue
        seen_hashes.add(file_hash)
        standards.extend(extract_standards(path, service))
    mappings = create_mappings(standards)

    (args.output / "standards.json").write_text(
        json.dumps([asdict(s) for s in standards], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "mappings.json").write_text(
        json.dumps(mappings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if mappings:
        with (args.output / "mappings.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(mappings[0]))
            writer.writeheader(); writer.writerows(mappings)
    summary = {
        "standards_2015": sum(s.version == "2015" for s in standards),
        "standards_2022": sum(s.version == "2022" for s in standards),
        "mappings": len(mappings),
        "review_required": sum(row["review_required"] for row in mappings),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
