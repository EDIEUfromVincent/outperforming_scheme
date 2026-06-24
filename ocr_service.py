"""교육과정·기출 PDF를 위한 페이지 단위 OCR 서비스."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from pypdf import PdfReader


@dataclass
class OCRPage:
    page: int
    text: str
    method: str
    confidence: float | None
    native_char_count: int


@dataclass
class OCRDocument:
    document_id: str
    source: str
    pages: list[OCRPage]
    warnings: list[str]

    @property
    def methods(self) -> list[str]:
        return sorted({page.method for page in self.pages})

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "source": self.source,
            "methods": self.methods,
            "warnings": self.warnings,
            "pages": [asdict(page) for page in self.pages],
        }


def normalize_text(text: str) -> str:
    # 일부 구형 한글 PDF의 잘못된 텍스트 레이어가 만든 고립 surrogate 제거
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    text = text.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def text_quality(text: str) -> float:
    """한글 문서의 OCR 필요성을 판단하는 0~1 휴리스틱 점수."""
    compact = re.sub(r"\s", "", text)
    if not compact:
        return 0.0
    useful = len(re.findall(r"[가-힣A-Za-z0-9]", compact)) / len(compact)
    length = min(1.0, len(compact) / 350)
    return round(useful * length, 4)


class OCRService:
    """텍스트 레이어를 우선하고 저품질 페이지만 OCR한다."""

    def __init__(
        self,
        cache_dir: str | Path = "ocr_cache",
        backend: str = "auto",
        dpi: int = 250,
        minimum_quality: float = 0.45,
    ):
        if backend not in {"auto", "tesseract", "easyocr", "never"}:
            raise ValueError(f"지원하지 않는 OCR backend: {backend}")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.dpi = dpi
        self.minimum_quality = minimum_quality
        self._easyocr_reader = None

    @staticmethod
    def file_id(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def extract_pdf(self, pdf_path: str | Path, force: bool = False) -> OCRDocument:
        path = Path(pdf_path)
        document_id = self.file_id(path)
        cache_signature = f"{self.backend}-{self.dpi}-{self.minimum_quality:.2f}"
        cache_path = self.cache_dir / f"{document_id}-{cache_signature}.json"
        if cache_path.exists() and not force:
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                return OCRDocument(
                    document_id=data["document_id"],
                    source=str(path),
                    pages=[OCRPage(**page) for page in data["pages"]],
                    warnings=data.get("warnings", []),
                )
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
                # 중단된 이전 작업이 남긴 불완전 캐시는 새로 생성한다.
                cache_path.unlink(missing_ok=True)

        reader = PdfReader(path)
        native = [normalize_text(page.extract_text() or "") for page in reader.pages]
        pages: list[OCRPage] = []
        warnings: list[str] = []
        low_quality = [i for i, text in enumerate(native) if text_quality(text) < self.minimum_quality]
        ocr_results: dict[int, tuple[str, str, float | None]] = {}
        if low_quality and self.backend != "never":
            try:
                ocr_results = self._ocr_selected_pages(path, low_quality)
            except RuntimeError as exc:
                warnings.append(str(exc))

        for index, text in enumerate(native):
            if index in ocr_results:
                ocr_text, method, confidence = ocr_results[index]
                if len(re.sub(r"\s", "", ocr_text)) >= len(re.sub(r"\s", "", text)):
                    pages.append(OCRPage(index + 1, ocr_text, method, confidence, len(text)))
                    continue
            pages.append(OCRPage(index + 1, text, "pypdf", None, len(text)))

        unresolved = [page.page for page in pages if text_quality(page.text) < self.minimum_quality]
        if unresolved:
            suffix = "…" if len(unresolved) > 20 else ""
            warnings.append(f"저품질 페이지 재검수 필요: {unresolved[:20]}{suffix}")
        result = OCRDocument(document_id, str(path), pages, warnings)
        cache_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def _ocr_selected_pages(
        self, pdf_path: Path, page_indexes: Sequence[int]
    ) -> dict[int, tuple[str, str, float | None]]:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("OCR PDF 렌더링에 pymupdf가 필요합니다.") from exc
        backend = self._select_backend()
        document = fitz.open(pdf_path)
        matrix = fitz.Matrix(self.dpi / 72, self.dpi / 72)
        output = {}
        try:
            for index in page_indexes:
                pixmap = document[index].get_pixmap(matrix=matrix, alpha=False)
                if backend == "tesseract":
                    output[index] = self._tesseract(pixmap.samples, pixmap.width, pixmap.height)
                else:
                    output[index] = self._easyocr(pixmap.samples, pixmap.width, pixmap.height)
        finally:
            document.close()
        return output

    def _select_backend(self) -> str:
        if self.backend == "tesseract":
            if not shutil.which("tesseract"):
                raise RuntimeError("Tesseract 실행 파일을 찾지 못했습니다.")
            return "tesseract"
        if self.backend == "easyocr":
            return "easyocr"
        if shutil.which("tesseract"):
            return "tesseract"
        try:
            import easyocr  # noqa: F401
            return "easyocr"
        except ImportError as exc:
            raise RuntimeError(
                "사용 가능한 OCR 엔진이 없습니다. Tesseract(kor) 또는 EasyOCR을 설치하세요."
            ) from exc

    @staticmethod
    def _image_array(samples: bytes, width: int, height: int):
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("OCR 전처리에 opencv-python과 numpy가 필요합니다.") from exc
        rgb = np.frombuffer(samples, dtype=np.uint8).reshape(height, width, 3)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        denoised = cv2.medianBlur(gray, 3)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
        return cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def _tesseract(self, samples: bytes, width: int, height: int) -> tuple[str, str, float | None]:
        try:
            import pytesseract
            from pytesseract import Output
        except ImportError as exc:
            raise RuntimeError("pytesseract가 필요합니다.") from exc
        image = self._image_array(samples, width, height)
        available = set(pytesseract.get_languages(config=""))
        language = "kor+eng" if "kor" in available else "eng"
        data = pytesseract.image_to_data(
            image, lang=language, config="--oem 3 --psm 3", output_type=Output.DICT
        )
        lines: dict[tuple[int, int, int], list[str]] = {}
        confidences = []
        for index, word in enumerate(data["text"]):
            word = word.strip()
            try:
                confidence = float(data["conf"][index])
            except (TypeError, ValueError):
                confidence = -1
            if not word or confidence < 25:
                continue
            key = (data["block_num"][index], data["par_num"][index], data["line_num"][index])
            lines.setdefault(key, []).append(word)
            confidences.append(confidence)
        text = normalize_text("\n".join(" ".join(words) for words in lines.values()))
        average = round(sum(confidences) / len(confidences) / 100, 4) if confidences else None
        return text, "tesseract", average

    def _easyocr(self, samples: bytes, width: int, height: int) -> tuple[str, str, float | None]:
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError("easyocr가 필요합니다.") from exc
        image = self._image_array(samples, width, height)
        if self._easyocr_reader is None:
            self._easyocr_reader = easyocr.Reader(["ko", "en"], gpu=False)
        results = self._easyocr_reader.readtext(image, detail=1, paragraph=False)
        items, confidences = [], []
        for bbox, text, confidence in results:
            if confidence < 0.25 or not text.strip():
                continue
            y = sum(point[1] for point in bbox) / len(bbox)
            x = min(point[0] for point in bbox)
            items.append((y, x, text.strip()))
            confidences.append(float(confidence))
        items.sort(key=lambda item: (round(item[0] / 12), item[1]))
        text = normalize_text("\n".join(item[2] for item in items))
        average = round(sum(confidences) / len(confidences), 4) if confidences else None
        return text, "easyocr", average
