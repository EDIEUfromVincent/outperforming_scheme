# OCR·교육과정 지식베이스 통합

## 처리 흐름

`app.py → main.py → langchain_service.py → ocr_service.py → FAISS`

1. PDF 텍스트 레이어를 `pypdf`로 먼저 읽습니다.
2. 페이지별 텍스트 품질이 낮을 때만 250 DPI 이미지로 렌더링합니다.
3. OpenCV의 median blur, CLAHE, Otsu 이진화를 적용합니다.
4. Tesseract(`kor+eng`, PSM 3)를 우선 사용하고 EasyOCR을 fallback으로 사용합니다.
5. 문서 해시별 결과를 `ocr_cache`에 저장하여 같은 OCR을 반복하지 않습니다.
6. 문서 유형·개정 연도·교과·학년군·페이지·OCR 방식을 메타데이터로 보존합니다.
7. `faiss_db/documents.json`으로 동일 문서의 중복 색인을 막습니다.

## 설치

```bash
pip install -r requirements.txt
```

Tesseract는 Python 패키지와 별도로 실행 파일 및 한국어 언어 데이터가 필요합니다.

```bash
# macOS 예시
brew install tesseract tesseract-lang

# Ubuntu/Debian 예시
sudo apt-get install tesseract-ocr tesseract-ocr-kor
```

EasyOCR을 fallback으로 사용하려면 다음을 추가로 설치합니다.

```bash
pip install easyocr
```

환경변수 `OCR_BACKEND`는 `auto`, `tesseract`, `easyocr`, `never` 중 하나입니다.
기본값은 `auto`입니다.

벡터 검색은 기본적으로 로컬 문자 n-gram 임베딩을 사용합니다. OpenAI API 키는
검색된 근거를 비교·설명하고 수업 설계 답변을 생성할 때만 사용합니다.

## 사용

FastAPI와 Streamlit을 실행한 뒤 사이드바의 **교육과정 자료 일괄 색인** 버튼을
누르면 `parsed_exams_native/교육과정`의 PDF를 처리합니다. 이미 색인한 동일
파일은 건너뜁니다.

개별 PDF 업로드도 같은 OCR·메타데이터 파이프라인을 사용합니다. 화면에는
사용된 추출 방식과 재검수가 필요한 페이지가 표시됩니다.

## 보안

Clova OCR 실습 노트북의 URL이나 secret key는 서비스 코드로 가져오지 않습니다.
외부 OCR API를 도입할 경우 반드시 환경변수/비밀 저장소를 사용하고, 문서를
외부 서비스로 전송한다는 점을 별도로 검토해야 합니다.
