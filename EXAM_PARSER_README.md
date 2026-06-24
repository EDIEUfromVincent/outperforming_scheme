# 교육과정 A/B 시험지 파서

2015~2026학년도 초등학교 교육과정 A/B PDF를 페이지, 과목, 대문항,
소문항 단위로 분리합니다. 텍스트 PDF는 `pypdf`, 스캔 PDF는
OpenCV 전처리(그레이스케일 → 노이즈 제거 → CLAHE → Otsu 이진화) 후
EasyOCR을 사용합니다.

## 실행

```bash
# OCR 패키지 설치(2015년 시험지 처리에 필요)
pip install -r requirements-parser-ocr.txt

# 현재 폴더의 2015~2026 A/B 전체 파싱
python exam_parser.py . -o parsed_exams

# 특정 범위 또는 특정 유형만 파싱
python exam_parser.py . -o parsed_exams --years 2020-2026 --forms A

# OCR 없이 텍스트 레이어만 사용한 빠른 점검
python exam_parser.py . -o parsed_exams_native --ocr never
```

결과는 시험지별 JSON, 전체 대문항을 모은 `questions.csv`, 처리 상태를
모은 `summary.json`으로 저장됩니다. JSON의 `pages`에는 원문 페이지 텍스트와
추출 방식이 함께 기록되어 재검수가 가능합니다.

## 주의점

- 2015년 A/B는 텍스트 레이어가 거의 없어 OCR 패키지와 최초 실행 시 내려받는
  EasyOCR 한글 모델이 필요합니다.
- 수식, 악보, 지도, 도형 자체는 이미지 의미를 복원하지 않습니다. 주변 문장은
  추출되지만 시각 자료까지 구조화하려면 별도의 이미지/도형 인식 단계가 필요합니다.
- `summary.json`의 `warnings`와 문항 수(통상 11개)를 먼저 확인한 뒤 결과를 사용하세요.
