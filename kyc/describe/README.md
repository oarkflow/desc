# Local Image Description Service

Self-hosted FastAPI service for lightweight image descriptions. It uses YOLOv8n for object detection, a rule-based caption engine, optional local Tesseract OCR for English/Nepali text, and heuristic tamper signals. There are no external API calls at request time.

## Setup

```bash
python3 -m pip install -r kyc/requirements.txt
python3 kyc/scripts/download_models.py
```

All model artifacts are downloaded into `kyc/models/`, which is ignored by git except for a placeholder file.

Or use Make:

```bash
make setup
```

KYC requirements include the image description runtime dependencies.

For OCR, install the Tesseract binary and language packs through your OS package manager:

```bash
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-nep
```

The service still works without OCR and returns an empty `text` field.

## Run

```bash
python -m kyc.ocr_service --serve --port 8000
```

With Make:

```bash
make run PORT=8010
```

## Response

`POST /describe` returns detected objects, generated caption, OCR text, OCR languages, tags, image dimensions, and a `tamper` object with heuristic `verdict`, `score`, and signal details.

Tamper analysis is a local heuristic check, not forensic proof. It looks at metadata, JPEG error level, local noise consistency, and edge consistency.

## Test

```bash
pytest
```

With Make:

```bash
make test
```

The tests generate JPEG, PNG, WebP, and BMP sample images in memory and verify that the upload pipeline accepts each format.

## Face Search

```bash
make face-search
```

Or from the face folder:

```bash
make -C kyc/face search QUERY=tests/test-1.webp FOLDER=tests
```
