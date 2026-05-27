FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OCR_CACHE_DIR=/app/.ocr_cache \
    PADDLE_PDX_CACHE_HOME=/app/.ocr_cache/paddlex \
    MPLCONFIGDIR=/app/.ocr_cache/matplotlib \
    TESSDATA_PREFIX=/app/.ocr_cache/tessdata \
    OCR_HOST=0.0.0.0 \
    OCR_PORT=8000 \
    OCR_WORKERS=1 \
    OCR_LOG_LEVEL=info \
    OCR_DEVICE=cpu \
    OCR_GPU_ID=0 \
    OCR_USE_GPU=false \
    OCR_TAMPER_ENABLED=true \
    OCR_TAMPER_MODE=standard \
    OCR_MAX_PDF_PAGES=5 \
    OCR_TAMPER_REVIEW_THRESHOLD=0.35 \
    OCR_TAMPER_REJECT_THRESHOLD=0.70

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
        libjpeg62-turbo \
        libpng16-16 \
        libtiff6 \
        tesseract-ocr \
        tesseract-ocr-eng \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY ocr_service.py .
COPY ocr ./ocr
COPY describe ./describe
COPY face ./face
COPY config ./config
COPY data ./data
COPY scripts ./scripts
COPY docker-entrypoint.sh /usr/local/bin/ocr-service

RUN chmod +x /usr/local/bin/ocr-service \
    && mkdir -p /app/.ocr_cache \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${OCR_PORT}/healthz" || exit 1

ENTRYPOINT ["ocr-service"]
CMD ["--serve"]
