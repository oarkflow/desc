.PHONY: setup models run check test clean

VENV_PYTHON := python3.12
PYTHON := .venv/bin/python
PIP := .venv/bin/python -m pip
ROOT := $(abspath $(CURDIR)/..)
MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib
PORT ?= 5555
APP_ENV := MPLCONFIGDIR="$(MPLCONFIGDIR)" KYC_DB="$(CURDIR)/kyc.db" EVIDENCE_DIR="$(CURDIR)/evidence" DOCUMENT_TYPES_CONFIG="$(CURDIR)/config/document_types.yaml"

setup: .venv/.python-version
	$(PIP) install -r requirements.txt
	$(PYTHON) scripts/download_models.py

.venv/.python-version: requirements.txt
	$(VENV_PYTHON) -m venv --clear .venv
	$(PYTHON) --version > .venv/.python-version

models:
	$(PYTHON) scripts/download_models.py

run: setup
	mkdir -p "$(MPLCONFIGDIR)"
	cd "$(ROOT)" && $(APP_ENV) OCR_PORT="$(PORT)" kyc/$(PYTHON) -m kyc.ocr_service --serve --port "$(PORT)"

check: setup
	mkdir -p "$(MPLCONFIGDIR)"
	cd "$(ROOT)" && $(APP_ENV) kyc/$(PYTHON) -c "from kyc.blink import APILivenessDetector; from kyc.kyc import OCRGatewayClient; detector = APILivenessDetector(); gateway = OCRGatewayClient(); print(f'import ok ({detector.backend}); OCR gateway: {gateway.base_url}')"

test: setup
	cd "$(ROOT)" && $(APP_ENV) kyc/$(PYTHON) -m unittest discover -s kyc/tests

clean:
	rm -rf .venv .cache kyc.db kyc.db-* evidence
