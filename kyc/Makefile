.PHONY: setup run check test clean

VENV_PYTHON := python3.12
PYTHON := .venv/bin/python
PIP := .venv/bin/python -m pip
ROOT := $(abspath $(CURDIR)/..)
MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib
FACE_MODEL := models/face_landmarker.task
FACE_MODEL_URL := https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
PORT ?= 5555
APP_ENV := MPLCONFIGDIR="$(MPLCONFIGDIR)" KYC_DB="$(CURDIR)/kyc.db" EVIDENCE_DIR="$(CURDIR)/evidence" DOCUMENT_TYPES_CONFIG="$(CURDIR)/config/document_types.yaml"

setup: .venv/.python-version $(FACE_MODEL)
	$(PIP) install -r requirements.txt

.venv/.python-version: requirements.txt
	$(VENV_PYTHON) -m venv --clear .venv
	$(PYTHON) --version > .venv/.python-version

$(FACE_MODEL):
	mkdir -p models
	curl -L --fail -o "$(FACE_MODEL)" "$(FACE_MODEL_URL)"

run: setup
	mkdir -p "$(MPLCONFIGDIR)"
	cd "$(ROOT)" && $(APP_ENV) PORT="$(PORT)" kyc/$(PYTHON) -m kyc.index

check: setup
	mkdir -p "$(MPLCONFIGDIR)"
	cd "$(ROOT)" && $(APP_ENV) kyc/$(PYTHON) -c "from kyc.blink import APILivenessDetector; from kyc.kyc import OCRGatewayClient; detector = APILivenessDetector(); gateway = OCRGatewayClient(); print(f'import ok ({detector.backend}); OCR gateway: {gateway.base_url}')"

test: setup
	cd "$(ROOT)" && $(APP_ENV) kyc/$(PYTHON) -m unittest discover -s kyc/tests

clean:
	rm -rf .venv .cache kyc.db kyc.db-* evidence
