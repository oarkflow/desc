.PHONY: setup run check test clean

VENV_PYTHON := python3.12
PYTHON := .venv/bin/python
PIP := .venv/bin/python -m pip
MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib
FACE_MODEL := models/face_landmarker.task
FACE_MODEL_URL := https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
TESSDATA_DIR := models/tessdata
ENG_TESSDATA := $(TESSDATA_DIR)/eng.traineddata
NEP_TESSDATA := $(TESSDATA_DIR)/nep.traineddata
ENG_TESSDATA_URL := https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata
NEP_TESSDATA_URL := https://github.com/tesseract-ocr/tessdata_best/raw/main/nep.traineddata
PORT ?= 5555

setup: .venv/.python-version $(FACE_MODEL) $(ENG_TESSDATA) $(NEP_TESSDATA)
	$(PIP) install -r requirements.txt

.venv/.python-version: requirements.txt
	$(VENV_PYTHON) -m venv --clear .venv
	$(PYTHON) --version > .venv/.python-version

$(FACE_MODEL):
	mkdir -p models
	curl -L --fail -o "$(FACE_MODEL)" "$(FACE_MODEL_URL)"

$(NEP_TESSDATA):
	mkdir -p "$(TESSDATA_DIR)"
	curl -L --fail -o "$(NEP_TESSDATA)" "$(NEP_TESSDATA_URL)"

$(ENG_TESSDATA):
	mkdir -p "$(TESSDATA_DIR)"
	curl -L --fail -o "$(ENG_TESSDATA)" "$(ENG_TESSDATA_URL)"

run: setup
	mkdir -p "$(MPLCONFIGDIR)"
	MPLCONFIGDIR="$(MPLCONFIGDIR)" TESSDATA_PREFIX="$(TESSDATA_DIR)" PORT="$(PORT)" $(PYTHON) index.py

check: setup
	mkdir -p "$(MPLCONFIGDIR)"
	MPLCONFIGDIR="$(MPLCONFIGDIR)" TESSDATA_PREFIX="$(TESSDATA_DIR)" $(PYTHON) -c "from blink import APILivenessDetector; from kyc import OCRService; detector = APILivenessDetector(); print(f'imports ok ({detector.backend}); OCR languages: {OCRService().available_languages()}')"

test: setup
	MPLCONFIGDIR="$(MPLCONFIGDIR)" TESSDATA_PREFIX="$(TESSDATA_DIR)" $(PYTHON) -m unittest discover -s tests
	MPLCONFIGDIR="$(MPLCONFIGDIR)" TESSDATA_PREFIX="$(TESSDATA_DIR)" $(PYTHON) scripts/print_sample_ocr.py

clean:
	rm -rf .venv .cache kyc.db kyc.db-* evidence
