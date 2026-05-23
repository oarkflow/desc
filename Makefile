.PHONY: setup run check clean

VENV_PYTHON := python3.12
PYTHON := .venv/bin/python
PIP := .venv/bin/python -m pip
MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib
FACE_MODEL := models/face_landmarker.task
FACE_MODEL_URL := https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
PORT ?= 5555

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
	MPLCONFIGDIR="$(MPLCONFIGDIR)" PORT="$(PORT)" $(PYTHON) index.py

check: setup
	mkdir -p "$(MPLCONFIGDIR)"
	MPLCONFIGDIR="$(MPLCONFIGDIR)" $(PYTHON) -c "from blink import APILivenessDetector; detector = APILivenessDetector(); print(f'imports ok ({detector.backend})')"

clean:
	rm -rf .venv .cache
