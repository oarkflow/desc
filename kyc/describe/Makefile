SHELL := /usr/bin/env bash

PYTHON ?= python3
ROOT := ../..
ROOT_ABS := $(abspath $(ROOT))
VENV ?= $(ROOT_ABS)/kyc/.venv
BIN := $(VENV)/bin
PY := $(BIN)/python
PIP := $(PY) -m pip
UVICORN := $(BIN)/uvicorn

HOST ?= 127.0.0.1
PORT ?= 8010
IMAGE ?= kyc/testdata/citizenship.jpg

.PHONY: help venv install models setup run test smoke describe clean ocr-deps face-search

help:
	@echo "Targets:"
	@echo "  make setup        Create .venv, install dependencies, download models"
	@echo "  make install      Install Python dependencies into .venv"
	@echo "  make models       Download YOLOv8n, MediaPipe, YuNet, and SFace models"
	@echo "  make run          Run FastAPI service on $(HOST):$(PORT)"
	@echo "  make describe     POST IMAGE=<path> to the running service"
	@echo "  make test         Run pytest"
	@echo "  make smoke        Run local smoke test"
	@echo "  make ocr-deps     Install Tesseract English/Nepali OCR packages"
	@echo "  make face-search  Search kyc/face/tests for kyc/face/tests/test-1.webp"
	@echo "  make clean        Remove Python caches"

venv:
	@test -x "$(PY)" || $(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install -r $(ROOT_ABS)/kyc/requirements.txt

models: install
	$(PY) $(ROOT_ABS)/kyc/scripts/download_models.py

setup: install models

run: setup
	cd $(ROOT_ABS) && $(PY) -m kyc.ocr_service --serve --host $(HOST) --port $(PORT)

describe:
	curl -sS -X POST "http://$(HOST):$(PORT)/describe" -F "file=@$(IMAGE)"

test: setup
	cd $(ROOT_ABS) && $(PY) -m pytest kyc/tests/describe -vv

smoke: setup
	cd $(ROOT_ABS) && $(PY) kyc/scripts/describe_smoke_test.py

ocr-deps:
	sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-nep

face-search: setup
	cd $(ROOT_ABS) && $(PY) -m kyc.face.cli search kyc/face/tests/test-1.webp kyc/face/tests

clean:
	rm -rf .pytest_cache __pycache__ kyc/describe/__pycache__
