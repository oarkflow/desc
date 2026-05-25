SHELL := /usr/bin/env bash

PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
PY := $(BIN)/python
PIP := $(PY) -m pip
UVICORN := $(BIN)/uvicorn

HOST ?= 127.0.0.1
PORT ?= 8010
IMAGE ?= testdata/Airbrush-OBJECT-REMOVER-1774003408083.jpg

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
	@echo "  make face-search  Search face/tests for face/tests/test-1.webp"
	@echo "  make clean        Remove Python caches"

venv:
	@test -x "$(PY)" || $(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install -r requirements.txt

models: install
	$(PY) scripts/download_models.py

setup: install models

run: setup
	$(UVICORN) app.main:app --host $(HOST) --port $(PORT)

describe:
	curl -sS -X POST "http://$(HOST):$(PORT)/describe" -F "file=@$(IMAGE)"

test: setup
	$(PY) -m pytest -vv

smoke: setup
	$(PY) scripts/smoke_test.py

ocr-deps:
	sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-nep

face-search: setup
	$(PY) -m face.cli search face/tests/test-1.webp face/tests

clean:
	rm -rf .pytest_cache app/__pycache__ face/__pycache__ tests/__pycache__ scripts/__pycache__
