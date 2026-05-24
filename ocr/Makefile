.PHONY: setup run run-ocr run-gateway check test clean

-include .env
export

VENV_PYTHON ?= python3.11
PYTHON := .venv/bin/python
PIP := .venv/bin/python -m pip

OCR_HOST ?= 0.0.0.0
OCR_PORT ?= 8001
OCR_WORKERS ?= 1
OCR_LOG_LEVEL ?= info
OCR_CACHE_DIR ?= $(CURDIR)/.ocr_cache
MPLCONFIGDIR ?= $(OCR_CACHE_DIR)/matplotlib
GATEWAY_PORT ?= 8000
GATEWAY_OCR_UPSTREAM ?= http://127.0.0.1:$(OCR_PORT)
GATEWAY_CONFIG_DIR ?= $(CURDIR)/config
GATEWAY_DATA_DIR ?= $(CURDIR)/data
GATEWAY_BACKUP_DIR ?= $(CURDIR)/.gateway_backups

setup: .venv/.python-version
	$(PIP) install -r requirements.txt

.venv/.python-version: requirements.txt
	$(VENV_PYTHON) -m venv --clear .venv
	$(PYTHON) --version > .venv/.python-version

run: setup
	mkdir -p "$(OCR_CACHE_DIR)" "$(MPLCONFIGDIR)" "$(GATEWAY_BACKUP_DIR)"
	@set -e; \
	trap 'kill $$ocr_pid $$gateway_pid 2>/dev/null || true' INT TERM EXIT; \
	echo "Starting OCR on http://$(OCR_HOST):$(OCR_PORT)"; \
	OCR_CACHE_DIR="$(OCR_CACHE_DIR)" MPLCONFIGDIR="$(MPLCONFIGDIR)" $(PYTHON) ocr_service.py --serve --host "$(OCR_HOST)" --port "$(OCR_PORT)" --workers "$(OCR_WORKERS)" --log-level "$(OCR_LOG_LEVEL)" & \
	ocr_pid=$$!; \
	echo "Starting gateway on http://127.0.0.1:$(GATEWAY_PORT) -> $(GATEWAY_OCR_UPSTREAM)"; \
	GATEWAY_PORT="$(GATEWAY_PORT)" GATEWAY_OCR_UPSTREAM="$(GATEWAY_OCR_UPSTREAM)" GATEWAY_CONFIG_DIR="$(GATEWAY_CONFIG_DIR)" GATEWAY_DATA_DIR="$(GATEWAY_DATA_DIR)" GATEWAY_BACKUP_DIR="$(GATEWAY_BACKUP_DIR)" go run ./cmd/gateway & \
	gateway_pid=$$!; \
	wait $$ocr_pid $$gateway_pid

run-ocr: setup
	mkdir -p "$(OCR_CACHE_DIR)" "$(MPLCONFIGDIR)"
	OCR_CACHE_DIR="$(OCR_CACHE_DIR)" MPLCONFIGDIR="$(MPLCONFIGDIR)" $(PYTHON) ocr_service.py --serve --host "$(OCR_HOST)" --port "$(OCR_PORT)" --workers "$(OCR_WORKERS)" --log-level "$(OCR_LOG_LEVEL)"

run-gateway:
	mkdir -p "$(GATEWAY_BACKUP_DIR)"
	GATEWAY_PORT="$(GATEWAY_PORT)" GATEWAY_OCR_UPSTREAM="$(GATEWAY_OCR_UPSTREAM)" GATEWAY_CONFIG_DIR="$(GATEWAY_CONFIG_DIR)" GATEWAY_DATA_DIR="$(GATEWAY_DATA_DIR)" GATEWAY_BACKUP_DIR="$(GATEWAY_BACKUP_DIR)" go run ./cmd/gateway

check: setup
	$(PYTHON) -m py_compile ocr_service.py
	go test ./cmd/gateway

test: check

clean:
	rm -rf .venv .ocr_cache .gateway_backups
