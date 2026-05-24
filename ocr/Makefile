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
LOCAL_GATEWAY_OCR_UPSTREAM := http://127.0.0.1:$(OCR_PORT)
GATEWAY_MAX_ACTIVE ?= 1
GATEWAY_MAX_QUEUE ?= 8
GATEWAY_UPSTREAM_TIMEOUT_SECONDS ?= 120
GATEWAY_DEFAULT_ACCURACY_MODE ?= accurate
GATEWAY_DEFAULT_RETRY ?= true
GATEWAY_CONFIG_DIR ?= $(CURDIR)/config
GATEWAY_DATA_DIR ?= $(CURDIR)/data
GATEWAY_BACKUP_DIR ?= $(CURDIR)/.gateway_backups
LOG_DIR ?= $(CURDIR)/.logs
SETUP_LOG ?= $(LOG_DIR)/setup.log
OCR_RUN_LOG ?= $(LOG_DIR)/ocr.log
GATEWAY_RUN_LOG ?= $(LOG_DIR)/gateway.log

setup: .venv/.python-version
	@mkdir -p "$(LOG_DIR)"
	@echo "Installing Python dependencies (logs: $(SETUP_LOG))"
	@$(PIP) install -r requirements.txt >"$(SETUP_LOG)" 2>&1 || { cat "$(SETUP_LOG)"; exit 1; }

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
	echo "Waiting for OCR health at $(LOCAL_GATEWAY_OCR_UPSTREAM)/healthz"; \
	for i in $$(seq 1 90); do \
		if curl -fsS "$(LOCAL_GATEWAY_OCR_UPSTREAM)/healthz" >/dev/null 2>&1; then break; fi; \
		if ! kill -0 $$ocr_pid 2>/dev/null; then echo "OCR exited before becoming healthy"; exit 1; fi; \
		sleep 1; \
		if [ "$$i" = "90" ]; then echo "Timed out waiting for OCR health"; exit 1; fi; \
	done; \
	echo "Starting gateway on http://127.0.0.1:$(GATEWAY_PORT) -> $(LOCAL_GATEWAY_OCR_UPSTREAM)"; \
	GATEWAY_PORT="$(GATEWAY_PORT)" GATEWAY_OCR_UPSTREAM="$(LOCAL_GATEWAY_OCR_UPSTREAM)" GATEWAY_MAX_ACTIVE="$(GATEWAY_MAX_ACTIVE)" GATEWAY_MAX_QUEUE="$(GATEWAY_MAX_QUEUE)" GATEWAY_UPSTREAM_TIMEOUT_SECONDS="$(GATEWAY_UPSTREAM_TIMEOUT_SECONDS)" GATEWAY_DEFAULT_ACCURACY_MODE="$(GATEWAY_DEFAULT_ACCURACY_MODE)" GATEWAY_DEFAULT_RETRY="$(GATEWAY_DEFAULT_RETRY)" GATEWAY_CONFIG_DIR="$(GATEWAY_CONFIG_DIR)" GATEWAY_DATA_DIR="$(GATEWAY_DATA_DIR)" GATEWAY_BACKUP_DIR="$(GATEWAY_BACKUP_DIR)" go run ./cmd/gateway & \
	gateway_pid=$$!; \
	wait $$ocr_pid $$gateway_pid

run-ocr: setup
	@mkdir -p "$(OCR_CACHE_DIR)" "$(MPLCONFIGDIR)" "$(LOG_DIR)"
	@echo "Starting OCR on http://$(OCR_HOST):$(OCR_PORT) (logs: $(OCR_RUN_LOG))"
	@OCR_CACHE_DIR="$(OCR_CACHE_DIR)" MPLCONFIGDIR="$(MPLCONFIGDIR)" $(PYTHON) ocr_service.py --serve --host "$(OCR_HOST)" --port "$(OCR_PORT)" --workers "$(OCR_WORKERS)" --log-level "$(OCR_LOG_LEVEL)" >"$(OCR_RUN_LOG)" 2>&1

run-gateway:
	@mkdir -p "$(GATEWAY_BACKUP_DIR)" "$(LOG_DIR)"
	@echo "Starting gateway on http://127.0.0.1:$(GATEWAY_PORT) -> $(LOCAL_GATEWAY_OCR_UPSTREAM) (logs: $(GATEWAY_RUN_LOG))"
	@GATEWAY_PORT="$(GATEWAY_PORT)" GATEWAY_OCR_UPSTREAM="$(LOCAL_GATEWAY_OCR_UPSTREAM)" GATEWAY_MAX_ACTIVE="$(GATEWAY_MAX_ACTIVE)" GATEWAY_MAX_QUEUE="$(GATEWAY_MAX_QUEUE)" GATEWAY_UPSTREAM_TIMEOUT_SECONDS="$(GATEWAY_UPSTREAM_TIMEOUT_SECONDS)" GATEWAY_DEFAULT_ACCURACY_MODE="$(GATEWAY_DEFAULT_ACCURACY_MODE)" GATEWAY_DEFAULT_RETRY="$(GATEWAY_DEFAULT_RETRY)" GATEWAY_CONFIG_DIR="$(GATEWAY_CONFIG_DIR)" GATEWAY_DATA_DIR="$(GATEWAY_DATA_DIR)" GATEWAY_BACKUP_DIR="$(GATEWAY_BACKUP_DIR)" go run ./cmd/gateway >"$(GATEWAY_RUN_LOG)" 2>&1

check: setup
	$(PYTHON) -m py_compile ocr_service.py
	go test ./cmd/gateway

test: check
	$(PYTHON) -m unittest discover -s tests

clean:
	rm -rf .venv .ocr_cache .gateway_backups
