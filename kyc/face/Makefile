SHELL := /usr/bin/env bash

ROOT := ../..
ROOT_ABS := $(abspath $(ROOT))
VENV ?= $(ROOT_ABS)/kyc/.venv
PY := $(VENV)/bin/python
PIP := $(PY) -m pip

IMAGE ?= tests/obama1.jpg
QUERY ?= tests/test-1.webp
FOLDER ?= tests
OUT ?= /tmp/face_annotated.jpg
JSON ?= /tmp/face_result.json
DB ?= /tmp/face_db
LABEL ?= person
CROP_DIR ?= /tmp/face_crops
OVERLAY_DIR ?= /tmp/face_overlays
OVERLAY ?= boxes
POINTS_DIR ?= /tmp/face_points
VERBOSE ?=

.PHONY: help install models setup analyze detect detect-all crop search enroll recognize batch test clean

help:
	@echo "Face targets:"
	@echo "  make setup       Install root deps and download all models"
	@echo "  make models      Download all shared models"
	@echo "  make analyze     Analyze IMAGE=$(IMAGE)"
	@echo "  make detect      Detect faces with OVERLAY=$(OVERLAY)"
	@echo "  make detect-all  Detect all images in FOLDER=$(FOLDER), store points in POINTS_DIR=$(POINTS_DIR)"
	@echo "  make crop        Crop detected faces from IMAGE=$(IMAGE)"
	@echo "  make search      Search FOLDER=$(FOLDER) or image-list file for QUERY=$(QUERY)"
	@echo "  make enroll      Enroll LABEL=$(LABEL) from IMAGE=$(IMAGE) or a folder"
	@echo "  make recognize   Recognize IMAGE=$(IMAGE) using DB=$(DB)"
	@echo "  make batch       Analyze all images in FOLDER=$(FOLDER)"
	@echo "  make test        Run face-related tests"
	@echo "  make clean       Remove Python caches"

install:
	$(MAKE) -C $(ROOT_ABS)/kyc setup

models:
	$(MAKE) -C $(ROOT_ABS)/kyc models

setup:
	$(MAKE) -C $(ROOT_ABS)/kyc setup

analyze: setup
	cd $(ROOT_ABS) && $(PY) -m kyc.face.cli \
		--landmark-mode mediapipe \
		--mediapipe-model kyc/models/face_landmarker.task \
		analyze kyc/face/$(IMAGE) \
		--output $(OUT) \
		--json $(JSON) \
		--metrics

detect: setup
	cd $(ROOT_ABS) && $(PY) -m kyc.face.cli \
		--landmark-mode mediapipe \
		--mediapipe-model kyc/models/face_landmarker.task \
		detect kyc/face/$(IMAGE) \
		--overlay $(OVERLAY) \
		--output $(OUT) \
		--json $(JSON)

detect-all: setup
	cd $(ROOT_ABS) && $(PY) -m kyc.face.cli \
		--landmark-mode mediapipe \
		--mediapipe-model kyc/models/face_landmarker.task \
		detect-all kyc/face/$(FOLDER) \
		--output-dir $(POINTS_DIR) \
		--overlay $(OVERLAY) \
		--crop

crop: setup
	cd $(ROOT_ABS) && $(PY) -m kyc.face.cli \
		--landmark-mode mediapipe \
		--mediapipe-model kyc/models/face_landmarker.task \
		detect kyc/face/$(IMAGE) \
		--overlay none \
		--crop-dir $(CROP_DIR) \
		--json $(JSON)

search: setup
	cd $(ROOT_ABS) && $(PY) -m kyc.face.cli search kyc/face/$(QUERY) kyc/face/$(FOLDER) \
		--overlay-dir $(OVERLAY_DIR) \
		--crop-dir $(CROP_DIR) \
		$(VERBOSE) \
		--json $(JSON)

enroll: setup
	cd $(ROOT_ABS) && $(PY) -m kyc.face.cli enroll "$(LABEL)" kyc/face/$(IMAGE) --db $(DB)

recognize: setup
	cd $(ROOT_ABS) && $(PY) -m kyc.face.cli recognize kyc/face/$(IMAGE) --db $(DB) --json $(JSON) --output $(OUT)

batch: setup
	cd $(ROOT_ABS) && $(PY) -m kyc.face.cli batch kyc/face/$(FOLDER) --output-dir /tmp/face_batch

test: setup
	cd $(ROOT_ABS) && $(PY) -m pytest kyc/tests/face -vv

clean:
	$(MAKE) -C $(ROOT_ABS)/kyc clean
