from __future__ import annotations

import io
import json
import mimetypes
import os
import re
import resource
import shutil
import subprocess
import sys
import time
import uuid
import logging
from contextlib import redirect_stdout
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

DEFAULT_CACHE_DIR = Path(os.getenv("OCR_CACHE_DIR", "./.ocr_cache")).resolve()
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(DEFAULT_CACHE_DIR / "paddlex"))
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("TESSDATA_PREFIX", str(DEFAULT_CACHE_DIR / "tessdata"))

import cv2
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from paddleocr import PaddleOCR
import yaml


# ----------------------------
# Config
# ----------------------------

class Settings:
    APP_NAME = "Production OCR Service"
    MAX_FILE_MB = int(os.getenv("OCR_MAX_FILE_MB", "15"))
    MIN_CONFIDENCE = float(os.getenv("OCR_MIN_CONFIDENCE", "0.45"))
    INTERNAL_MIN_CONFIDENCE = float(os.getenv("OCR_INTERNAL_MIN_CONFIDENCE", "0.30"))
    OCR_LANG = os.getenv("OCR_LANG", "ne")
    OCR_VERSION = os.getenv("OCR_VERSION", "PP-OCRv5")
    TEXT_DETECTION_MODEL = os.getenv("OCR_TEXT_DETECTION_MODEL", "PP-OCRv5_mobile_det")
    TEXT_RECOGNITION_MODEL = os.getenv(
        "OCR_TEXT_RECOGNITION_MODEL",
        "devanagari_PP-OCRv5_mobile_rec",
    )
    DEVANAGARI_TEXT_RECOGNITION_MODEL = os.getenv(
        "OCR_DEVANAGARI_TEXT_RECOGNITION_MODEL",
        "devanagari_PP-OCRv5_mobile_rec",
    )
    USE_GPU = os.getenv("OCR_USE_GPU", "false").lower() == "true"
    GPU_ID = os.getenv("OCR_GPU_ID", "0")
    OCR_DEVICE = os.getenv("OCR_DEVICE", "").strip()
    USE_TEXTLINE_ORIENTATION = (
        os.getenv("OCR_USE_TEXTLINE_ORIENTATION", "false").lower() == "true"
    )
    DET_LIMIT_SIDE_LEN = int(os.getenv("OCR_DET_LIMIT_SIDE_LEN", "1280"))
    DET_LIMIT_TYPE = os.getenv("OCR_DET_LIMIT_TYPE", "max")
    DET_BOX_THRESH = float(os.getenv("OCR_DET_BOX_THRESH", "0.45"))
    DET_UNCLIP_RATIO = float(os.getenv("OCR_DET_UNCLIP_RATIO", "1.8"))
    SAVE_DEBUG = os.getenv("OCR_SAVE_DEBUG", "false").lower() == "true"
    DEBUG_DIR = Path(os.getenv("OCR_DEBUG_DIR", "./debug_ocr"))
    CLEAN_BACKGROUND = os.getenv("OCR_CLEAN_BACKGROUND", "false").lower() == "true"
    USE_TESSERACT_REPAIR = (
        os.getenv("OCR_USE_TESSERACT_REPAIR", "true").lower() == "true"
    )
    TESSERACT_LANG = os.getenv("OCR_TESSERACT_LANG", "nep")
    LOCATION_GAZETTEER_PATH = Path(
        os.getenv("OCR_LOCATION_GAZETTEER_PATH", "./data/nepal_admin_areas.yaml")
    )
    DOCUMENT_PROFILES_PATH = Path(
        os.getenv("OCR_DOCUMENT_PROFILES_PATH", "./config/document_profiles.yaml")
    )
    CACHE_DIR = DEFAULT_CACHE_DIR
    SERVER_HOST = os.getenv("OCR_HOST", "0.0.0.0")
    SERVER_PORT = int(os.getenv("OCR_PORT", "8000"))
    SERVER_WORKERS = int(os.getenv("OCR_WORKERS", "1"))
    SERVER_LOG_LEVEL = os.getenv("OCR_LOG_LEVEL", "info")
    SERVER_KEEP_ALIVE = int(os.getenv("OCR_KEEP_ALIVE", "15"))


settings = Settings()


def ocr_device() -> str:
    if settings.OCR_DEVICE:
        if settings.OCR_DEVICE.lower() == "gpu":
            return f"gpu:{settings.GPU_ID}"
        return settings.OCR_DEVICE
    if settings.USE_GPU:
        return f"gpu:{settings.GPU_ID}"
    return "cpu"


def ocr_uses_gpu() -> bool:
    return ocr_device().lower().startswith("gpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("ocr")


def resource_snapshot() -> dict[str, float]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "wall_time_seconds": time.perf_counter(),
        "process_cpu_seconds": time.process_time(),
        "user_cpu_seconds": float(usage.ru_utime),
        "system_cpu_seconds": float(usage.ru_stime),
        "max_rss_kb": normalize_max_rss_kb(float(usage.ru_maxrss)),
    }


def normalize_max_rss_kb(raw_max_rss: float) -> float:
    if sys.platform == "darwin":
        return round(raw_max_rss / 1024, 2)
    return round(raw_max_rss, 2)


def resource_delta(started: dict[str, float]) -> dict[str, Any]:
    ended = resource_snapshot()
    wall_seconds = max(ended["wall_time_seconds"] - started["wall_time_seconds"], 0.0)
    process_cpu_seconds = max(
        ended["process_cpu_seconds"] - started["process_cpu_seconds"],
        0.0,
    )
    user_cpu_seconds = max(
        ended["user_cpu_seconds"] - started["user_cpu_seconds"],
        0.0,
    )
    system_cpu_seconds = max(
        ended["system_cpu_seconds"] - started["system_cpu_seconds"],
        0.0,
    )

    return {
        "wall_ms": int(wall_seconds * 1000),
        "cpu_ms": int(process_cpu_seconds * 1000),
        "user_cpu_ms": int(user_cpu_seconds * 1000),
        "system_cpu_ms": int(system_cpu_seconds * 1000),
        "cpu_hours": round(process_cpu_seconds / 3600, 8),
        "cpu_utilization_ratio": round(process_cpu_seconds / wall_seconds, 4)
        if wall_seconds > 0
        else 0.0,
        "max_rss_mb": round(ended["max_rss_kb"] / 1024, 2),
        "max_rss_delta_mb": round(
            max(ended["max_rss_kb"] - started["max_rss_kb"], 0.0) / 1024,
            2,
        ),
    }


# ----------------------------
# OCR Engine Singleton
# ----------------------------

class OCREngine:
    _instances: dict[tuple[str, str, str, str, str], "OCREngine"] = {}

    def __init__(
        self,
        lang: str,
        text_detection_model: Optional[str] = None,
        text_recognition_model: Optional[str] = None,
        ocr_version: Optional[str] = None,
    ) -> None:
        self.lang = lang
        self.text_detection_model = text_detection_model or ""
        self.text_recognition_model = text_recognition_model or ""
        self.ocr_version = ocr_version or settings.OCR_VERSION
        self.device = ocr_device()
        logger.info(
            "Loading PaddleOCR model for lang=%s det=%s rec=%s device=%s...",
            lang,
            self.text_detection_model or "auto",
            self.text_recognition_model or "auto",
            self.device,
        )
        ocr_kwargs: dict[str, Any] = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": settings.USE_TEXTLINE_ORIENTATION,
            "device": self.device,
            "text_rec_score_thresh": min(
                settings.INTERNAL_MIN_CONFIDENCE,
                settings.MIN_CONFIDENCE,
            ),
            "text_det_limit_side_len": settings.DET_LIMIT_SIDE_LEN,
            "text_det_limit_type": settings.DET_LIMIT_TYPE,
            "text_det_box_thresh": settings.DET_BOX_THRESH,
            "text_det_unclip_ratio": settings.DET_UNCLIP_RATIO,
        }

        if self.text_detection_model:
            ocr_kwargs["text_detection_model_name"] = self.text_detection_model
        elif lang == settings.OCR_LANG and settings.TEXT_DETECTION_MODEL:
            ocr_kwargs["text_detection_model_name"] = settings.TEXT_DETECTION_MODEL

        if self.text_recognition_model:
            ocr_kwargs["text_recognition_model_name"] = self.text_recognition_model
        elif lang == settings.OCR_LANG and settings.TEXT_RECOGNITION_MODEL:
            ocr_kwargs["text_recognition_model_name"] = settings.TEXT_RECOGNITION_MODEL

        if (
            "text_detection_model_name" not in ocr_kwargs
            and "text_recognition_model_name" not in ocr_kwargs
        ):
            ocr_kwargs["lang"] = lang
            ocr_kwargs["ocr_version"] = self.ocr_version

        self.ocr = PaddleOCR(**ocr_kwargs)
        logger.info("PaddleOCR model loaded for lang=%s device=%s", lang, self.device)

    @classmethod
    def instance(
        cls,
        lang: Optional[str] = None,
        text_detection_model: Optional[str] = None,
        text_recognition_model: Optional[str] = None,
        ocr_version: Optional[str] = None,
    ) -> "OCREngine":
        resolved_lang = (lang or settings.OCR_LANG).strip()
        if not resolved_lang:
            resolved_lang = "ne"

        key = (
            resolved_lang,
            text_detection_model or "",
            text_recognition_model or "",
            ocr_version or settings.OCR_VERSION,
            ocr_device(),
        )
        if key not in cls._instances:
            cls._instances[key] = OCREngine(
                resolved_lang,
                text_detection_model=text_detection_model,
                text_recognition_model=text_recognition_model,
                ocr_version=ocr_version,
            )
        return cls._instances[key]

    def read(
        self,
        image: np.ndarray,
        min_confidence: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        if hasattr(self.ocr, "predict"):
            return self._read_v3(image, min_confidence)

        return self._read_legacy(image, min_confidence)

    def _read_v3(
        self,
        image: np.ndarray,
        min_confidence: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        result = self.ocr.predict(input=image)
        threshold = settings.MIN_CONFIDENCE if min_confidence is None else min_confidence

        output: list[dict[str, Any]] = []

        if not result:
            return output

        for page in result:
            page_data = self._page_result_to_dict(page)
            texts = page_data.get("rec_texts") or []
            scores = page_data.get("rec_scores") or []
            boxes = page_data.get("rec_polys")
            if boxes is None:
                boxes = page_data.get("rec_boxes")
            if boxes is None:
                boxes = []

            for index, text in enumerate(texts):
                confidence = float(scores[index]) if index < len(scores) else 0.0

                if confidence < threshold:
                    continue

                output.append(
                    {
                        "text": str(text).strip(),
                        "confidence": round(confidence, 4),
                        "box": self._to_jsonable_box(
                            boxes[index] if index < len(boxes) else []
                        ),
                    }
                )

        return output

    def _read_legacy(
        self,
        image: np.ndarray,
        min_confidence: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        result = self.ocr.ocr(image, cls=True)
        threshold = settings.MIN_CONFIDENCE if min_confidence is None else min_confidence

        output: list[dict[str, Any]] = []

        if not result:
            return output

        for page in result:
            if not page:
                continue
            for line in page:
                box = line[0]
                text = line[1][0]
                confidence = float(line[1][1])

                if confidence < threshold:
                    continue

                output.append(
                    {
                        "text": text.strip(),
                        "confidence": round(confidence, 4),
                        "box": self._to_jsonable_box(box),
                    }
                )

        return output

    @staticmethod
    def _page_result_to_dict(page: Any) -> dict[str, Any]:
        page_json = getattr(page, "json", page)
        if callable(page_json):
            page_json = page_json()

        if not isinstance(page_json, dict):
            return {}

        res = page_json.get("res", page_json)
        return res if isinstance(res, dict) else {}

    @staticmethod
    def _to_jsonable_box(box: Any) -> Any:
        if hasattr(box, "tolist"):
            return box.tolist()
        return box


# ----------------------------
# Response Models
# ----------------------------

class OCRItem(BaseModel):
    text: str
    confidence: float
    box: list[Any]
    source_pass: str = "default"


class OCRField(BaseModel):
    value: str
    confidence: float
    source_text: str
    raw_value: str = ""
    normalized_value: str = ""
    requires_review: bool = False
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class OCRResponse(BaseModel):
    request_id: str
    filename: str
    mime_type: str
    file_size_bytes: int
    width: int
    height: int
    processing_ms: int
    document_type: str
    document_type_confidence: float
    full_text: str
    values: dict[str, str]
    fields: dict[str, OCRField]
    items: list[OCRItem]
    objects: list[dict[str, Any]] = Field(default_factory=list)
    object_summary: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)


# ----------------------------
# Object Detection
# ----------------------------

def empty_object_summary() -> dict[str, Any]:
    return {
        "has_id_card": False,
        "id_card_confidence": 0.0,
        "face_count": 0,
        "text_region_count": 0,
    }


def detection_box(
    x: float,
    y: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    x = max(0.0, min(float(x), float(image_width)))
    y = max(0.0, min(float(y), float(image_height)))
    width = max(0.0, min(float(width), float(image_width) - x))
    height = max(0.0, min(float(height), float(image_height) - y))
    return {
        "pixel": {
            "x": int(round(x)),
            "y": int(round(y)),
            "width": int(round(width)),
            "height": int(round(height)),
        },
        "normalized": {
            "x": round(x / max(image_width, 1), 6),
            "y": round(y / max(image_height, 1), 6),
            "width": round(width / max(image_width, 1), 6),
            "height": round(height / max(image_height, 1), 6),
        },
    }


def summarize_objects(objects: list[dict[str, Any]]) -> dict[str, Any]:
    id_card_confidence = max(
        (float(item.get("confidence") or 0.0) for item in objects if item.get("label") == "id_card"),
        default=0.0,
    )
    return {
        "has_id_card": id_card_confidence > 0,
        "id_card_confidence": round(id_card_confidence, 4),
        "face_count": sum(1 for item in objects if item.get("label") == "face"),
        "text_region_count": sum(1 for item in objects if item.get("label") == "text_region"),
    }


class ObjectDetectionProvider:
    provider_name = "base"

    def detect(self, image: np.ndarray, ocr_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raise NotImplementedError


class OpenCVObjectDetectionProvider(ObjectDetectionProvider):
    provider_name = "opencv"

    def __init__(self) -> None:
        cascade_dir = Path(cv2.data.haarcascades)
        self.face_cascade = cv2.CascadeClassifier(
            str(cascade_dir / "haarcascade_frontalface_default.xml")
        )

    def detect(self, image: np.ndarray, ocr_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        objects: list[dict[str, Any]] = []
        objects.extend(self.detect_id_cards(image))
        objects.extend(self.detect_faces(image))
        objects.extend(self.text_region_detections(image, ocr_items))
        return objects

    def detect_id_cards(self, image: np.ndarray) -> list[dict[str, Any]]:
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[dict[str, Any]] = []
        image_area = float(width * height)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < image_area * 0.05:
                continue

            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            x, y, card_w, card_h = cv2.boundingRect(polygon)
            if card_w <= 0 or card_h <= 0:
                continue

            aspect = card_w / float(card_h)
            normalized_aspect = aspect if aspect >= 1 else 1 / aspect
            if normalized_aspect < 1.2 or normalized_aspect > 2.4:
                continue

            fill_ratio = area / float(card_w * card_h)
            if fill_ratio < 0.55:
                continue

            area_ratio = min(area / image_area, 1.0)
            vertex_bonus = 0.12 if len(polygon) == 4 else 0.0
            confidence = min(0.99, 0.45 + area_ratio + vertex_bonus + min(fill_ratio, 1.0) * 0.2)
            points = polygon.reshape(-1, 2).tolist()
            candidates.append(
                {
                    "label": "id_card",
                    "confidence": round(confidence, 4),
                    "box": detection_box(x, y, card_w, card_h, width, height),
                    "polygon": [
                        {
                            "x": int(point[0]),
                            "y": int(point[1]),
                            "normalized_x": round(float(point[0]) / max(width, 1), 6),
                            "normalized_y": round(float(point[1]) / max(height, 1), 6),
                        }
                        for point in points
                    ],
                    "source": self.provider_name,
                }
            )

        candidates.sort(key=lambda item: item["confidence"], reverse=True)
        return candidates[:1]

    def detect_faces(self, image: np.ndarray) -> list[dict[str, Any]]:
        height, width = image.shape[:2]
        if self.face_cascade.empty() or height == 0 or width == 0:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        detections = []
        for x, y, face_w, face_h in faces:
            detections.append(
                {
                    "label": "face",
                    "confidence": 0.75,
                    "box": detection_box(x, y, face_w, face_h, width, height),
                    "source": self.provider_name,
                }
            )
        return detections

    def text_region_detections(
        self,
        image: np.ndarray,
        ocr_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        height, width = image.shape[:2]
        detections = []
        for item in ocr_items:
            bounds = box_bounds(item.get("box") or [])
            if not bounds:
                continue
            x1, y1, x2, y2 = bounds
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                {
                    "label": "text_region",
                    "confidence": round(float(item.get("confidence") or 0.0), 4),
                    "box": detection_box(x1, y1, x2 - x1, y2 - y1, width, height),
                    "source": "ocr",
                }
            )
        return detections


class ObjectDetectionService:
    def __init__(self, provider_name: Optional[str] = None) -> None:
        self.provider_name = (provider_name or os.getenv("OBJECT_DETECTION_PROVIDER", "opencv")).strip().lower()
        self.provider = self.create_provider(self.provider_name)

    def create_provider(self, provider_name: str) -> ObjectDetectionProvider:
        if provider_name in {"", "opencv"}:
            return OpenCVObjectDetectionProvider()
        if provider_name == "yolo":
            raise ValueError("YOLO object detection provider is not installed in this build")
        raise ValueError(f"Unsupported object detection provider: {provider_name}")

    def detect(self, image: np.ndarray, ocr_items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        objects = self.provider.detect(image, ocr_items)
        return objects, summarize_objects(objects)


# ----------------------------
# Document Profile Config
# ----------------------------

class PreprocessingConfig(BaseModel):
    upscale: bool = True
    denoise: bool = False
    threshold: bool = False
    crop_border: bool = True
    enhance: bool = True
    clean_background: bool = False
    max_image_side: Optional[int] = None


class OCRPassConfig(BaseModel):
    name: str = "default"
    mode: str = "default"
    lang: str = "ne"
    script: Optional[str] = None
    source_kind: str = "printed"
    text_detection_model: Optional[str] = None
    text_recognition_model: Optional[str] = None
    ocr_version: Optional[str] = None
    min_confidence: Optional[float] = None
    retry_fields: list[str] = Field(default_factory=list)
    run_if_missing_fields: list[str] = Field(default_factory=list)
    run_if_below_confidence: Optional[float] = None
    timeout_seconds: Optional[float] = None
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)


class OCRConfig(BaseModel):
    passes: list[OCRPassConfig] = Field(
        default_factory=lambda: [OCRPassConfig()]
    )
    timeout_seconds: Optional[float] = None
    max_passes: Optional[int] = None
    max_image_side: Optional[int] = None
    retry_padding_px: int = 24


class DetectCue(BaseModel):
    text: str


class DetectConfig(BaseModel):
    cues: list[DetectCue] = Field(default_factory=list)
    min_score: int = 1


class ValidatorConfig(BaseModel):
    type: str
    pattern: Optional[str] = None


class ProfileFieldConfig(BaseModel):
    labels: list[str] = Field(default_factory=list)
    strategies: list[str] = Field(default_factory=list)
    validators: list[ValidatorConfig] = Field(default_factory=list)
    normalizers: list[str] = Field(default_factory=list)
    components: list[str] = Field(default_factory=list)
    regex: Optional[str] = None
    gazetteer: Optional[str] = None
    review_threshold: Optional[float] = None
    anchor_region: Optional[list[float]] = None
    source_passes: list[str] = Field(default_factory=list)
    source_kinds: list[str] = Field(default_factory=list)
    review_source_kinds: list[str] = Field(default_factory=list)
    retry_region: Optional[list[float]] = None
    same_as_field: Optional[str] = None
    consistency_field: Optional[str] = None
    gazetteer_hint_fields: list[str] = Field(default_factory=list)
    template: Optional[str] = None


class ExtractionRules(BaseModel):
    digit_mappings: dict[str, str] = Field(default_factory=dict)
    clean_separator_pattern: str = r"[.:|]+"
    normalize_separator_pattern: str = r"[\s.:|,/\\\-–—]+"
    trim_leading_pattern: str = r"^[\-–—,;:]+"
    trim_trailing_pattern: str = r"[\-–—,;]+$"
    token_strip_chars: str = ".,;:-_ "
    label_only_values: list[str] = Field(default_factory=list)
    label_fragments: list[str] = Field(default_factory=list)
    strip_prefix_pattern: str = ""
    split_before_patterns: dict[str, list[str]] = Field(default_factory=dict)
    compact_remove_tokens: list[str] = Field(default_factory=list)
    insignificant_tokens: list[str] = Field(default_factory=list)
    date_component_pattern: str = ""
    digit_shape_replacements: dict[str, str] = Field(default_factory=dict)
    digit_shape_replacements_by_role: dict[str, dict[str, str]] = Field(default_factory=dict)
    digit_shape_token_patterns_by_role: dict[str, str] = Field(default_factory=dict)
    citizenship_label_fragments: list[str] = Field(default_factory=list)
    citizenship_digit_confusions: dict[str, list[str]] = Field(default_factory=dict)
    infer_patterns: dict[str, str] = Field(default_factory=dict)


class DocumentProfile(BaseModel):
    detect: DetectConfig = Field(default_factory=DetectConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    fields: dict[str, ProfileFieldConfig] = Field(default_factory=dict)


class DocumentProfilesConfig(BaseModel):
    extraction_rules: ExtractionRules = Field(default_factory=ExtractionRules)
    document_type_dir: Optional[str] = None
    document_type_files: list[str] = Field(default_factory=list)
    document_types: dict[str, DocumentProfile] = Field(default_factory=dict)


_profiles_cache: Optional[DocumentProfilesConfig] = None


def load_document_profiles() -> DocumentProfilesConfig:
    global _profiles_cache
    if _profiles_cache is not None:
        return _profiles_cache

    path = settings.DOCUMENT_PROFILES_PATH
    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        logger.warning("Document profiles not found: %s", path)
        _profiles_cache = DocumentProfilesConfig()
        return _profiles_cache

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        logger.warning("Document profiles config must be a mapping: %s", path)
        _profiles_cache = DocumentProfilesConfig()
        return _profiles_cache

    config = DocumentProfilesConfig.model_validate(data)
    config.document_types.update(load_external_document_types(data, path.parent))
    _profiles_cache = config
    return _profiles_cache


def load_external_document_types(
    root_data: dict[str, Any],
    base_dir: Path,
) -> dict[str, DocumentProfile]:
    document_types: dict[str, DocumentProfile] = {}
    candidate_paths: list[Path] = []

    document_type_dir = root_data.get("document_type_dir")
    if isinstance(document_type_dir, str) and document_type_dir.strip():
        directory = resolve_config_path(document_type_dir, base_dir)
        if directory.exists():
            candidate_paths.extend(sorted(directory.glob("*.yaml")))
            candidate_paths.extend(sorted(directory.glob("*.yml")))
        else:
            logger.warning("Document type config directory not found: %s", directory)

    document_type_files = root_data.get("document_type_files") or []
    if isinstance(document_type_files, list):
        for file_path in document_type_files:
            if isinstance(file_path, str) and file_path.strip():
                candidate_paths.append(resolve_config_path(file_path, base_dir))

    seen_paths: set[Path] = set()
    for candidate_path in candidate_paths:
        resolved_path = candidate_path.resolve()
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        document_types.update(load_document_type_file(resolved_path))

    return document_types


def load_document_type_file(path: Path) -> dict[str, DocumentProfile]:
    if not path.exists():
        logger.warning("Document type config file not found: %s", path)
        return {}

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        logger.warning("Document type config must be a mapping: %s", path)
        return {}

    if isinstance(data.get("document_types"), dict):
        return {
            str(document_type): DocumentProfile.model_validate(profile_data or {})
            for document_type, profile_data in data["document_types"].items()
        }

    document_type = data.get("document_type") or data.get("id")
    if isinstance(document_type, str) and document_type.strip():
        profile_data = {
            key: value
            for key, value in data.items()
            if key not in {"document_type", "id"}
        }
        return {document_type.strip(): DocumentProfile.model_validate(profile_data)}

    logger.warning("Document type config has no document_type/id: %s", path)
    return {}


def resolve_config_path(path: str, base_dir: Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return base_dir / resolved


def resolve_document_profile(
    document_type: Optional[str],
    items: list[dict[str, Any]],
) -> tuple[str, Optional[DocumentProfile], float]:
    profiles = load_document_profiles()
    if document_type:
        profile = profiles.document_types.get(document_type)
        if profile:
            return document_type, profile, 1.0
        return document_type, None, 0.0

    full_text = normalize_text(items)
    best_type = "unknown"
    best_profile: Optional[DocumentProfile] = None
    best_score = 0
    best_total = 0

    for candidate_type, profile in profiles.document_types.items():
        score = sum(
            1
            for cue in profile.detect.cues
            if cue.text and label_similarity(cue.text, full_text)
        )
        total = max(len(profile.detect.cues), 1)
        if score >= profile.detect.min_score and score > best_score:
            best_type = candidate_type
            best_profile = profile
            best_score = score
            best_total = total

    confidence = best_score / best_total if best_total else 0.0
    return best_type, best_profile, round(confidence, 4)


# ----------------------------
# Image Utilities
# ----------------------------

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}


def upload_mime_type(file: UploadFile) -> str:
    content_type = (file.content_type or "").lower()
    if content_type in ALLOWED_MIME_TYPES:
        return content_type
    guessed_type = mimetypes.guess_type(file.filename or "")[0]
    return (guessed_type or content_type or "unknown").lower()


def validate_upload(file: UploadFile, data: bytes) -> None:
    mime_type = upload_mime_type(file)
    if mime_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type or 'unknown'}",
        )

    max_bytes = settings.MAX_FILE_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max allowed: {settings.MAX_FILE_MB} MB",
        )

    if not data:
        raise HTTPException(status_code=400, detail="Empty file")


def load_image(data: bytes) -> np.ndarray:
    try:
        pil = Image.open(io.BytesIO(data))
        pil = ImageOps.exif_transpose(pil)
        pil = pil.convert("RGB")
        return np.array(pil)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc


def preprocess_image(
    image_rgb: np.ndarray,
    upscale: bool = True,
    denoise: bool = False,
    threshold: bool = False,
    crop_border: bool = True,
    enhance: bool = True,
    clean_background: bool = settings.CLEAN_BACKGROUND,
    max_image_side: Optional[int] = None,
) -> np.ndarray:
    """
    Production-safe preprocessing:
    - converts RGB to BGR for OpenCV/Paddle
    - crops plain scanner/camera border around the document
    - upscales low-resolution documents without binarizing security patterns
    - boosts local contrast and sharpens thin Devanagari strokes
    """

    if crop_border:
        image_rgb = crop_plain_border(image_rgb)

    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image.shape[:2]

    target_side = max_image_side or 1800
    if upscale and max(w, h) < target_side:
        scale = target_side / max(w, h)
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
    elif max_image_side and max(w, h) > max_image_side:
        scale = max_image_side / max(w, h)
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    if clean_background:
        image = remove_security_background(image)

    if enhance:
        image = enhance_document_contrast(image)

    if denoise:
        image = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=3,
            hColor=3,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    if threshold:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return image


def remove_security_background(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    yellow_background = (
        (hsv[:, :, 0] >= 15)
        & (hsv[:, :, 0] <= 45)
        & (hsv[:, :, 1] > 25)
        & (hsv[:, :, 2] > 120)
    )
    dark_ink = gray < 165
    red_ink = (
        ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170))
        & (hsv[:, :, 1] > 55)
        & (hsv[:, :, 2] < 245)
    )
    blue_ink = (
        (hsv[:, :, 0] >= 90)
        & (hsv[:, :, 0] <= 135)
        & (hsv[:, :, 1] > 45)
        & (hsv[:, :, 2] < 245)
    )

    ink = (dark_ink | red_ink | blue_ink) & ~(yellow_background & (gray > 125))
    mask = (ink.astype("uint8")) * 255
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1)),
        iterations=1,
    )

    cleaned = np.full_like(image_bgr, 255)
    cleaned[mask > 0] = (0, 0, 0)
    return cleaned


def crop_plain_border(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    non_border = gray < 245
    rows = np.where(non_border.any(axis=1))[0]
    cols = np.where(non_border.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        return image_rgb

    pad = 8
    y1 = max(int(rows[0]) - pad, 0)
    y2 = min(int(rows[-1]) + pad + 1, image_rgb.shape[0])
    x1 = max(int(cols[0]) - pad, 0)
    x2 = min(int(cols[-1]) + pad + 1, image_rgb.shape[1])

    cropped = image_rgb[y1:y2, x1:x2]
    if cropped.shape[0] < 100 or cropped.shape[1] < 100:
        return image_rgb

    return cropped


def enhance_document_contrast(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(enhanced, 1.35, blur, -0.35, 0)


def normalize_text(items: list[dict[str, Any]]) -> str:
    return "\n".join(item["text"] for item in items if item["text"])


def filter_ocr_items(
    items: list[dict[str, Any]],
    min_confidence: float = settings.MIN_CONFIDENCE,
) -> list[dict[str, Any]]:
    return [
        item
        for item in items
        if float(item.get("confidence", 0.0)) >= min_confidence
    ]


def run_profile_ocr(
    image_rgb: np.ndarray,
    document_type: Optional[str] = None,
    lang: Optional[str] = None,
    fallback_preprocessing: Optional[PreprocessingConfig] = None,
    accuracy_mode: str = "accurate",
    retry: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, Optional[DocumentProfile], float, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    runtime_meta: dict[str, Any] = {
        "accuracy_mode": accuracy_mode,
        "retry_enabled": retry,
        "ocr_passes_run": [],
        "ocr_passes_skipped": [],
        "retry_fields": [],
        "budget_exhausted": False,
    }
    bootstrap = fallback_preprocessing or PreprocessingConfig(
        clean_background=settings.CLEAN_BACKGROUND
    )
    bootstrap_image = preprocess_image(
        image_rgb,
        upscale=bootstrap.upscale,
        denoise=bootstrap.denoise,
        threshold=bootstrap.threshold,
        crop_border=bootstrap.crop_border,
        enhance=bootstrap.enhance,
        clean_background=bootstrap.clean_background,
        max_image_side=bootstrap.max_image_side,
    )

    engine = OCREngine.instance(lang or settings.OCR_LANG)
    bootstrap_items = engine.read(
        bootstrap_image,
        min_confidence=settings.INTERNAL_MIN_CONFIDENCE,
    )
    for item in bootstrap_items:
        item["source_pass"] = "bootstrap"
        item["source_kind"] = "printed"
        item["script"] = ""
    runtime_meta["ocr_passes_run"].append(
        {"name": "bootstrap", "scope": "full_page", "items": len(bootstrap_items)}
    )

    resolved_type, profile, confidence = resolve_document_profile(
        document_type,
        bootstrap_items,
    )
    if not profile:
        runtime_meta["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        return (
            bootstrap_items,
            filter_ocr_items(bootstrap_items),
            resolved_type,
            None,
            confidence,
            bootstrap_image,
            runtime_meta,
        )

    started = time.perf_counter()
    extraction_items: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()
    primary_processed = bootstrap_image
    timeout_seconds = ocr_timeout_seconds(profile, accuracy_mode)
    max_passes = ocr_max_passes(profile, accuracy_mode)

    for pass_index, pass_config in enumerate(profile.ocr.passes):
        if pass_config.mode != "default":
            runtime_meta["ocr_passes_skipped"].append(
                {"name": pass_config.name, "reason": f"mode:{pass_config.mode}"}
            )
            continue
        if pass_budget_exhausted(runtime_meta, started, timeout_seconds, max_passes):
            runtime_meta["budget_exhausted"] = True
            runtime_meta["ocr_passes_skipped"].append(
                {"name": pass_config.name, "reason": "budget_exhausted"}
            )
            continue

        processed, pass_items = run_ocr_pass(
            image_rgb,
            pass_config,
            lang=lang,
            default_max_side=profile.ocr.max_image_side,
        )
        if pass_index == 0:
            primary_processed = processed
        append_ocr_items(extraction_items, pass_items, pass_config, seen)
        runtime_meta["ocr_passes_run"].append(
            {"name": pass_config.name, "scope": "full_page", "items": len(pass_items)}
        )

    fields = extract_structured_fields(extraction_items, primary_processed, profile)
    if retry:
        retry_profile_ocr(
            image_rgb,
            primary_processed,
            profile,
            fields,
            extraction_items,
            seen,
            runtime_meta,
            started,
            timeout_seconds,
            max_passes,
            lang,
        )

    items = filter_ocr_items(extraction_items)
    runtime_meta["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    return extraction_items, items, resolved_type, profile, confidence, primary_processed, runtime_meta


def run_ocr_pass(
    image_rgb: np.ndarray,
    pass_config: OCRPassConfig,
    lang: Optional[str] = None,
    default_max_side: Optional[int] = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    processed = preprocess_image(
        image_rgb,
        upscale=pass_config.preprocessing.upscale,
        denoise=pass_config.preprocessing.denoise,
        threshold=pass_config.preprocessing.threshold,
        crop_border=pass_config.preprocessing.crop_border,
        enhance=pass_config.preprocessing.enhance,
        clean_background=pass_config.preprocessing.clean_background,
        max_image_side=pass_config.preprocessing.max_image_side or default_max_side,
    )
    pass_engine = OCREngine.instance(
        lang or pass_config.lang,
        text_detection_model=pass_config.text_detection_model,
        text_recognition_model=resolved_text_recognition_model(pass_config),
        ocr_version=pass_config.ocr_version,
    )
    pass_items = pass_engine.read(
        processed,
        min_confidence=pass_config.min_confidence or settings.INTERNAL_MIN_CONFIDENCE,
    )
    return processed, pass_items


def append_ocr_items(
    output: list[dict[str, Any]],
    items: list[dict[str, Any]],
    pass_config: OCRPassConfig,
    seen: set[tuple[str, tuple[int, int, int, int]]],
    offset: tuple[int, int] = (0, 0),
) -> None:
    for item in items:
        if offset != (0, 0):
            item["box"] = offset_box(item.get("box") or [], offset)
        bounds = box_bounds(item.get("box") or []) or (0, 0, 0, 0)
        key = (
            clean_ocr_line(item.get("text", "")),
            tuple(int(round(value / 8)) for value in bounds),
        )
        if key in seen:
            continue
        seen.add(key)
        item["source_pass"] = pass_config.name
        item["source_kind"] = pass_config.source_kind
        item["script"] = pass_config.script or ""
        output.append(item)


def offset_box(box: Any, offset: tuple[int, int]) -> Any:
    if hasattr(box, "tolist"):
        box = box.tolist()
    if not isinstance(box, list):
        return box
    dx, dy = offset
    shifted: list[Any] = []
    for point in box:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            shifted.append([point[0] + dx, point[1] + dy, *point[2:]])
        else:
            shifted.append(point)
    return shifted


def ocr_timeout_seconds(profile: DocumentProfile, accuracy_mode: str) -> float:
    if profile.ocr.timeout_seconds:
        return profile.ocr.timeout_seconds
    return {"fast": 5.0, "balanced": 8.0, "accurate": 10.0}.get(accuracy_mode, 10.0)


def ocr_max_passes(profile: DocumentProfile, accuracy_mode: str) -> int:
    if profile.ocr.max_passes:
        return profile.ocr.max_passes
    return {"fast": 2, "balanced": 3, "accurate": 4}.get(accuracy_mode, 4)


def pass_budget_exhausted(
    runtime_meta: dict[str, Any],
    started: float,
    timeout_seconds: float,
    max_passes: int,
) -> bool:
    if len(runtime_meta["ocr_passes_run"]) >= max_passes:
        return True
    return False


def retry_profile_ocr(
    image_rgb: np.ndarray,
    processed_image: np.ndarray,
    profile: DocumentProfile,
    fields: dict[str, OCRField],
    extraction_items: list[dict[str, Any]],
    seen: set[tuple[str, tuple[int, int, int, int]]],
    runtime_meta: dict[str, Any],
    started: float,
    timeout_seconds: float,
    max_passes: int,
    lang: Optional[str],
) -> None:
    retry_passes = [ocr_pass for ocr_pass in profile.ocr.passes if ocr_pass.mode == "retry_only"]
    if not retry_passes:
        return

    for pass_config in retry_passes:
        retry_fields = fields_for_retry(profile.fields, fields, pass_config)
        if not retry_fields:
            runtime_meta["ocr_passes_skipped"].append(
                {"name": pass_config.name, "reason": "no_retry_fields"}
            )
            continue

        for field_name in retry_fields:
            if pass_budget_exhausted(runtime_meta, started, timeout_seconds, max_passes):
                runtime_meta["budget_exhausted"] = True
                runtime_meta["ocr_passes_skipped"].append(
                    {"name": pass_config.name, "field": field_name, "reason": "budget_exhausted"}
                )
                return

            field_config = profile.fields[field_name]
            crop = retry_crop_for_field(processed_image, fields.get(field_name), field_config)
            if not crop:
                runtime_meta["ocr_passes_skipped"].append(
                    {"name": pass_config.name, "field": field_name, "reason": "no_retry_region"}
                )
                continue

            crop_image, offset = crop
            retry_image = preprocess_retry_crop(crop_image, pass_config.preprocessing)
            pass_engine = OCREngine.instance(
                lang or pass_config.lang,
                text_detection_model=pass_config.text_detection_model,
                text_recognition_model=resolved_text_recognition_model(pass_config),
                ocr_version=pass_config.ocr_version,
            )
            pass_items = pass_engine.read(
                retry_image,
                min_confidence=pass_config.min_confidence or settings.INTERNAL_MIN_CONFIDENCE,
            )
            append_ocr_items(extraction_items, pass_items, pass_config, seen, offset)
            runtime_meta["ocr_passes_run"].append(
                {
                    "name": pass_config.name,
                    "scope": "field_crop",
                    "field": field_name,
                    "items": len(pass_items),
                }
            )
            if field_name not in runtime_meta["retry_fields"]:
                runtime_meta["retry_fields"].append(field_name)


def preprocess_retry_crop(
    image_bgr: np.ndarray,
    preprocessing: PreprocessingConfig,
) -> np.ndarray:
    image = image_bgr.copy()
    if preprocessing.clean_background:
        image = remove_security_background(image)
    if preprocessing.enhance:
        image = enhance_document_contrast(image)
    if preprocessing.denoise:
        image = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=3,
            hColor=3,
            templateWindowSize=7,
            searchWindowSize=21,
        )
    if preprocessing.threshold:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return image


def fields_for_retry(
    configured_fields: dict[str, ProfileFieldConfig],
    fields: dict[str, OCRField],
    pass_config: OCRPassConfig,
) -> list[str]:
    candidate_fields = pass_config.retry_fields or list(configured_fields)
    selected: list[str] = []
    threshold = pass_config.run_if_below_confidence

    for field_name in candidate_fields:
        field_config = configured_fields.get(field_name)
        if not field_config:
            continue

        field = fields.get(field_name)
        missing_requested = (
            not pass_config.run_if_missing_fields
            or field_name in pass_config.run_if_missing_fields
        )
        if field is None and missing_requested:
            selected.append(field_name)
            continue

        if field is None:
            continue

        if threshold is not None and field.confidence < threshold:
            selected.append(field_name)
            continue

        if field.requires_review and field_config.review_source_kinds:
            selected.append(field_name)

    return list(dict.fromkeys(selected))


def retry_crop_for_field(
    processed_image: np.ndarray,
    field: Optional[OCRField],
    field_config: ProfileFieldConfig,
) -> Optional[tuple[np.ndarray, tuple[int, int]]]:
    height, width = processed_image.shape[:2]
    bounds: Optional[tuple[float, float, float, float]] = None

    region = field_config.retry_region or field_config.anchor_region
    if region and len(region) == 4:
        bounds = normalized_region_to_bounds(region, width, height)
    elif field:
        bounds = evidence_bounds(field.evidence)

    if not bounds:
        return None

    padding = 32
    x1 = max(int(bounds[0]) - padding, 0)
    y1 = max(int(bounds[1]) - padding, 0)
    x2 = min(int(bounds[2]) + padding, width)
    y2 = min(int(bounds[3]) + padding, height)
    if x2 <= x1 or y2 <= y1:
        return None

    return processed_image[y1:y2, x1:x2].copy(), (x1, y1)


def evidence_bounds(evidence: list[dict[str, Any]]) -> Optional[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for item in evidence:
        bounds = item.get("bounds")
        if isinstance(bounds, list) and len(bounds) == 4:
            boxes.append(tuple(float(value) for value in bounds))

    if not boxes:
        return None

    return (
        min(bounds[0] for bounds in boxes),
        min(bounds[1] for bounds in boxes),
        max(bounds[2] for bounds in boxes),
        max(bounds[3] for bounds in boxes),
    )


def resolved_text_recognition_model(pass_config: OCRPassConfig) -> Optional[str]:
    if pass_config.text_recognition_model:
        return pass_config.text_recognition_model

    if (pass_config.script or "").strip().lower() == "devanagari":
        return settings.DEVANAGARI_TEXT_RECOGNITION_MODEL

    return None


def extract_structured_fields(
    items: list[dict[str, Any]],
    image: Optional[np.ndarray] = None,
    profile: Optional[DocumentProfile] = None,
) -> dict[str, OCRField]:
    lines = [item for item in items if item.get("text", "").strip()]
    fields: dict[str, OCRField] = {}
    configured_fields = profile.fields if profile else {}

    extract_anchor_region_fields(fields, lines, configured_fields, image)

    for index, item in enumerate(lines):
        text = clean_ocr_line(item["text"])
        confidence = float(item.get("confidence", 0.0))

        if not text:
            continue

        for field_name, field_config in configured_fields.items():
            if not item_allowed_for_field(item, field_config):
                continue
            labels = field_config.labels

            details: dict[str, Any] = {}
            value = ""
            if not field_config.strategies or "same_line_after_label" in field_config.strategies:
                value = value_after_label(text, labels)
            if value and not is_probable_value(value):
                value = ""
            if value:
                details = {
                    "method": "same_line_after_label",
                    "label": matched_label(text, labels),
                    "ocr_line": ocr_line_detail(item, index),
                }
            if not value and has_label(text, labels) and (
                not field_config.strategies
                or "same_row_right_of_label" in field_config.strategies
                or "same_row_left_of_label" in field_config.strategies
                or "below_label" in field_config.strategies
            ):
                nearby = value_near_label(
                    lines,
                    index,
                    strategies=field_config.strategies,
                    field_config=field_config,
                    prefer_devanagari="clean_devanagari_name" in field_config.normalizers,
                )
                if nearby:
                    value = nearby["value"]
                    details = nearby["details"]
                    confidence = float(nearby.get("confidence", confidence))

            if value:
                if "append_following_line" in field_config.strategies:
                    following = following_line_value(lines, index, field_config)
                    if following:
                        value = f"{value} {following['value']}"
                        details["following_value_ocr_line"] = following["line"]
                        confidence = min(confidence, following["confidence"])
                add_field(fields, field_name, value, confidence, text, details, field_config)

    extract_regex_fields(fields, lines, configured_fields)
    for field_name, field_config in configured_fields.items():
        if "citizenship_number" in field_config.normalizers:
            refine_citizenship_number(fields, lines, field_name, field_config)
        if (
            "bs_date_components" in field_config.normalizers
            or "ad_date_components" in field_config.normalizers
        ):
            refine_date_of_birth(fields, lines, field_name, field_config)
    apply_profile_normalizers(fields, image, configured_fields)
    apply_template_fields(fields, configured_fields)
    infer_unlabeled_fields(fields, lines, configured_fields)
    return fields


def field_values(fields: dict[str, OCRField]) -> dict[str, str]:
    return {name: field.value for name, field in fields.items()}


def extract_regex_fields(
    fields: dict[str, OCRField],
    lines: list[dict[str, Any]],
    configured_fields: dict[str, ProfileFieldConfig],
) -> None:
    full_text = normalize_text(lines)
    ascii_text = translate_configured_digits(full_text)
    for field_name, field_config in configured_fields.items():
        if field_name in fields:
            continue
        if "regex_from_full_text" not in field_config.strategies:
            continue
        pattern = field_config.regex
        if not pattern and field_config.validators:
            pattern = next(
                (
                    validator.pattern
                    for validator in field_config.validators
                    if validator.type == "regex" and validator.pattern
                ),
                None,
            )
        if not pattern:
            continue
        match = re.search(pattern, full_text) or re.search(pattern, ascii_text)
        if not match:
            continue
        add_field(
            fields,
            field_name,
            match.group(0),
            0.5,
            match.group(0),
            {
                "method": "regex_from_full_text",
                "pattern": pattern,
            },
            field_config,
        )


def extract_anchor_region_fields(
    fields: dict[str, OCRField],
    lines: list[dict[str, Any]],
    configured_fields: dict[str, ProfileFieldConfig],
    image: Optional[np.ndarray],
) -> None:
    if image is None:
        return

    for field_name, field_config in configured_fields.items():
        if field_name in fields:
            continue
        if "anchor_region" not in field_config.strategies:
            continue

        region_value = value_from_anchor_region(lines, field_config, image)
        if not region_value:
            continue

        add_field(
            fields,
            field_name,
            region_value["value"],
            region_value["confidence"],
            region_value["value"],
            region_value["details"],
            field_config,
        )


def value_from_anchor_region(
    lines: list[dict[str, Any]],
    field_config: ProfileFieldConfig,
    image: np.ndarray,
) -> Optional[dict[str, Any]]:
    region = field_config.anchor_region or []
    if len(region) != 4:
        return None

    height, width = image.shape[:2]
    x1, y1, x2, y2 = normalized_region_to_bounds(region, width, height)
    region_items: list[tuple[float, float, int, dict[str, Any]]] = []

    for index, item in enumerate(lines):
        if not item_allowed_for_field(item, field_config):
            continue
        text = clean_ocr_line(item.get("text", ""))
        if not text or len(normalized_for_match(text)) < 2:
            continue
        if is_label_only_value(text):
            continue
        if has_label(text, field_config.labels):
            continue

        bounds = box_bounds(item.get("box") or [])
        if not bounds:
            continue

        bx1, by1, bx2, by2 = bounds
        mid_x = (bx1 + bx2) / 2
        mid_y = (by1 + by2) / 2
        if x1 <= mid_x <= x2 and y1 <= mid_y <= y2:
            region_items.append((by1, bx1, index, item))

    if not region_items:
        return None

    region_items = dedupe_region_items(region_items)
    region_items.sort(key=lambda row: (row[0], row[1]))
    texts = [clean_ocr_line(item.get("text", "")) for _, _, _, item in region_items]
    confidences = [float(item.get("confidence", 0.0)) for _, _, _, item in region_items]
    value = clean_ocr_line(" ".join(texts))
    if not value:
        return None

    evidence_lines = [
        ocr_line_detail(item, index)
        for _, _, index, item in region_items
    ]
    return {
        "value": value,
        "confidence": min(confidences) if confidences else 0.0,
        "details": {
            "method": "anchor_region",
            "region": [round(value, 4) for value in region],
            "value_ocr_lines": evidence_lines,
        },
    }


def normalized_region_to_bounds(
    region: list[float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(value) for value in region]
    if max(abs(value) for value in (x1, y1, x2, y2)) <= 1.0:
        return x1 * width, y1 * height, x2 * width, y2 * height
    return x1, y1, x2, y2


def item_allowed_for_field(
    item: dict[str, Any],
    field_config: ProfileFieldConfig,
) -> bool:
    if field_config.source_passes and item.get("source_pass", "") not in field_config.source_passes:
        return False
    if field_config.source_kinds and item.get("source_kind", "") not in field_config.source_kinds:
        return False
    return True


def dedupe_region_items(
    region_items: list[tuple[float, float, int, dict[str, Any]]],
) -> list[tuple[float, float, int, dict[str, Any]]]:
    kept: list[tuple[float, float, int, dict[str, Any]]] = []
    for candidate in sorted(
        region_items,
        key=lambda row: float(row[3].get("confidence", 0.0)),
        reverse=True,
    ):
        candidate_bounds = box_bounds(candidate[3].get("box") or [])
        if not candidate_bounds:
            continue
        if any(
            boxes_overlap_ratio(candidate_bounds, box_bounds(item.get("box") or [])) > 0.75
            for _, _, _, item in kept
        ):
            continue
        kept.append(candidate)
    return kept


def boxes_overlap_ratio(
    first: tuple[float, float, float, float],
    second: Optional[tuple[float, float, float, float]],
) -> float:
    if not second:
        return 0.0
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    first_area = max((ax2 - ax1) * (ay2 - ay1), 1.0)
    second_area = max((bx2 - bx1) * (by2 - by1), 1.0)
    return intersection / min(first_area, second_area)


def apply_profile_normalizers(
    fields: dict[str, OCRField],
    image: Optional[np.ndarray],
    configured_fields: dict[str, ProfileFieldConfig],
) -> None:
    for field_name, field_config in configured_fields.items():
        if field_name not in fields:
            continue

        if "nepali_digits_to_ascii" in field_config.normalizers:
            normalize_existing_field(fields, field_name, field_config, "nepali_digits_to_ascii")

        if "gazetteer_match" in field_config.strategies or field_config.gazetteer:
            repair_single_location_field(fields, field_name, field_config, image)

        if field_name in fields:
            mark_review_if_needed(
                fields,
                field_name,
                field_config.review_threshold or settings.MIN_CONFIDENCE,
            )

    for field_name, field_config in configured_fields.items():
        if field_config.same_as_field:
            repair_same_as_field(fields, field_name, field_config.same_as_field)
        if "person_name_repair" in field_config.normalizers and field_config.consistency_field:
            repair_person_name_field(
                fields,
                field_name,
                field_config.consistency_field,
                image,
            )


def apply_template_fields(
    fields: dict[str, OCRField],
    configured_fields: dict[str, ProfileFieldConfig],
) -> None:
    for field_name, field_config in configured_fields.items():
        if not field_config.template:
            continue
        if field_name in fields and "template" not in field_config.strategies:
            continue

        rendered = render_field_template(field_config.template, fields)
        if not rendered:
            continue

        dependencies = template_dependencies(field_config.template)
        evidence = [
            {
                "type": "template_source",
                "field": dependency,
                "value": fields[dependency].value,
                "confidence": fields[dependency].confidence,
            }
            for dependency in dependencies
            if dependency in fields
        ]
        confidence_values = [
            fields[dependency].confidence
            for dependency in dependencies
            if dependency in fields
        ]
        fields[field_name] = OCRField(
            value=rendered,
            confidence=round(min(confidence_values) if confidence_values else 0.0, 4),
            source_text=rendered,
            raw_value=rendered,
            normalized_value=rendered,
            requires_review=any(
                fields[dependency].requires_review
                for dependency in dependencies
                if dependency in fields
            ),
            evidence=evidence,
            details={
                "method": "template",
                "template": field_config.template,
                "dependencies": dependencies,
            },
        )


def render_field_template(template: str, fields: dict[str, OCRField]) -> str:
    missing = [
        name
        for name in template_dependencies(template)
        if name not in fields or not fields[name].value
    ]
    if missing:
        return ""

    value = template
    for name in template_dependencies(template):
        value = value.replace(f"{{{{{name}}}}}", fields[name].value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def template_dependencies(template: str) -> list[str]:
    return re.findall(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}", template)


def normalize_existing_field(
    fields: dict[str, OCRField],
    field_name: str,
    field_config: ProfileFieldConfig,
    normalizer: str,
) -> None:
    field = fields[field_name]
    if normalizer == "nepali_digits_to_ascii":
        value = translate_configured_digits(field.value)
    else:
        return

    if value == field.value:
        return

    fields[field_name] = OCRField(
        value=value,
        confidence=field.confidence,
        source_text=field.source_text,
        raw_value=field.raw_value or field.value,
        normalized_value=value,
        requires_review=field.requires_review,
        evidence=field.evidence,
        details={
            **field.details,
            "normalizers": [
                *field.details.get("normalizers", []),
                normalizer,
            ],
        },
    )


def mark_review_if_needed(
    fields: dict[str, OCRField],
    field_name: str,
    threshold: float,
) -> None:
    field = fields[field_name]
    if field.confidence >= threshold:
        return
    details = dict(field.details)
    details["requires_review"] = True
    details["review_reason"] = "low_confidence"
    fields[field_name] = OCRField(
        value=field.value,
        confidence=field.confidence,
        source_text=field.source_text,
        raw_value=field.raw_value or field.value,
        normalized_value=field.normalized_value or field.value,
        requires_review=True,
        evidence=field.evidence,
        details=details,
    )


def add_field(
    fields: dict[str, OCRField],
    name: str,
    value: str,
    confidence: float,
    source_text: str,
    details: Optional[dict[str, Any]] = None,
    field_config: Optional[ProfileFieldConfig] = None,
) -> None:
    normalized = normalize_field_value(name, value, field_config)
    if not normalized or is_label_only_value(normalized):
        return

    existing = fields.get(name)
    if existing and existing.confidence >= confidence:
        return

    details = details or {}
    requires_review = field_requires_source_review(details, field_config)
    if requires_review:
        details = {
            **details,
            "requires_review": True,
            "review_reason": "handwritten_or_uncertain_source",
        }

    fields[name] = OCRField(
        value=normalized,
        confidence=round(float(confidence), 4),
        source_text=source_text,
        raw_value=value,
        normalized_value=normalized,
        requires_review=requires_review,
        evidence=details_to_evidence(details),
        details=details,
    )


def field_requires_source_review(
    details: dict[str, Any],
    field_config: Optional[ProfileFieldConfig],
) -> bool:
    if not field_config or not field_config.review_source_kinds:
        return False

    review_kinds = set(field_config.review_source_kinds)
    for evidence in details_to_evidence(details):
        if evidence.get("source_kind") in review_kinds:
            return True

    return False


def details_to_evidence(details: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if not details:
        return []
    evidence: list[dict[str, Any]] = []
    for key in ("ocr_line", "label_ocr_line", "value_ocr_line", "following_value_ocr_line"):
        value = details.get(key)
        if isinstance(value, dict):
            evidence.append({"type": key, **value})
    if "value_ocr_lines" in details and isinstance(details["value_ocr_lines"], list):
        for value in details["value_ocr_lines"]:
            if isinstance(value, dict):
                evidence.append({"type": "value_ocr_line", **value})
    if "fragments" in details and isinstance(details["fragments"], list):
        for fragment in details["fragments"]:
            if isinstance(fragment, dict):
                evidence.append({"type": "fragment", **fragment})
    return evidence


def following_line_value(
    lines: list[dict[str, Any]],
    index: int,
    field_config: ProfileFieldConfig,
) -> Optional[dict[str, Any]]:
    if index + 1 >= len(lines):
        return None
    item = lines[index + 1]
    if not item_allowed_for_field(item, field_config):
        return None
    value = clean_ocr_line(item.get("text", ""))
    if not value or is_label_only_value(value) or has_label(value, field_config.labels):
        return None
    return {
        "value": value,
        "confidence": float(item.get("confidence", 0.0)),
        "line": ocr_line_detail(item, index + 1),
    }


def clean_ocr_line(text: str) -> str:
    text = text.strip()
    separator_pattern = extraction_rules().clean_separator_pattern
    if separator_pattern:
        text = re.sub(separator_pattern, " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_field_value(
    field_name: str,
    value: str,
    field_config: Optional[ProfileFieldConfig] = None,
) -> str:
    value = clean_ocr_line(value)
    value = strip_embedded_labels(field_name, value, field_config)
    rules = extraction_rules()
    if rules.trim_leading_pattern:
        value = re.sub(rules.trim_leading_pattern, "", value)
    strip_prefix_pattern = extraction_rules().strip_prefix_pattern
    if strip_prefix_pattern:
        value = re.sub(strip_prefix_pattern, "", value)
    if rules.trim_trailing_pattern:
        value = re.sub(rules.trim_trailing_pattern, "", value)

    for pattern in extraction_rules().split_before_patterns.get(field_name, []):
        value = re.split(pattern, value, maxsplit=1)[0]

    return value.strip()


def strip_embedded_labels(
    field_name: str,
    value: str,
    field_config: Optional[ProfileFieldConfig] = None,
) -> str:
    labels = field_config.labels if field_config else configured_labels_for_field(field_name)
    cleaned = value

    for label in labels:
        pattern = re.escape(label).replace("\\ ", r"\s*")
        cleaned = re.sub(rf"^\s*{pattern}\s*[:\-–—]?\s*", "", cleaned)

    return cleaned


def is_label_only_value(value: str) -> bool:
    value_key = normalized_for_match(value)
    if not value_key:
        return True

    label_keys = {
        normalized_for_match(label)
        for labels in all_configured_labels().values()
        for label in labels
    }
    label_keys.update(normalized_for_match(value) for value in extraction_rules().label_only_values)

    if value_key in label_keys:
        return True

    label_fragments = extraction_rules().label_fragments
    fragment_hits = sum(1 for fragment in label_fragments if fragment in value)
    return fragment_hits >= 2 and len(value_key) < 18


def extraction_rules() -> ExtractionRules:
    return load_document_profiles().extraction_rules


def translate_configured_digits(text: str) -> str:
    mappings = extraction_rules().digit_mappings
    if not mappings:
        return text
    return text.translate(str.maketrans(mappings))


def all_configured_labels() -> dict[str, list[str]]:
    labels: dict[str, list[str]] = {}
    for profile in load_document_profiles().document_types.values():
        for field_name, field_config in profile.fields.items():
            labels.setdefault(field_name, [])
            labels[field_name].extend(field_config.labels)
    return labels


def configured_labels_for_field(field_name: str) -> list[str]:
    return all_configured_labels().get(field_name, [])


def normalized_for_match(text: str) -> str:
    return re.sub(extraction_rules().normalize_separator_pattern, "", text)


def label_similarity(label: str, text: str) -> bool:
    return normalized_for_match(label) in normalized_for_match(text)


def has_label(text: str, labels: list[str]) -> bool:
    return any(label_similarity(label, text) for label in labels)


def matched_label(text: str, labels: list[str]) -> str:
    for label in labels:
        if label_similarity(label, text):
            return label
    return ""


def value_after_label(text: str, labels: list[str]) -> str:
    for label in labels:
        pattern = re.escape(label).replace("\\ ", r"\s*")
        match = re.search(pattern + r"\s*[:\-–—]?\s*(.+)$", text)
        if match:
            return match.group(1)

    return ""


def extract_before_label(text: str, labels: list[str]) -> str:
    for label in labels:
        position = text.find(label)
        if position > 0:
            return text[:position]
    return ""


def value_near_label(
    lines: list[dict[str, Any]],
    index: int,
    strategies: Optional[list[str]] = None,
    field_config: Optional[ProfileFieldConfig] = None,
    prefer_devanagari: bool = False,
) -> Optional[dict[str, Any]]:
    strategies = strategies or ["same_row_right_of_label", "below_label"]
    label_box = lines[index].get("box") or []
    label_bounds = box_bounds(label_box)
    if label_bounds:
        lx1, ly1, lx2, ly2 = label_bounds
        label_mid_y = (ly1 + ly2) / 2
        label_height = max(ly2 - ly1, 1)
        candidates: list[tuple[float, dict[str, Any]]] = []

        for other_index, item in enumerate(lines):
            if other_index == index:
                continue
            if field_config and not item_allowed_for_field(item, field_config):
                continue

            value = clean_ocr_line(item.get("text", ""))
            if not is_probable_value(value):
                continue
            if prefer_devanagari and not contains_devanagari(value):
                continue

            bounds = box_bounds(item.get("box") or [])
            if not bounds:
                continue

            ox1, oy1, ox2, oy2 = bounds
            other_mid_y = (oy1 + oy2) / 2
            same_row = abs(other_mid_y - label_mid_y) <= max(label_height * 0.9, 28)
            row_direction: Optional[tuple[str, float]] = None
            if (
                "same_row_right_of_label" in strategies
                and same_row
                and ox1 >= lx1
                and ox2 > lx2
            ):
                row_direction = ("same_row_right_of_label", ox1 - lx2)
            elif (
                "same_row_left_of_label" in strategies
                and same_row
                and ox2 <= lx2
                and ox1 < lx1
            ):
                row_direction = ("same_row_left_of_label", lx1 - ox2)

            if row_direction:
                method, distance = row_direction
                candidates.append(
                    (
                        candidate_rank(value, distance, prefer_devanagari),
                        {
                            "value": value,
                            "confidence": float(item.get("confidence", 0.0)),
                            "details": {
                                "method": method,
                                "label_ocr_line": ocr_line_detail(lines[index], index),
                                "value_ocr_line": ocr_line_detail(item, other_index),
                                "distance_px": round(float(distance), 2),
                            },
                        },
                    )
                )

        if candidates:
            candidates.sort(key=lambda pair: pair[0])
            best = candidates[0][1]
            best["details"]["candidate_count"] = len(candidates)
            best["details"]["candidate_values"] = [
                candidate["value"] for _, candidate in candidates[:5]
            ]
            return best

    if "below_label" not in strategies:
        return None

    for offset in (1, 2):
        next_index = index + offset
        if next_index >= len(lines):
            continue

        value = clean_ocr_line(lines[next_index].get("text", ""))
        if field_config and not item_allowed_for_field(lines[next_index], field_config):
            continue
        if is_probable_value(value):
            if prefer_devanagari and not contains_devanagari(value):
                continue
            return {
                "value": value,
                "confidence": float(lines[next_index].get("confidence", 0.0)),
                "details": {
                    "method": "following_line_after_label",
                    "label_ocr_line": ocr_line_detail(lines[index], index),
                    "value_ocr_line": ocr_line_detail(lines[next_index], next_index),
                    "line_offset": offset,
                },
            }

    return None


def contains_devanagari(text: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", text))


def candidate_rank(value: str, distance: float, prefer_devanagari: bool) -> float:
    rank = max(distance, 0)
    if prefer_devanagari and contains_devanagari(value):
        rank -= 250
    if len(normalized_for_match(value)) <= 2:
        rank += 200
    return rank


def box_bounds(box: Any) -> Optional[tuple[float, float, float, float]]:
    if not box:
        return None

    points = box.tolist() if hasattr(box, "tolist") else box
    if not isinstance(points, list):
        return None

    flat_points: list[list[float]] = []
    for point in points:
        if isinstance(point, list) and len(point) >= 2:
            flat_points.append(point)

    if not flat_points:
        return None

    xs = [float(point[0]) for point in flat_points]
    ys = [float(point[1]) for point in flat_points]
    return min(xs), min(ys), max(xs), max(ys)


def ocr_line_detail(item: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "index": index,
        "text": item.get("text", ""),
        "clean_text": clean_ocr_line(item.get("text", "")),
        "confidence": round(float(item.get("confidence", 0.0)), 4),
        "box": item.get("box") or [],
        "bounds": list(box_bounds(item.get("box") or []) or ()),
        "source_pass": item.get("source_pass", "default"),
        "source_kind": item.get("source_kind", "printed"),
        "script": item.get("script", ""),
    }


def is_probable_value(text: str) -> bool:
    if not text or len(text) < 2:
        return False

    rules = extraction_rules()
    if normalized_for_match(text) in {
        normalized_for_match(value) for value in rules.label_only_values
    }:
        return False

    return not any(word in text for word in rules.label_fragments)


def repair_single_location_field(
    fields: dict[str, OCRField],
    field_name: str,
    field_config: ProfileFieldConfig,
    image: Optional[np.ndarray],
) -> None:
    field = fields.get(field_name)
    if not field:
        return

    tesseract_text = tesseract_text_for_field(field, image)
    evidence_values = [
        field.value,
        *devanagari_tokens(field.value),
        *latin_tokens(field.value),
        *devanagari_tokens(tesseract_text),
        *latin_tokens(tesseract_text),
    ]
    for hint_field in field_config.gazetteer_hint_fields:
        if hint_field in fields:
            evidence_values.append(fields[hint_field].value)

    repaired_value = best_gazetteer_location_match(
        evidence_values,
        threshold=field_config.review_threshold or 0.68,
        gazetteer_path=field_config.gazetteer,
    )
    if not repaired_value or repaired_value == field.value:
        return

    update_repaired_field(
        fields,
        field_name,
        repaired_value,
        field.confidence,
        field.source_text,
        {
            "method": "location_gazetteer_match",
            "previous_value": field.value,
            "tesseract_text": tesseract_text,
            "gazetteer": field_config.gazetteer or str(settings.LOCATION_GAZETTEER_PATH),
            "previous_details": field.details,
        },
    )


def best_gazetteer_location_match(
    evidence_values: list[str],
    threshold: float,
    gazetteer_path: Optional[str] = None,
) -> str:
    gazetteer = location_gazetteer(gazetteer_path)
    evidence = [
        normalize_field_value("", value)
        for value in evidence_values
        if value
    ]
    evidence = [value for value in evidence if value]
    if not gazetteer or not evidence:
        return ""

    best_location = ""
    best_similarity = 0.0
    for location in gazetteer:
        similarity = max(text_similarity(value, location) for value in evidence)
        if similarity > best_similarity:
            best_location = location
            best_similarity = similarity

    if not best_location or best_similarity < threshold:
        return ""

    return best_location


_location_gazetteer_cache: dict[str, list[str]] = {}


def location_gazetteer(gazetteer_path: Optional[str] = None) -> list[str]:
    path = Path(gazetteer_path) if gazetteer_path else settings.LOCATION_GAZETTEER_PATH
    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        return []

    cache_key = str(path.resolve())
    if cache_key in _location_gazetteer_cache:
        return _location_gazetteer_cache[cache_key]

    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        values = flatten_gazetteer_values(data)
    else:
        values = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    unique_values = list(dict.fromkeys(value for value in values if value))
    _location_gazetteer_cache[cache_key] = unique_values
    return unique_values


def flatten_gazetteer_values(data: Any) -> list[str]:
    if isinstance(data, str):
        value = data.strip()
        return [value] if value else []

    if isinstance(data, list):
        values: list[str] = []
        for item in data:
            values.extend(flatten_gazetteer_values(item))
        return values

    if isinstance(data, dict):
        values: list[str] = []
        for value in data.values():
            values.extend(flatten_gazetteer_values(value))
        return values

    return []


def devanagari_tokens(text: str) -> list[str]:
    strip_chars = extraction_rules().token_strip_chars
    return [
        token.strip(strip_chars)
        for token in re.findall(r"[\u0900-\u097F]+", text)
        if len(token.strip(strip_chars)) >= 3
    ]


def latin_tokens(text: str) -> list[str]:
    return [
        token.strip(".,;:-_ ")
        for token in re.findall(r"[A-Za-z]+", text)
        if len(token.strip(".,;:-_ ")) >= 3
    ]


def repair_same_as_field(
    fields: dict[str, OCRField],
    field_name: str,
    source_field_name: str,
) -> None:
    field = fields.get(field_name)
    source = fields.get(source_field_name)
    if not field or not source:
        return

    similarity = text_similarity(field.value, source.value)
    if similarity < 0.82:
        return

    update_repaired_field(
        fields,
        field_name,
        source.value,
        max(field.confidence, source.confidence),
        field.source_text,
        {
            "method": "same_as_field",
            "previous_value": field.value,
            "source_field": source_field_name,
            "source_value": source.value,
            "similarity": round(similarity, 4),
            "previous_details": field.details,
        },
    )


def repair_person_name_field(
    fields: dict[str, OCRField],
    field_name: str,
    consistency_field_name: str,
    image: Optional[np.ndarray],
) -> None:
    field = fields.get(field_name)
    consistency_field = fields.get(consistency_field_name)
    if not field or not consistency_field:
        return

    original_tokens = split_name_tokens(field.value)
    if not original_tokens:
        return

    repaired_tokens = original_tokens[:]
    repairs: list[dict[str, Any]] = []

    tesseract_text = tesseract_text_for_field(field, image)
    tesseract_first_name = first_significant_devanagari_token(tesseract_text)
    if (
        tesseract_first_name
        and len(tesseract_first_name) > len(repaired_tokens[0]) + 1
        and text_similarity(tesseract_first_name, repaired_tokens[0]) < 0.7
    ):
        repairs.append(
            {
                "component": "given_name",
                "previous": repaired_tokens[0],
                "replacement": tesseract_first_name,
                "source": "tesseract_nep_field_crop",
                "tesseract_text": tesseract_text,
            }
        )
        repaired_tokens[0] = tesseract_first_name

    consistency_tokens = split_name_tokens(consistency_field.value)
    if consistency_tokens:
        full_surname = consistency_tokens[-1]
        current_surname = repaired_tokens[-1]
        surname_similarity = text_similarity(current_surname, full_surname)
        if current_surname != full_surname and surname_similarity >= 0.65:
            repairs.append(
                {
                    "component": "surname",
                    "previous": current_surname,
                    "replacement": full_surname,
                    "source": "surname_consistency_with_full_name",
                    "similarity": round(surname_similarity, 4),
                }
            )
            repaired_tokens[-1] = full_surname

    if not repairs:
        return

    repaired_value = " ".join(repaired_tokens)
    update_repaired_field(
        fields,
        field_name,
        repaired_value,
        field.confidence,
        field.source_text,
        {
            "method": "person_name_repair",
            "previous_value": field.value,
            "repairs": repairs,
            "previous_details": field.details,
        },
    )


def update_repaired_field(
    fields: dict[str, OCRField],
    name: str,
    value: str,
    confidence: float,
    source_text: str,
    repair_details: dict[str, Any],
) -> None:
    fields[name] = OCRField(
        value=normalize_field_value(name, value),
        confidence=round(float(confidence), 4),
        source_text=source_text,
        raw_value=repair_details.get("previous_value", value),
        normalized_value=normalize_field_value(name, value),
        requires_review=bool(repair_details.get("requires_review", False)),
        evidence=details_to_evidence(repair_details.get("previous_details", {})),
        details=repair_details,
    )


def split_name_tokens(value: str) -> list[str]:
    return [token for token in clean_ocr_line(value).split(" ") if token]


def text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(
        None,
        normalized_for_match(left).casefold(),
        normalized_for_match(right).casefold(),
    ).ratio()


def tesseract_text_for_field(
    field: OCRField,
    image: Optional[np.ndarray],
) -> str:
    if image is None or not settings.USE_TESSERACT_REPAIR:
        return ""
    if not tesseract_nep_available():
        return ""

    value_line = field.details.get("value_ocr_line") or field.details.get("ocr_line")
    if not isinstance(value_line, dict):
        return ""

    bounds = value_line.get("bounds") or []
    if len(bounds) != 4:
        return ""

    cache_dir = settings.CACHE_DIR / "tesseract"
    cache_dir.mkdir(parents=True, exist_ok=True)

    best_text = ""
    for pad_y in (14, 16, 10, 18):
        crop = crop_bounds_with_padding(image, bounds, pad_x=36, pad_y=pad_y)
        if crop.size == 0:
            continue

        crop_path = cache_dir / f"{uuid.uuid4().hex}.jpg"

        try:
            cv2.imwrite(str(crop_path), crop)
            completed = subprocess.run(
                [
                    "tesseract",
                    str(crop_path),
                    "stdout",
                    "-l",
                    settings.TESSERACT_LANG,
                    "--psm",
                    "7",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "TESSDATA_PREFIX": str(settings.CACHE_DIR / "tessdata"),
                },
                timeout=5,
            )
        except Exception:
            logger.debug("Tesseract repair failed", exc_info=True)
            continue
        finally:
            try:
                crop_path.unlink(missing_ok=True)
            except OSError:
                pass

        if completed.returncode != 0:
            logger.debug("Tesseract repair stderr: %s", completed.stderr.strip())
            continue

        text = clean_ocr_line(completed.stdout)
        if first_significant_devanagari_token(text):
            return text
        if text and not best_text:
            best_text = text

    return best_text


def tesseract_nep_available() -> bool:
    tessdata = settings.CACHE_DIR / "tessdata" / f"{settings.TESSERACT_LANG}.traineddata"
    return bool(shutil.which("tesseract") and tessdata.exists())


def crop_bounds_with_padding(
    image: np.ndarray,
    bounds: list[Any],
    pad_x: int,
    pad_y: int,
) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(float(value))) for value in bounds]
    height, width = image.shape[:2]
    x1 = max(x1 - pad_x, 0)
    y1 = max(y1 - pad_y, 0)
    x2 = min(x2 + pad_x, width)
    y2 = min(y2 + pad_y, height)
    return image[y1:y2, x1:x2]


def first_significant_devanagari_token(text: str) -> str:
    tokens = re.findall(r"[\u0900-\u097F]+", text)
    candidates: list[str] = []
    insignificant = {
        normalized_for_match(token)
        for token in extraction_rules().insignificant_tokens
    }
    for token in tokens:
        if re.search(r"\d", token):
            break
        if normalized_for_match(token) in insignificant:
            continue
        if len(token) >= 3:
            candidates.append(token)

    if not candidates:
        return ""

    return max(candidates[:3], key=len)


def refine_date_of_birth(
    fields: dict[str, OCRField],
    lines: list[dict[str, Any]],
    field_name: str,
    field_config: ProfileFieldConfig,
) -> None:
    label_line = date_of_birth_label_line(lines, field_config)
    if not label_line:
        return

    row_candidates = date_of_birth_row_candidates(lines, label_line)
    if not row_candidates:
        return

    components = date_of_birth_components(row_candidates, field_config)
    if not {"year", "month", "day"}.issubset(components):
        return

    year = int(components["year"]["digits"])
    month = int(components["month"]["digits"])
    day = int(components["day"]["digits"])
    if not (1900 <= year <= 2099 and 1 <= month <= 12 and 1 <= day <= 32):
        return

    value = f"{year:04d}-{month:02d}-{day:02d}"
    existing = fields.get(field_name)
    if existing and is_complete_iso_date(existing.value):
        return

    used_components = [components["year"], components["month"], components["day"]]
    confidence = sum(component["confidence"] for component in used_components) / 3
    fields[field_name] = OCRField(
        value=value,
        confidence=round(float(confidence), 4),
        source_text=" ".join(component["text"] for component in used_components),
        raw_value=" ".join(component["text"] for component in used_components),
        normalized_value=value,
        evidence=[
            ocr_line_detail(component["item"], component["index"])
            for component in used_components
        ],
        details={
            "method": "dob_row_saal_mahina_gate",
            "label_ocr_line": ocr_line_detail(label_line[1], label_line[0]),
            "components": {
                name: {
                    "digits": component["digits"],
                    "ocr_line": ocr_line_detail(component["item"], component["index"]),
                    "repair": component["repair"],
                }
                for name, component in components.items()
            },
            "row_candidates": [
                ocr_line_detail(candidate["item"], candidate["index"])
                for candidate in row_candidates
            ],
        },
    )


def date_of_birth_label_line(
    lines: list[dict[str, Any]],
    field_config: ProfileFieldConfig,
) -> Optional[tuple[int, dict[str, Any]]]:
    labels = field_config.labels
    if not labels:
        return None
    for index, item in enumerate(lines):
        if has_label(clean_ocr_line(item.get("text", "")), labels):
            return index, item

    return None


def date_of_birth_row_candidates(
    lines: list[dict[str, Any]],
    label_line: tuple[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    label_index, label_item = label_line
    label_bounds = box_bounds(label_item.get("box") or [])
    if not label_bounds:
        return []

    lx1, ly1, lx2, ly2 = label_bounds
    label_mid_y = (ly1 + ly2) / 2
    label_height = max(ly2 - ly1, 1)
    row_tolerance = max(label_height * 1.1, 55)

    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(lines):
        bounds = box_bounds(item.get("box") or [])
        if not bounds:
            continue

        ox1, oy1, ox2, oy2 = bounds
        mid_y = (oy1 + oy2) / 2
        if abs(mid_y - label_mid_y) > row_tolerance:
            continue
        if index != label_index and ox2 < lx2:
            continue
        if ox1 > lx2 + 850:
            continue

        text = clean_ocr_line(item.get("text", ""))
        if not text:
            continue

        candidates.append(
            {
                "index": index,
                "item": item,
                "text": text,
                "confidence": float(item.get("confidence", 0.0)),
                "bounds": bounds,
            }
        )

    candidates.sort(key=lambda candidate: candidate["bounds"][0])
    return candidates


def date_of_birth_components(
    candidates: list[dict[str, Any]],
    field_config: ProfileFieldConfig,
) -> dict[str, dict[str, Any]]:
    components: dict[str, dict[str, Any]] = {}
    rules = extraction_rules()

    if rules.date_component_pattern:
        for candidate in candidates:
            ascii_text = translate_configured_digits(candidate["text"])
            match = re.search(rules.date_component_pattern, ascii_text)
            if match and len(match.groups()) >= 3:
                year, month, day = match.groups()[:3]
                return {
                    "year": {**candidate, "digits": year, "repair": "same_line_component"},
                    "month": {**candidate, "digits": month, "repair": "same_line_component"},
                    "day": {**candidate, "digits": day, "repair": "same_line_component"},
                }

    for candidate in candidates:
        parsed = parse_dob_digits(
            component_text(candidate["text"], "year", field_config),
            role="year",
        )
        if parsed and len(parsed["digits"]) == 4 and "year" not in components:
            components["year"] = {**candidate, **parsed}
            break

    year_right = components["year"]["bounds"][2] if "year" in components else -1

    for candidate in candidates:
        if candidate["bounds"][0] <= year_right:
            continue
        parsed = parse_dob_digits(
            component_text(candidate["text"], "month", field_config),
            role="month",
        )
        if not parsed:
            continue
        month = int(parsed["digits"])
        if 1 <= month <= 12:
            components["month"] = {**candidate, **parsed}
            break

    month_right = components["month"]["bounds"][2] if "month" in components else year_right

    for candidate in candidates:
        if candidate["bounds"][0] <= month_right:
            continue
        parsed = parse_dob_digits(
            component_text(candidate["text"], "day", field_config),
            role="day",
        )
        if not parsed:
            continue
        day = int(parsed["digits"])
        if 1 <= day <= 32:
            components["day"] = {**candidate, **parsed}
            break

    return components


def component_text(text: str, role: str, field_config: ProfileFieldConfig) -> str:
    components = field_config.components
    if not components:
        return text

    role_indexes = {"year": 0, "month": 1, "day": 2}
    component_index = role_indexes.get(role)
    if component_index is None or component_index >= len(components):
        return text

    label = components[component_index]
    match = re.search(re.escape(label), text, flags=re.IGNORECASE)
    if not match:
        return text

    segment = text[match.end():]
    for next_label in components[component_index + 1:]:
        next_match = re.search(re.escape(next_label), segment, flags=re.IGNORECASE)
        if next_match:
            segment = segment[:next_match.start()]
            break

    return segment.strip(" :.-–—")


def parse_dob_digits(text: str, role: str) -> Optional[dict[str, str]]:
    compact = re.sub(r"\s+", "", translate_configured_digits(text))
    for token in extraction_rules().compact_remove_tokens:
        compact = compact.replace(token, "")

    replacements = {
        **extraction_rules().digit_shape_replacements,
        **extraction_rules().digit_shape_replacements_by_role.get(role, {}),
    }

    if re.search(r"\d", compact):
        repaired = compact.translate(str.maketrans(replacements))
        digits = re.sub(r"\D", "", repaired)
        if digits:
            if role == "year" and len(digits) > 4:
                digits = digits[:4]
            if role == "year" and len(digits) == 3 and digits.startswith("9"):
                digits = f"1{digits}"
            return {"digits": digits, "repair": "latin_digit_shape_in_numeric_token"}

    lower = compact.lower()
    token_pattern = extraction_rules().digit_shape_token_patterns_by_role.get(role, "")
    if lower != compact:
        if token_pattern and not re.fullmatch(token_pattern, lower):
            return None
        repaired = lower.translate(str.maketrans(replacements))
        digits = re.sub(r"\D", "", repaired)
        if digits:
            return {"digits": digits, "repair": "latin_digit_shape_token"}

    if token_pattern and not re.fullmatch(token_pattern, compact):
        return None

    repaired = compact.translate(str.maketrans(replacements))
    digits = re.sub(r"\D", "", repaired)
    if digits and digits != re.sub(r"\D", "", compact):
        return {"digits": digits, "repair": "digit_shape_token"}

    return None


def is_complete_iso_date(value: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()))


def refine_citizenship_number(
    fields: dict[str, OCRField],
    lines: list[dict[str, Any]],
    field_name: str,
    field_config: ProfileFieldConfig,
) -> None:
    existing = fields.get(field_name)
    existing_value = existing.value if existing else ""
    line_candidate = best_citizenship_number_line_candidate(lines, field_config, existing_value)
    if line_candidate and (
        not existing_value or is_richer_citizenship_number(line_candidate["value"], existing_value)
    ):
        fields[field_name] = OCRField(
            value=line_candidate["value"],
            confidence=round(float(line_candidate["confidence"]), 4),
            source_text=line_candidate["text"],
            raw_value=line_candidate["text"],
            normalized_value=line_candidate["value"],
            requires_review=True,
            evidence=[ocr_line_detail(line_candidate["item"], line_candidate["index"])],
            details={
                "method": "multi_pass_citizenship_number_consensus",
                "requires_review": True,
                "review_reason": "conflicting_ocr_number_candidates",
                "alternatives": line_candidate["alternatives"],
                "repair_reason": line_candidate["repair_reason"],
                "line_candidate": ocr_line_detail(line_candidate["item"], line_candidate["index"]),
                "previous_value": existing_value,
            },
        )
        return
    if existing_value and re.fullmatch(r"\d{5,6}-\d{3,6}", existing_value):
        return

    candidates = citizenship_number_candidates(lines)
    if not candidates:
        return

    label_line = citizenship_number_label_line(lines, field_config)
    candidate_groups = group_number_candidates(candidates, label_line)
    if not candidate_groups:
        return

    best_group = candidate_groups[0]
    if label_line:
        label_bounds = box_bounds(label_line[1].get("box") or [])
        if label_bounds:
            label_mid_y = (label_bounds[1] + label_bounds[3]) / 2
            candidate_groups.sort(
                key=lambda group: (
                    abs(group_mid_y(group) - label_mid_y),
                    -len(group),
                    group[0]["bounds"][0],
                )
            )
            best_group = candidate_groups[0]

    fragments = [candidate["token"] for candidate in best_group]
    combined = format_citizenship_number_fragments(fragments)
    if not combined:
        return

    existing = fields.get(field_name)
    existing_value = existing.value if existing else ""
    line_candidate = best_citizenship_number_line_candidate(lines, field_config, existing_value or combined)
    if line_candidate and is_richer_citizenship_number(line_candidate["value"], combined):
        combined = line_candidate["value"]
    elif existing_value and not is_richer_citizenship_number(combined, existing_value):
        if line_candidate and is_richer_citizenship_number(line_candidate["value"], existing_value):
            combined = line_candidate["value"]
        else:
            return

    confidence = sum(candidate["confidence"] for candidate in best_group) / len(best_group)
    source_text = " ".join(candidate["text"] for candidate in best_group)
    details: dict[str, Any] = {
        "method": "same_row_numeric_fragments",
        "fragments": [
            ocr_line_detail(candidate["item"], candidate["index"])
            for candidate in best_group
        ],
        "formatted_fragments": [
            format_citizenship_number_fragment(fragment) for fragment in fragments
        ],
        "candidate_count": len(candidates),
    }
    alternatives = citizenship_number_alternatives(fragments)
    if alternatives:
        previous_combined = combined
        combined = alternatives[0]
        details["requires_review"] = True
        details["previous_value"] = previous_combined
        details["alternatives"] = alternatives
        details["repair_reason"] = "single_digit_shape_confusion"
    if label_line:
        details["label_ocr_line"] = ocr_line_detail(label_line[1], label_line[0])

    if line_candidate and line_candidate["value"] == combined:
        details["requires_review"] = True
        details["previous_value"] = existing_value or details.get("previous_value", "")
        details["alternatives"] = line_candidate["alternatives"]
        details["repair_reason"] = line_candidate["repair_reason"]
        details["line_candidate"] = ocr_line_detail(line_candidate["item"], line_candidate["index"])
        confidence = min(confidence, line_candidate["confidence"])
        source_text = line_candidate["text"]

    fields[field_name] = OCRField(
        value=combined,
        confidence=round(float(confidence), 4),
        source_text=source_text,
        raw_value=source_text,
        normalized_value=combined,
        requires_review=bool(details.get("requires_review", False)),
        evidence=details_to_evidence(details),
        details=details,
    )


def best_citizenship_number_line_candidate(
    lines: list[dict[str, Any]],
    field_config: ProfileFieldConfig,
    existing_value: str = "",
) -> Optional[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(lines):
        if not item_allowed_for_field(item, field_config):
            continue
        text = clean_ocr_line(item.get("text", ""))
        if not text:
            continue

        value = citizenship_number_from_text_line(text)
        if not value:
            if not has_label(text, field_config.labels) and not re.search(r"\d{4,}", text):
                continue
            continue
        candidates.append(
            {
                "index": index,
                "item": item,
                "text": text,
                "value": value,
                "confidence": float(item.get("confidence", 0.0)),
            }
        )

    if existing_value:
        existing_normalized = normalize_citizenship_number_candidate(existing_value)
        if re.fullmatch(r"\d{5,6}-\d{3,6}", existing_normalized):
            existing_prefix = existing_normalized.split("-", 1)[0]
            candidates = [
                candidate
                for candidate in candidates
                if candidate["value"].split("-", 1)[0] == existing_prefix
            ]
        candidates.append(
            {
                "index": -1,
                "item": {"text": existing_value, "confidence": 1.0, "box": []},
                "text": existing_value,
                "value": existing_normalized,
                "confidence": 1.0,
            }
        )

    candidates = [candidate for candidate in candidates if candidate["value"]]
    if not candidates:
        return None

    candidates.sort(key=lambda candidate: float(candidate["confidence"]), reverse=True)
    merged = merged_citizenship_number_candidate(candidates)
    if merged:
        return merged

    candidates.sort(
        key=lambda candidate: (
            -len(re.sub(r"\D", "", candidate["value"])),
            -float(candidate["confidence"]),
        )
    )
    best = candidates[0]
    best["alternatives"] = [candidate["value"] for candidate in candidates[:5]]
    best["repair_reason"] = "best_complete_citizenship_number_candidate"
    return best


def citizenship_number_from_text_line(text: str) -> str:
    cleaned = translate_configured_digits(text)
    cleaned = cleaned.translate(
        str.maketrans(
            {
                "O": "0",
                "o": "0",
                "S": "8",
                "s": "8",
                "%": "8",
                "x": "8",
                "X": "8",
                "A": "1",
                "g": "9",
                "p": "8",
            }
        )
    )
    match = re.search(r"([0-9]{5,6})\s*[-–—]\s*([0-9^]{3,6})", cleaned)
    if not match:
        return ""

    left, right = match.groups()
    right = right.replace("^", "")
    if len(right) < 3:
        return ""
    return f"{left}-{right}"


def normalize_citizenship_number_candidate(value: str) -> str:
    normalized = citizenship_number_from_text_line(value)
    if normalized:
        return normalized
    value = translate_configured_digits(value)
    value = re.sub(r"\s+", "", value)
    return value


def merged_citizenship_number_candidate(
    candidates: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    for shorter in candidates:
        for longer in candidates:
            merged = merge_citizenship_number_values(shorter["value"], longer["value"])
            if not merged:
                continue
            confidence = min(float(shorter["confidence"]), float(longer["confidence"]))
            return {
                "index": longer["index"],
                "item": longer["item"],
                "text": f"{shorter['text']} | {longer['text']}",
                "value": merged,
                "confidence": confidence,
                "alternatives": list(
                    dict.fromkeys([shorter["value"], longer["value"], *[candidate["value"] for candidate in candidates]])
                )[:8],
                "repair_reason": "merged_short_and_long_citizenship_number_candidates",
            }
    return None


def merge_citizenship_number_values(first: str, second: str) -> str:
    first_match = re.fullmatch(r"(\d{5,6})-(\d{3,6})", first)
    second_match = re.fullmatch(r"(\d{5,6})-(\d{3,6})", second)
    if not first_match or not second_match:
        return ""

    left_a, right_a = first_match.groups()
    left_b, right_b = second_match.groups()
    if left_a != left_b:
        return ""

    shorter, longer = (right_a, right_b) if len(right_a) < len(right_b) else (right_b, right_a)
    if len(longer) != len(shorter) + 1:
        return ""

    for index in range(len(longer)):
        if longer[:index] + longer[index + 1:] == shorter:
            return f"{left_a}-{longer}"

    if len(shorter) >= 4 and shorter[:2] == longer[:2] and shorter[-1] == longer[-1]:
        return f"{left_a}-{shorter[:-1]}{longer[-2:]}"

    return ""


def citizenship_number_candidates(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(lines):
        text = clean_ocr_line(item.get("text", ""))
        token = citizenship_number_token(text)
        if not token:
            continue

        bounds = box_bounds(item.get("box") or [])
        if not bounds:
            continue

        candidates.append(
            {
                "index": index,
                "item": item,
                "text": text,
                "token": token,
                "confidence": float(item.get("confidence", 0.0)),
                "bounds": bounds,
            }
        )

    return candidates


def citizenship_number_token(text: str) -> str:
    ascii_text = translate_configured_digits(text)
    if has_document_label_word(ascii_text):
        return ""

    if re.fullmatch(r"\d{3,4}[./\-]\d{1,2}[./\-]\d{1,2}", ascii_text):
        return ""

    compact = re.sub(r"\s+", "", ascii_text)
    matches = re.findall(r"[A-Za-z\u0900-\u097F]*\d[\d/\\.\-]*[A-Za-z\u0900-\u097F]*", compact)
    if not matches:
        return ""

    token = max(matches, key=len)
    token = re.sub(r"^[^\d]+", "", token)
    token = re.sub(r"[^\dA-Za-z\u0900-\u097F/\\.\-]+", "", token)
    digits = re.sub(r"\D", "", token)
    if len(digits) < 2:
        return ""

    return token


def has_document_label_word(text: str) -> bool:
    normalized = normalized_for_match(text)
    label_fragments = extraction_rules().citizenship_label_fragments
    return any(fragment in normalized for fragment in label_fragments)


def citizenship_number_label_line(
    lines: list[dict[str, Any]],
    field_config: ProfileFieldConfig,
) -> Optional[tuple[int, dict[str, Any]]]:
    labels = field_config.labels
    if not labels:
        return None
    for index, item in enumerate(lines):
        if has_label(clean_ocr_line(item.get("text", "")), labels):
            return index, item

    return None


def group_number_candidates(
    candidates: list[dict[str, Any]],
    label_line: Optional[tuple[int, dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    if label_line:
        label_bounds = box_bounds(label_line[1].get("box") or [])
        if label_bounds:
            lx1, ly1, lx2, ly2 = label_bounds
            label_mid_y = (ly1 + ly2) / 2
            label_height = max(ly2 - ly1, 1)
            row_tolerance = max(label_height * 1.4, 70)
            nearby = [
                candidate
                for candidate in candidates
                if abs(candidate_mid_y(candidate) - label_mid_y) <= row_tolerance
                and candidate["bounds"][2] >= lx1
                and candidate["bounds"][0] <= lx2 + 700
            ]
            groups = group_candidates_by_row(nearby)
            if groups:
                return groups

    return group_candidates_by_row(candidates)


def group_candidates_by_row(candidates: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    row_tolerance = 70

    for candidate in sorted(candidates, key=lambda item: candidate_mid_y(item)):
        for group in groups:
            if abs(candidate_mid_y(candidate) - group_mid_y(group)) <= row_tolerance:
                group.append(candidate)
                break
        else:
            groups.append([candidate])

    for group in groups:
        group.sort(key=lambda item: item["bounds"][0])

    groups.sort(
        key=lambda group: (
            -len(group),
            -sum(len(re.sub(r"\D", "", item["token"])) for item in group),
            group[0]["bounds"][1],
        )
    )
    return groups


def candidate_mid_y(candidate: dict[str, Any]) -> float:
    bounds = candidate["bounds"]
    return (bounds[1] + bounds[3]) / 2


def group_mid_y(group: list[dict[str, Any]]) -> float:
    return sum(candidate_mid_y(candidate) for candidate in group) / len(group)


def format_citizenship_number_fragments(fragments: list[str]) -> str:
    formatted = [
        format_citizenship_number_fragment(fragment)
        for fragment in fragments
        if fragment.strip()
    ]
    formatted = [fragment for fragment in formatted if fragment]
    return "/".join(formatted)


def format_citizenship_number_fragment(fragment: str) -> str:
    fragment = translate_configured_digits(fragment)
    fragment = fragment.replace("\\", "/")
    fragment = re.sub(r"\s+", "", fragment)
    fragment = re.sub(r"[^0-9A-Za-z\u0900-\u097F/.\-]", "", fragment)

    fragment = re.sub(r"^(\d+)([\u0900-\u097F]+)$", r"\1 \2", fragment)
    return fragment.strip(" /.-")


def citizenship_number_alternatives(fragments: list[str]) -> list[str]:
    formatted = [
        format_citizenship_number_fragment(fragment)
        for fragment in fragments
        if fragment.strip()
    ]
    alternatives: list[str] = []

    for index, fragment in enumerate(formatted):
        match = re.fullmatch(r"(\d*?)([35])(\s*[\u0900-\u097F]+)", fragment)
        if not match:
            continue

        prefix, digit, suffix = match.groups()
        replacements = extraction_rules().citizenship_digit_confusions
        for replacement in replacements.get(digit, []):
            candidate_fragments = formatted[:]
            candidate_fragments[index] = f"{prefix}{replacement}{suffix}"
            candidate = "/".join(candidate_fragments)
            if candidate not in alternatives:
                alternatives.append(candidate)

    return alternatives


def is_richer_citizenship_number(candidate: str, existing: str) -> bool:
    candidate_digits = re.sub(r"\D", "", candidate)
    existing_digits = re.sub(r"\D", "", existing)
    if len(candidate_digits) != len(existing_digits):
        return len(candidate_digits) > len(existing_digits)

    candidate_separators = len(re.findall(r"[/.\-\s]", candidate))
    existing_separators = len(re.findall(r"[/.\-\s]", existing))
    return candidate_separators > existing_separators


def infer_unlabeled_fields(
    fields: dict[str, OCRField],
    lines: list[dict[str, Any]],
    configured_fields: dict[str, ProfileFieldConfig],
) -> None:
    inferable_fields = {
        field_name
        for field_name, field_config in configured_fields.items()
        if "infer_unlabeled" in field_config.strategies
    }
    for index, item in enumerate(lines):
        text = clean_ocr_line(item["text"])
        confidence = float(item.get("confidence", 0.0))
        ascii_text = translate_configured_digits(text)

        for field_name in inferable_fields:
            if field_name in fields:
                continue
            pattern = extraction_rules().infer_patterns.get(field_name, "")
            match = re.search(pattern, ascii_text) if pattern else None
            if match:
                add_field(
                    fields,
                    field_name,
                    match.group(0),
                    confidence,
                    text,
                    {
                        "method": "infer_unlabeled",
                        "ocr_line": ocr_line_detail(item, index),
                        "pattern": pattern,
                    },
                    configured_fields[field_name],
                )


def save_debug_image(request_id: str, image: np.ndarray) -> None:
    if not settings.SAVE_DEBUG:
        return

    settings.DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    path = settings.DEBUG_DIR / f"{request_id}.jpg"
    cv2.imwrite(str(path), image)


# ----------------------------
# FastAPI App
# ----------------------------

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/admin/reload")
def admin_reload() -> dict[str, Any]:
    global _profiles_cache
    _profiles_cache = None
    _location_gazetteer_cache.clear()
    profiles = load_document_profiles()
    return {
        "status": "reloaded",
        "document_types": sorted(profiles.document_types.keys()),
    }


@app.post("/admin/validate/document-type")
async def admin_validate_document_type(request: Request) -> JSONResponse:
    content = (await request.body()).decode("utf-8")
    try:
        data = yaml.safe_load(content) or {}
        if not isinstance(data, dict):
            raise ValueError("Document type YAML must be a mapping")
        if isinstance(data.get("document_types"), dict):
            for profile_data in data["document_types"].values():
                DocumentProfile.model_validate(profile_data or {})
        else:
            document_type = data.get("document_type") or data.get("id")
            if not isinstance(document_type, str) or not document_type.strip():
                raise ValueError("document_type or id is required")
            profile_data = {
                key: value
                for key, value in data.items()
                if key not in {"document_type", "id"}
            }
            DocumentProfile.model_validate(profile_data)
    except Exception as exc:
        return JSONResponse(
            status_code=422,
            content={"valid": False, "error": str(exc)},
        )
    return JSONResponse({"valid": True})


@app.post("/admin/validate/profiles")
async def admin_validate_profiles(request: Request) -> JSONResponse:
    content = (await request.body()).decode("utf-8")
    try:
        data = yaml.safe_load(content) or {}
        if not isinstance(data, dict):
            raise ValueError("Profiles YAML must be a mapping")
        DocumentProfilesConfig.model_validate(data)
    except Exception as exc:
        return JSONResponse(
            status_code=422,
            content={"valid": False, "error": str(exc)},
        )
    return JSONResponse({"valid": True})


@app.post("/admin/parse/document-type")
async def admin_parse_document_type(request: Request) -> JSONResponse:
    content = (await request.body()).decode("utf-8")
    try:
        data = yaml.safe_load(content) or {}
        if not isinstance(data, dict):
            raise ValueError("Document type YAML must be a mapping")
        document_type = data.get("document_type") or data.get("id") or ""
        profile_data = {
            key: value
            for key, value in data.items()
            if key not in {"document_type", "id"}
        }
        profile = DocumentProfile.model_validate(profile_data)
        return JSONResponse(
            {
                "document_type": document_type,
                "profile": profile.model_dump(exclude_none=True),
            }
        )
    except Exception as exc:
        return JSONResponse(
            status_code=422,
            content={"valid": False, "error": str(exc)},
        )


@app.post("/admin/render/document-type")
async def admin_render_document_type(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        document_type = str(payload.get("document_type") or "").strip()
        if not document_type:
            raise ValueError("document_type is required")
        profile_data = payload.get("profile") or {}
        profile = DocumentProfile.model_validate(profile_data)
        data = {
            "document_type": document_type,
            **profile.model_dump(exclude_none=True),
        }
        content = yaml.safe_dump(
            data,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
        return JSONResponse({"content": content})
    except Exception as exc:
        return JSONResponse(
            status_code=422,
            content={"valid": False, "error": str(exc)},
        )


@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    lang: Optional[str] = Query(None),
    document_type: Optional[str] = Query(None),
    detect_objects: bool = Query(True),
    upscale: bool = Query(True),
    denoise: bool = Query(False),
    threshold: bool = Query(False),
    crop_border: bool = Query(True),
    enhance: bool = Query(True),
    clean_background: bool = Query(settings.CLEAN_BACKGROUND),
    accuracy_mode: str = Query("accurate"),
    retry: bool = Query(True),
    values_only: bool = Query(True),
    fields_only: bool = Query(False),
    include_stats: bool = Query(False),
) -> JSONResponse:
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    resource_started = resource_snapshot()

    data = await file.read()
    validate_upload(file, data)

    image_rgb = load_image(data)
    original_h, original_w = image_rgb.shape[:2]

    try:
        fallback_preprocessing = PreprocessingConfig(
            upscale=upscale,
            denoise=denoise,
            threshold=threshold,
            crop_border=crop_border,
            enhance=enhance,
            clean_background=clean_background,
            max_image_side=1400,
        )
        (
            extraction_items,
            items,
            resolved_document_type,
            profile,
            document_type_confidence,
            processed,
            runtime_meta,
        ) = run_profile_ocr(
            image_rgb,
            document_type=document_type,
            lang=lang,
            fallback_preprocessing=fallback_preprocessing,
            accuracy_mode=accuracy_mode,
            retry=retry,
        )
    except Exception as exc:
        logger.exception("OCR failed")
        raise HTTPException(status_code=500, detail="OCR processing failed") from exc

    save_debug_image(request_id, processed)
    engine_lang = lang or settings.OCR_LANG
    fields = extract_structured_fields(extraction_items, processed, profile)
    values = field_values(fields)
    objects: list[dict[str, Any]] = []
    object_summary = empty_object_summary()
    if detect_objects:
        try:
            objects, object_summary = ObjectDetectionService().detect(image_rgb, items)
        except Exception:
            logger.exception("Object detection failed")
            object_summary = empty_object_summary()
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    resource_usage = resource_delta(resource_started)
    runtime_meta["device"] = ocr_device()
    runtime_meta["gpu"] = ocr_uses_gpu()
    runtime_meta["document_type"] = resolved_document_type
    runtime_meta["document_type_confidence"] = document_type_confidence
    runtime_meta["resource_usage"] = resource_usage

    if values_only:
        meta = {
            "document_type": resolved_document_type,
            "document_type_confidence": document_type_confidence,
        }
        if detect_objects:
            meta["object_summary"] = object_summary
        if include_stats:
            meta.update(
                {
                    "device": ocr_device(),
                    "gpu": ocr_uses_gpu(),
                    "processing_ms": elapsed_ms,
                    "resource_usage": resource_usage,
                }
            )
        return JSONResponse(
            {
                "document_type": resolved_document_type,
                "values": values,
                "objects": objects,
                "object_summary": object_summary,
                "meta": meta,
            }
        )

    if fields_only:
        return JSONResponse(
            {
                "request_id": request_id,
                "filename": file.filename or "unknown",
                "lang": engine_lang,
                "document_type": resolved_document_type,
                "document_type_confidence": document_type_confidence,
                "values": values,
                "fields": {
                    name: field.model_dump()
                    for name, field in fields.items()
                },
                "objects": objects,
                "object_summary": object_summary,
                "meta": runtime_meta,
            }
        )

    response = OCRResponse(
        request_id=request_id,
        filename=file.filename or "unknown",
        mime_type=file.content_type or "unknown",
        file_size_bytes=len(data),
        width=original_w,
        height=original_h,
        processing_ms=elapsed_ms,
        document_type=resolved_document_type,
        document_type_confidence=document_type_confidence,
        full_text=normalize_text(items),
        values=values,
        fields=fields,
        items=items,
        objects=objects,
        object_summary=object_summary,
        meta={
            "engine": "paddleocr",
            "lang": engine_lang,
            "device": ocr_device(),
            "document_type": resolved_document_type,
            "document_type_confidence": document_type_confidence,
            "gpu": ocr_uses_gpu(),
            "min_confidence": settings.MIN_CONFIDENCE,
            "internal_min_confidence": settings.INTERNAL_MIN_CONFIDENCE,
            "object_summary": object_summary,
            **runtime_meta,
            "preprocessing": {
                "upscale": upscale,
                "denoise": denoise,
                "threshold": threshold,
                "crop_border": crop_border,
                "enhance": enhance,
                "clean_background": clean_background,
            },
        },
    )

    return JSONResponse(response.model_dump())


# ----------------------------
# CLI Mode
# ----------------------------

def run_cli(
    image_path: str,
    lang: Optional[str] = None,
    document_type: Optional[str] = None,
    upscale: bool = True,
    denoise: bool = False,
    threshold: bool = False,
    crop_border: bool = True,
    enhance: bool = True,
    clean_background: bool = settings.CLEAN_BACKGROUND,
    json_output: bool = False,
    fields_only: bool = False,
    values_only: bool = False,
    accuracy_mode: str = "accurate",
    retry: bool = True,
) -> None:
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {image_path}")

    data = path.read_bytes()
    image_rgb = load_image(data)
    fallback_preprocessing = PreprocessingConfig(
        upscale=upscale,
        denoise=denoise,
        threshold=threshold,
        crop_border=crop_border,
        enhance=enhance,
        clean_background=clean_background,
        max_image_side=1400,
    )

    structured_output = json_output or fields_only or values_only

    if structured_output:
        with redirect_stdout(sys.stderr):
            (
                extraction_items,
                items,
                resolved_document_type,
                profile,
                document_type_confidence,
                processed,
                runtime_meta,
            ) = run_profile_ocr(
                image_rgb,
                document_type=document_type,
                lang=lang,
                fallback_preprocessing=fallback_preprocessing,
                accuracy_mode=accuracy_mode,
                retry=retry,
            )
    else:
        (
            extraction_items,
            items,
            resolved_document_type,
            profile,
            document_type_confidence,
            processed,
            runtime_meta,
        ) = run_profile_ocr(
            image_rgb,
            document_type=document_type,
            lang=lang,
            fallback_preprocessing=fallback_preprocessing,
            accuracy_mode=accuracy_mode,
            retry=retry,
        )

    fields = extract_structured_fields(extraction_items, processed, profile)

    if values_only:
        print(json.dumps(field_values(fields), ensure_ascii=False, indent=2))
        return

    if json_output or fields_only:
        payload = {
            "filename": path.name,
            "lang": lang or settings.OCR_LANG,
            "document_type": resolved_document_type,
            "document_type_confidence": document_type_confidence,
            "values": field_values(fields),
            "meta": runtime_meta,
            "fields": {
                name: field.model_dump()
                for name, field in fields.items()
            },
        }
        if not fields_only:
            payload["full_text"] = normalize_text(items)
            payload["items"] = items
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(normalize_text(items))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Production OCR Reader")
    parser.add_argument("--image", help="Run OCR on one image file")
    parser.add_argument("--lang", default=None, help="OCR language code, e.g. ne, en")
    parser.add_argument("--document-type", default=None, help="Document profile id")
    parser.add_argument("--no-upscale", action="store_true")
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--threshold", action="store_true")
    parser.add_argument("--no-crop-border", action="store_true")
    parser.add_argument("--no-enhance", action="store_true")
    parser.add_argument("--clean-background", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fields-only", action="store_true")
    parser.add_argument("--values-only", action="store_true")
    parser.add_argument(
        "--accuracy-mode",
        choices=("fast", "balanced", "accurate"),
        default="accurate",
    )
    parser.add_argument("--no-retry", action="store_true")
    parser.add_argument("--serve", action="store_true", help="Start API server")
    parser.add_argument("--host", default=settings.SERVER_HOST)
    parser.add_argument("--port", default=settings.SERVER_PORT, type=int)
    parser.add_argument("--workers", default=settings.SERVER_WORKERS, type=int)
    parser.add_argument("--log-level", default=settings.SERVER_LOG_LEVEL)
    parser.add_argument("--keep-alive", default=settings.SERVER_KEEP_ALIVE, type=int)
    parser.add_argument(
        "--device",
        default=None,
        help="Paddle device, e.g. cpu, gpu, gpu:0, gpu:1. Overrides OCR_DEVICE.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use gpu:<OCR_GPU_ID> unless --device is set.",
    )
    parser.add_argument("--gpu-id", default=None, help="GPU id for --use-gpu/OCR_USE_GPU")

    args = parser.parse_args()

    if args.gpu_id is not None:
        settings.GPU_ID = str(args.gpu_id)
    if args.device is not None:
        settings.OCR_DEVICE = args.device.strip()
    if args.use_gpu:
        settings.USE_GPU = True

    if args.image:
        run_cli(
            args.image,
            lang=args.lang,
            document_type=args.document_type,
            upscale=not args.no_upscale,
            denoise=args.denoise,
            threshold=args.threshold,
            crop_border=not args.no_crop_border,
            enhance=not args.no_enhance,
            clean_background=args.clean_background,
            json_output=args.json,
            fields_only=args.fields_only,
            values_only=args.values_only,
            accuracy_mode=args.accuracy_mode,
            retry=not args.no_retry,
        )
    elif args.serve:
        import uvicorn

        uvicorn.run(
            "ocr_service:app",
            host=args.host,
            port=args.port,
            workers=max(args.workers, 1),
            log_level=args.log_level,
            proxy_headers=True,
            forwarded_allow_ips="*",
            timeout_keep_alive=args.keep_alive,
        )
    else:
        parser.print_help()
