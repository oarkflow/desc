"""
Face attribute estimates derived from detections and landmarks.

These are local, best-effort signals. Demographic attributes are intentionally
left unknown unless a calibrated model is added, because geometry-only guesses
are not reliable enough for identity or verification workflows.
"""

from __future__ import annotations

import cv2
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .detector import FaceDetection
from .landmarks import LandmarkResult


AGE_BUCKETS = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60+"]
GENDER_LABELS = ["male_presenting", "female_presenting"]
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
DEMOGRAPHIC_CROP_PADDINGS = (0.0, 0.05, 0.10, 0.15)


DEFAULT_DEMOGRAPHIC_CALIBRATION = {
    "age": {
        "min_confidence": 0.35,
        "weights": {},
        "label_map": {},
    },
    "gender": {
        "min_confidence": 0.65,
        "weights": {},
        "label_map": {},
    },
}


@dataclass
class FaceAttributes:
    face_index: int
    facing_direction: str
    pose: Dict[str, Any]
    expression: Dict[str, Any]
    quality: Dict[str, Any]
    estimated_age: Dict[str, Any]
    gender: Dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "face_index": self.face_index,
            "facing_direction": self.facing_direction,
            "pose": self.pose,
            "expression": self.expression,
            "mood": self.expression,
            "quality": self.quality,
            "estimated_age": self.estimated_age,
            "gender": self.gender,
        }


def estimate_face_attributes(
    image: np.ndarray,
    detection: FaceDetection,
    landmark: Optional[LandmarkResult] = None,
    demographics: Optional[Dict[str, Any]] = None,
) -> FaceAttributes:
    yaw = _rounded(landmark.yaw_estimate()) if landmark else None
    roll = _rounded(_roll_estimate(landmark)) if landmark else None
    mouth_open = _rounded(landmark.mouth_open_ratio(), 4) if landmark else None
    facing = _facing_direction(yaw)

    pose = {
        "yaw_estimate_deg": yaw,
        "roll_estimate_deg": roll,
        "facing": facing,
        "is_frontal": yaw is not None and abs(yaw) <= 8,
        "note": "Pose is estimated from facial landmarks and is approximate.",
    }

    return FaceAttributes(
        face_index=detection.index,
        facing_direction=facing,
        pose=pose,
        expression=_expression(landmark, mouth_open),
        quality=_quality(image, detection, landmark),
        estimated_age=(demographics or {}).get("estimated_age", _unknown_age()),
        gender=(demographics or {}).get("gender", _unknown_gender()),
    )


class DemographicEstimator:
    """
    OpenCV DNN age/gender estimator.

    The model returns apparent age bucket and apparent gender presentation. These
    outputs are sensitive attributes, so low-confidence predictions are reported
    as unknown and calibration can raise/lower per-class thresholds.
    """

    def __init__(
        self,
        age_model_path: str,
        age_proto_path: str,
        gender_model_path: str,
        gender_proto_path: str,
        calibration_path: Optional[str] = None,
    ):
        self.age_net = _load_net(age_proto_path, age_model_path, "age")
        self.gender_net = _load_net(gender_proto_path, gender_model_path, "gender")
        self.calibration = load_demographic_calibration(calibration_path)

    @property
    def available(self) -> bool:
        return self.age_net is not None and self.gender_net is not None

    def predict(self, image: np.ndarray, detection: FaceDetection) -> Dict[str, Any]:
        if not self.available:
            return {
                "estimated_age": _unknown_age("Age model is not available."),
                "gender": _unknown_gender("Gender model is not available."),
            }

        raw = self.predict_probabilities(image, detection)
        if raw is None:
            return {
                "estimated_age": _unknown_age("Face crop is empty."),
                "gender": _unknown_gender("Face crop is empty."),
            }

        return {
            "estimated_age": _age_result(raw["age"], self.calibration.get("age", {})),
            "gender": _gender_result(raw["gender"], self.calibration.get("gender", {})),
        }

    def predict_probabilities(self, image: np.ndarray, detection: FaceDetection) -> Optional[Dict[str, np.ndarray]]:
        if not self.available:
            return None

        age_predictions = []
        gender_predictions = []
        for padding in DEMOGRAPHIC_CROP_PADDINGS:
            roi = detection.face_roi(image, padding=padding)
            if roi.size == 0:
                continue

            blob = cv2.dnn.blobFromImage(
                roi,
                scalefactor=1.0,
                size=(227, 227),
                mean=MODEL_MEAN_VALUES,
                swapRB=False,
                crop=False,
            )

            self.age_net.setInput(blob)
            age_predictions.append(self.age_net.forward()[0])
            self.gender_net.setInput(blob)
            gender_predictions.append(self.gender_net.forward()[0])

        if not age_predictions or not gender_predictions:
            return None

        age_scores = np.mean(np.vstack(age_predictions), axis=0)
        gender_scores = np.mean(np.vstack(gender_predictions), axis=0)

        return {
            "age": np.asarray(age_scores, dtype=float),
            "gender": np.asarray(gender_scores, dtype=float),
        }


def load_demographic_calibration(calibration_path: Optional[str]) -> Dict[str, Any]:
    calibration = json.loads(json.dumps(DEFAULT_DEMOGRAPHIC_CALIBRATION))
    if not calibration_path:
        return calibration

    path = Path(calibration_path)
    if not path.exists():
        return calibration

    with path.open() as f:
        user_config = json.load(f)

    for section in ("age", "gender"):
        if isinstance(user_config.get(section), dict):
            calibration[section].update(user_config[section])
    return calibration


def demographic_model_available(
    age_model_path: Optional[str],
    age_proto_path: Optional[str],
    gender_model_path: Optional[str],
    gender_proto_path: Optional[str],
) -> bool:
    return all(path and Path(path).exists() for path in (
        age_model_path,
        age_proto_path,
        gender_model_path,
        gender_proto_path,
    ))


def _facing_direction(yaw: Optional[float]) -> str:
    if yaw is None:
        return "unknown"
    if yaw <= -18:
        return "left"
    if yaw >= 18:
        return "right"
    if yaw <= -8:
        return "slightly_left"
    if yaw >= 8:
        return "slightly_right"
    return "front"


def _load_net(proto_path: str, model_path: str, label: str):
    if not Path(proto_path).exists() or not Path(model_path).exists():
        return None
    try:
        return cv2.dnn.readNet(str(model_path), str(proto_path))
    except cv2.error as exc:
        raise RuntimeError(f"Failed to load {label} demographic model: {exc}") from exc


def _age_result(scores: np.ndarray, calibration: Dict[str, Any]) -> Dict[str, Any]:
    probs = _calibrated_probabilities(scores, AGE_BUCKETS, calibration)
    idx = int(np.argmax(probs))
    band = AGE_BUCKETS[idx]
    confidence = float(probs[idx])
    thresholds = calibration.get("thresholds", {})
    min_confidence = float(thresholds.get(band, calibration.get("min_confidence", 0.35)))
    mapped = calibration.get("label_map", {}).get(band, band)
    if confidence < min_confidence:
        return _unknown_age(
            f"Best age bucket {mapped} confidence {confidence:.3f} is below threshold {min_confidence:.3f}.",
            raw_bucket=mapped,
            raw_confidence=confidence,
        )
    return {
        "band": mapped,
        "confidence": round(confidence, 4),
        "probabilities": _probability_dict(probs, AGE_BUCKETS, calibration.get("label_map", {})),
        "note": "Apparent age bucket from calibrated OpenCV DNN model; not proof of actual age.",
    }


def _gender_result(scores: np.ndarray, calibration: Dict[str, Any]) -> Dict[str, Any]:
    probs = _calibrated_probabilities(scores, GENDER_LABELS, calibration)
    idx = int(np.argmax(probs))
    label = GENDER_LABELS[idx]
    confidence = float(probs[idx])
    thresholds = calibration.get("thresholds", {})
    min_confidence = float(thresholds.get(label, calibration.get("min_confidence", 0.65)))
    mapped = calibration.get("label_map", {}).get(label, label)
    if confidence < min_confidence:
        return _unknown_gender(
            f"Best apparent gender label {mapped} confidence {confidence:.3f} is below threshold {min_confidence:.3f}.",
            raw_label=mapped,
            raw_confidence=confidence,
        )
    return {
        "label": mapped,
        "confidence": round(confidence, 4),
        "probabilities": _probability_dict(probs, GENDER_LABELS, calibration.get("label_map", {})),
        "note": "Apparent gender presentation from calibrated model; not gender identity.",
    }


def _calibrated_probabilities(scores: np.ndarray, labels: list[str], calibration: Dict[str, Any]) -> np.ndarray:
    probs = np.asarray(scores, dtype=float).reshape(-1)
    if probs.size != len(labels):
        probs = probs[: len(labels)]
    weights = calibration.get("weights", {})
    for idx, label in enumerate(labels[: len(probs)]):
        probs[idx] *= float(weights.get(label, 1.0))
    total = float(probs.sum())
    if total <= 0:
        return np.full(len(labels), 1.0 / len(labels), dtype=float)
    return probs / total


def _probability_dict(probs: np.ndarray, labels: list[str], label_map: Dict[str, str]) -> Dict[str, float]:
    return {
        label_map.get(label, label): round(float(probs[idx]), 4)
        for idx, label in enumerate(labels[: len(probs)])
    }


def _unknown_age(
    note: str = "Age model is not configured.",
    raw_bucket: Optional[str] = None,
    raw_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    result = {
        "band": "unknown",
        "confidence": 0.0,
        "note": note,
    }
    if raw_bucket is not None:
        result["raw_bucket"] = raw_bucket
        result["raw_confidence"] = round(float(raw_confidence or 0.0), 4)
    return result


def _unknown_gender(
    note: str = "Gender model is not configured.",
    raw_label: Optional[str] = None,
    raw_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    result = {
        "label": "unknown",
        "confidence": 0.0,
        "note": note,
    }
    if raw_label is not None:
        result["raw_label"] = raw_label
        result["raw_confidence"] = round(float(raw_confidence or 0.0), 4)
    return result


def _expression(landmark: Optional[LandmarkResult], mouth_open: Optional[float]) -> Dict[str, Any]:
    blendshapes = landmark.blendshapes if landmark else {}
    smile = _blend_avg(blendshapes, "mouthSmileLeft", "mouthSmileRight")
    frown = _blend_avg(blendshapes, "mouthFrownLeft", "mouthFrownRight")
    jaw_open = blendshapes.get("jawOpen")
    brow_down = _blend_avg(blendshapes, "browDownLeft", "browDownRight")
    brow_up = _blend_avg(blendshapes, "browOuterUpLeft", "browOuterUpRight")

    label = "neutral"
    confidence = 0.35
    signals = {}

    if smile is not None:
        signals["smile"] = round(smile, 4)
        if smile >= 0.35:
            label = "smiling"
            confidence = min(0.95, 0.45 + smile)

    open_signal = jaw_open if jaw_open is not None else mouth_open
    if open_signal is not None:
        signals["mouth_open"] = round(open_signal, 4)
        if open_signal >= 0.35 or (jaw_open is None and open_signal >= 0.08):
            label = "open_mouth"
            confidence = max(confidence, 0.65)

    if frown is not None:
        signals["frown"] = round(frown, 4)
        if frown >= 0.35 and smile is not None and smile < 0.25:
            label = "frowning"
            confidence = max(confidence, 0.65)

    if brow_down is not None:
        signals["brow_down"] = round(brow_down, 4)
    if brow_up is not None:
        signals["brow_up"] = round(brow_up, 4)

    if mouth_open is not None:
        signals.setdefault("mouth_open_ratio", mouth_open)

    return {
        "label": label,
        "confidence": round(float(confidence), 4),
        "signals": signals,
        "note": "Expression is a visual cue estimate, not a reliable emotion diagnosis.",
    }


def _quality(image: np.ndarray, detection: FaceDetection, landmark: Optional[LandmarkResult]) -> Dict[str, Any]:
    roi = detection.face_roi(image, padding=0.0)
    if roi.size == 0:
        return {"label": "poor", "score": 0.0, "signals": {"reason": "empty_face_crop"}}

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    h, w = image.shape[:2]
    face_area_ratio = float((detection.w * detection.h) / max(w * h, 1))
    landmark_coverage = _landmark_coverage(detection, landmark)

    blur_score = _clamp(blur / 180.0)
    brightness_score = _clamp(1.0 - abs(brightness - 128.0) / 128.0)
    contrast_score = _clamp(contrast / 64.0)
    size_score = _clamp(face_area_ratio / 0.08)
    coverage_score = 1.0 if landmark_coverage is None else landmark_coverage
    score = (
        blur_score * 0.30
        + brightness_score * 0.20
        + contrast_score * 0.15
        + size_score * 0.20
        + coverage_score * 0.15
    )

    label = "good" if score >= 0.72 else "fair" if score >= 0.45 else "poor"
    return {
        "label": label,
        "score": round(score, 4),
        "signals": {
            "sharpness": round(blur, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "face_area_ratio": round(face_area_ratio, 4),
            "landmark_coverage": None if landmark_coverage is None else round(landmark_coverage, 4),
        },
    }


def _roll_estimate(landmark: Optional[LandmarkResult]) -> Optional[float]:
    if not landmark or "right_eye" not in landmark.groups or "left_eye" not in landmark.groups:
        return None
    right = landmark.groups["right_eye"].mean(axis=0)
    left = landmark.groups["left_eye"].mean(axis=0)
    dx = float(left[0] - right[0])
    dy = float(left[1] - right[1])
    if abs(dx) < 1e-6:
        return None
    return float(np.degrees(np.arctan2(dy, dx)))


def _landmark_coverage(detection: FaceDetection, landmark: Optional[LandmarkResult]) -> Optional[float]:
    if landmark is None or landmark.points is None or len(landmark.points) == 0:
        return None
    x, y, w, h = detection.bbox
    pts = landmark.points
    inside = (
        (pts[:, 0] >= x)
        & (pts[:, 0] <= x + w)
        & (pts[:, 1] >= y)
        & (pts[:, 1] <= y + h)
    )
    return float(np.mean(inside))


def _blend_avg(blendshapes: Dict[str, float], left: str, right: str) -> Optional[float]:
    values = [blendshapes[name] for name in (left, right) if name in blendshapes]
    if not values:
        return None
    return float(np.mean(values))


def _rounded(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))
