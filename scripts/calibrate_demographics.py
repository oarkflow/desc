import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from face.attributes import AGE_BUCKETS, GENDER_LABELS, DemographicEstimator
from face.detector import FaceDetection, YuNetDetector
from face.image_loader import load_image


DEFAULT_YUNET_MODEL = "models/face_detection_yunet_2023mar.onnx"
DEFAULT_AGE_MODEL = "models/age_net.caffemodel"
DEFAULT_AGE_PROTO = "models/age_deploy.prototxt"
DEFAULT_GENDER_MODEL = "models/gender_net.caffemodel"
DEFAULT_GENDER_PROTO = "models/gender_deploy.prototxt"


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate age/gender confidence thresholds from a labeled validation CSV."
    )
    parser.add_argument("csv_path", help="CSV with image_path and optional age,gender,bbox columns")
    parser.add_argument("--output", default="models/demographic_calibration.json")
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument("--yunet-model", default=DEFAULT_YUNET_MODEL)
    parser.add_argument("--age-model", default=DEFAULT_AGE_MODEL)
    parser.add_argument("--age-proto", default=DEFAULT_AGE_PROTO)
    parser.add_argument("--gender-model", default=DEFAULT_GENDER_MODEL)
    parser.add_argument("--gender-proto", default=DEFAULT_GENDER_PROTO)
    args = parser.parse_args()

    estimator = DemographicEstimator(
        age_model_path=args.age_model,
        age_proto_path=args.age_proto,
        gender_model_path=args.gender_model,
        gender_proto_path=args.gender_proto,
    )
    if not estimator.available:
        raise SystemExit("Age/gender models are missing. Run scripts/download_models.py first.")

    detector = YuNetDetector(args.yunet_model)
    rows = _read_rows(args.csv_path)
    age_records = []
    gender_records = []
    skipped = []

    for row in rows:
        image_path = row.get("image_path") or row.get("path") or row.get("image")
        if not image_path:
            skipped.append({"row": row, "reason": "missing image_path"})
            continue
        try:
            image = load_image(image_path)
            detection = _detection_from_row(row) or _largest_detection(detector, image)
            if detection is None:
                skipped.append({"image_path": image_path, "reason": "no face detected"})
                continue
            raw = estimator.predict_probabilities(image, detection)
            if raw is None:
                skipped.append({"image_path": image_path, "reason": "empty face crop"})
                continue
        except Exception as exc:
            skipped.append({"image_path": image_path, "reason": str(exc)})
            continue

        age_label = _normalize_label(row.get("age") or row.get("age_bucket"))
        if age_label in AGE_BUCKETS:
            age_records.append((age_label, raw["age"]))

        gender_label = _normalize_gender(row.get("gender") or row.get("apparent_gender"))
        if gender_label in GENDER_LABELS:
            gender_records.append((gender_label, raw["gender"]))

    calibration = {
        "age": _calibrate_section(age_records, AGE_BUCKETS, default_min=0.35, target_precision=args.target_precision),
        "gender": _calibrate_section(gender_records, GENDER_LABELS, default_min=0.65, target_precision=args.target_precision),
        "metadata": {
            "samples": len(rows),
            "age_labeled_samples": len(age_records),
            "gender_labeled_samples": len(gender_records),
            "skipped": skipped,
            "target_precision": args.target_precision,
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(calibration, indent=2) + "\n")
    print(f"Saved demographic calibration to {output}")
    print(json.dumps(calibration["metadata"], indent=2))


def _read_rows(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _detection_from_row(row: dict) -> FaceDetection | None:
    bbox = row.get("bbox")
    if bbox:
        values = [float(part.strip()) for part in bbox.split(",")]
    elif all(row.get(name) for name in ("x", "y", "w", "h")):
        values = [float(row[name]) for name in ("x", "y", "w", "h")]
    else:
        return None
    if len(values) != 4:
        return None
    return FaceDetection(tuple(int(round(v)) for v in values), confidence=1.0, index=0)


def _largest_detection(detector: YuNetDetector, image) -> FaceDetection | None:
    detections = detector.detect(image)
    if not detections:
        return None
    return max(detections, key=lambda item: item.w * item.h)


def _calibrate_section(records: list[tuple[str, np.ndarray]], labels: list[str], default_min: float, target_precision: float) -> dict:
    if not records:
        return {
            "min_confidence": default_min,
            "thresholds": {},
            "weights": {},
            "label_map": {},
            "metrics": {"samples": 0, "accuracy": None},
        }

    weights = _class_weights(records, labels)
    calibrated = [(_normalize_probs(scores, labels, weights), truth) for truth, scores in records]
    thresholds = {
        label: _threshold_for_label(calibrated, labels, label, default_min, target_precision)
        for label in labels
    }
    correct = 0
    for probs, truth in calibrated:
        pred = labels[int(np.argmax(probs))]
        if pred == truth and float(np.max(probs)) >= thresholds.get(pred, default_min):
            correct += 1

    return {
        "min_confidence": default_min,
        "thresholds": thresholds,
        "weights": weights,
        "label_map": {},
        "metrics": {
            "samples": len(records),
            "accepted_accuracy": round(correct / len(records), 4),
            "label_counts": dict(_counts(truth for truth, _ in records)),
        },
    }


def _class_weights(records: list[tuple[str, np.ndarray]], labels: list[str]) -> dict:
    counts = _counts(truth for truth, _ in records)
    total = sum(counts.values())
    if total == 0:
        return {}
    expected = total / len(labels)
    weights = {}
    for label in labels:
        count = counts.get(label, 0)
        if count > 0:
            weights[label] = round(float(np.clip(expected / count, 0.5, 2.0)), 4)
    return weights


def _threshold_for_label(
    records: list[tuple[np.ndarray, str]],
    labels: list[str],
    label: str,
    default_min: float,
    target_precision: float,
) -> float:
    idx = labels.index(label)
    candidates = sorted({default_min, *[round(float(probs[idx]), 3) for probs, _ in records]})
    best = default_min
    best_accepted = -1
    for threshold in candidates:
        accepted = [(probs, truth) for probs, truth in records if labels[int(np.argmax(probs))] == label and probs[idx] >= threshold]
        if not accepted:
            continue
        precision = sum(1 for _, truth in accepted if truth == label) / len(accepted)
        if precision >= target_precision and len(accepted) > best_accepted:
            best = threshold
            best_accepted = len(accepted)
    return round(float(best), 4)


def _normalize_probs(scores: np.ndarray, labels: list[str], weights: dict) -> np.ndarray:
    probs = np.asarray(scores, dtype=float).reshape(-1)[: len(labels)]
    for idx, label in enumerate(labels[: len(probs)]):
        probs[idx] *= float(weights.get(label, 1.0))
    total = float(probs.sum())
    if total <= 0:
        return np.full(len(labels), 1.0 / len(labels), dtype=float)
    return probs / total


def _counts(values) -> defaultdict[str, int]:
    counts = defaultdict(int)
    for value in values:
        counts[value] += 1
    return counts


def _normalize_label(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().replace("(", "").replace(")", "")
    return value if value in AGE_BUCKETS else None


def _normalize_gender(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "male": "male_presenting",
        "man": "male_presenting",
        "m": "male_presenting",
        "male_presenting": "male_presenting",
        "female": "female_presenting",
        "woman": "female_presenting",
        "f": "female_presenting",
        "female_presenting": "female_presenting",
    }
    return aliases.get(value)


if __name__ == "__main__":
    main()
