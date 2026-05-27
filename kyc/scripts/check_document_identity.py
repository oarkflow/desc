import argparse
import json
import tempfile
from pathlib import Path

import cv2
from fastapi.testclient import TestClient

from check_common import KYC_ROOT, default_test_image, existing_path, image_mime, print_json, require


FACE_OBJECT_LABELS = {"face", "photo", "portrait"}
MODEL_DIR = KYC_ROOT / "models"


def object_pixel_box(obj):
    box = obj.get("box") or {}
    pixel = box.get("pixel") or {}
    try:
        x = int(pixel["x"])
        y = int(pixel["y"])
        width = int(pixel["width"])
        height = int(pixel["height"])
    except Exception:
        return None
    if width <= 0 or height <= 0:
        return None
    return x, y, width, height


def crop_document_faces(image_path, objects, output_dir):
    image = cv2.imread(str(image_path))
    require(image is not None, f"Could not read document image: {image_path}")
    height, width = image.shape[:2]
    output_dir.mkdir(parents=True, exist_ok=True)
    crops = []

    for index, obj in enumerate(objects):
        if obj.get("label") not in FACE_OBJECT_LABELS:
            continue
        box = object_pixel_box(obj)
        if not box:
            continue
        x, y, w, h = box
        pad_x = int(w * 0.18)
        pad_y = int(h * 0.18)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_y)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        path = output_dir / f"{Path(image_path).stem}_{obj.get('label')}_{index + 1}.jpg"
        cv2.imwrite(str(path), crop)
        crops.append(
            {
                "object_index": index,
                "label": obj.get("label"),
                "confidence": obj.get("confidence"),
                "box": obj.get("box"),
                "path": str(path),
            }
        )

    return crops


def compare_landmarks(reference_image, crop_paths, limit):
    from kyc.face.cli import (
        LANDMARK_REGION_SPECS,
        landmark_bounds,
        normalized_region_difference,
    )
    from kyc.face.image_loader import load_image
    from kyc.face.landmarks import MediaPipeLandmarkDetector

    model = MODEL_DIR / "face_landmarker.task"
    if not model.exists():
        return {
            "available": False,
            "reason": f"MediaPipe landmark model not found: {model}",
        }

    detector = MediaPipeLandmarkDetector(str(model), num_faces=max(1, limit))
    try:
        ref_image = load_image(str(reference_image))
        ref_landmarks = detector.detect(ref_image)
        if not ref_landmarks:
            return {"available": False, "reason": "No landmarks found in reference face image."}
        ref_landmark = max(ref_landmarks, key=lambda lm: landmark_bounds(lm)[2] * landmark_bounds(lm)[3])

        comparisons = []
        for crop_path in crop_paths[:limit]:
            crop_image = load_image(str(crop_path))
            crop_landmarks = detector.detect(crop_image)
            if not crop_landmarks:
                comparisons.append(
                    {
                        "crop": str(crop_path),
                        "available": False,
                        "reason": "No landmarks found in document face crop.",
                    }
                )
                continue
            crop_landmark = max(crop_landmarks, key=lambda lm: landmark_bounds(lm)[2] * landmark_bounds(lm)[3])
            regions = []
            for spec in LANDMARK_REGION_SPECS:
                diff = normalized_region_difference(ref_landmark, crop_landmark, spec)
                if diff is not None:
                    regions.append(diff)
            regions.sort(key=lambda item: item["distance"])
            comparisons.append(
                {
                    "crop": str(crop_path),
                    "available": True,
                    "reference_landmark_points": len(ref_landmark.points),
                    "document_landmark_points": len(crop_landmark.points),
                    "most_similar": regions[:3],
                    "most_different": list(reversed(regions[-3:])),
                    "note": "Landmark comparison is supporting geometry evidence; SFace cosine decides identity match.",
                }
            )
    finally:
        detector.close()

    return {"available": True, "comparisons": comparisons}


def match_document_faces(reference_image, crops, threshold, include_landmarks):
    from kyc.face.recognizer import SFaceSearcher

    crop_paths = [crop["path"] for crop in crops]
    if not crop_paths:
        return {
            "available": False,
            "reason": "No document face/photo/portrait crops were found to match.",
        }

    searcher = SFaceSearcher(
        yunet_model_path=str(MODEL_DIR / "face_detection_yunet_2023mar.onnx"),
        sface_model_path=str(MODEL_DIR / "face_recognition_sface_2021dec.onnx"),
        cosine_threshold=threshold,
    )
    matches = searcher.search(str(reference_image), crop_paths)
    payload = {
        "available": True,
        "reference_image": str(reference_image),
        "threshold": threshold,
        "matches": [match.to_dict() for match in matches],
        "matched": any(match.is_match for match in matches),
    }
    if include_landmarks:
        payload["landmark_comparison"] = compare_landmarks(reference_image, crop_paths, len(crop_paths))
    return payload


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract document fields, detect document face/photo objects, run document anti-spoofing, "
            "and optionally match the document face against a selfie/reference face."
        )
    )
    parser.add_argument("--image", type=existing_path, default=default_test_image("national-id.webp"))
    parser.add_argument(
        "--document-type",
        default=None,
        help=(
            "Expected document profile id for reporting. The CLI still auto-detects by default; "
            "combine with --force-document-type to bypass auto-detection."
        ),
    )
    parser.add_argument(
        "--force-document-type",
        action="store_true",
        help="Send --document-type to the OCR service as a forced profile override.",
    )
    parser.add_argument("--lang", default=None)
    parser.add_argument("--accuracy-mode", default="fast", choices=["fast", "accurate"])
    parser.add_argument("--retry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--match-face", type=existing_path, help="Selfie/reference image to match against document face/photo.")
    parser.add_argument("--match-threshold", type=float, default=0.363)
    parser.add_argument("--match-landmarks", action="store_true", help="Include 478-point landmark comparison evidence.")
    parser.add_argument(
        "--result-mode",
        choices=("values", "summary", "full"),
        default="values",
        help=(
            "JSON detail level. values returns only detected type and values; "
            "summary adds document face/object/tamper summaries; full adds raw field evidence and OCR text."
        ),
    )
    parser.add_argument("--crop-faces", action="store_true", help="Write detected document face/photo crops.")
    parser.add_argument("--include-fields", action="store_true", help="Include structured field evidence in the JSON result.")
    parser.add_argument("--include-full-text", action="store_true", help="Include full OCR text in the JSON result.")
    parser.add_argument("--include-objects", action="store_true", help="Include raw detected object records in the JSON result.")
    parser.add_argument("--include-tamper", action="store_true", help="Include full tamper result in the JSON result.")
    parser.add_argument("--include-meta", action="store_true", help="Include full OCR runtime metadata in the JSON result.")
    parser.add_argument("--output-dir", type=Path, default=Path("test-results/document-identity"))
    parser.add_argument("--json", type=Path, help="Write complete JSON result to this file.")
    parser.add_argument("--min-values", type=int, default=1)
    args = parser.parse_args()

    from kyc.ocr.service import app

    needs_full_response = (
        args.result_mode != "values"
        or args.match_face is not None
        or args.crop_faces
        or args.include_fields
        or args.include_full_text
        or args.include_objects
        or args.include_tamper
        or args.include_meta
    )
    params = {
        "values_only": str(not needs_full_response).lower(),
        "include_stats": str(args.include_stats).lower(),
        "detect_objects": str(needs_full_response).lower(),
        "accuracy_mode": args.accuracy_mode,
        "retry": str(args.retry).lower(),
    }
    if args.force_document_type and args.document_type:
        params["document_type"] = args.document_type
    if args.lang:
        params["lang"] = args.lang

    client = TestClient(app)
    with Path(args.image).open("rb") as handle:
        response = client.post(
            "/ocr",
            params=params,
            files={"file": (Path(args.image).name, handle, image_mime(args.image))},
        )

    require(response.status_code == 200, f"OCR request failed with HTTP {response.status_code}: {response.text[:500]}")
    payload = response.json()
    values = payload.get("values") or {}
    objects = payload.get("objects") or []
    require(len(values) >= args.min_values, f"OCR extracted {len(values)} values, expected at least {args.min_values}.")

    crops = []
    if args.crop_faces or args.match_face:
        crop_dir = args.output_dir / "document-faces"
        crops = crop_document_faces(args.image, objects, crop_dir)
    match_result = None
    if args.match_face:
        match_result = match_document_faces(args.match_face, crops, args.match_threshold, args.match_landmarks)

    result = {
        "check": "document_identity",
        "status": "pass",
        "image": str(args.image),
        "requested_document_type": args.document_type,
        "forced_document_type": args.document_type if args.force_document_type else None,
        "auto_detected": not args.force_document_type,
        "document_type": payload.get("document_type"),
        "field_count": len(values),
        "values": values,
    }
    if args.result_mode in {"summary", "full"}:
        result["object_summary"] = payload.get("object_summary")
        result["document_face_crops"] = crops
        result["document_face_match"] = match_result
        tamper = payload.get("tamper") or {}
        result["tamper_summary"] = {
            "status": tamper.get("status"),
            "tamper_score": tamper.get("tamper_score"),
            "manual_review_required": tamper.get("manual_review_required"),
            "flag_count": len(tamper.get("flags") or []),
        }
    if args.result_mode == "full" or args.include_fields:
        result["fields"] = payload.get("fields") or {}
    if args.result_mode == "full" or args.include_full_text:
        result["full_text"] = payload.get("full_text", "")
    if args.result_mode == "full" or args.include_objects:
        result["objects"] = payload.get("objects") or []
    if args.result_mode == "full" or args.include_tamper:
        result["tamper"] = payload.get("tamper")
    if args.result_mode == "full" or args.include_meta:
        result["meta"] = payload.get("meta", {})

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        result["json"] = str(args.json)

    print_json(result)


if __name__ == "__main__":
    main()
