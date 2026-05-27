#!/usr/bin/env python3
"""
face CLI — Face Recognition & Landmark Extraction
Usage examples:
  # Analyze a single image (detection + landmarks)
  python -m kyc.face.cli analyze photo.jpg --output out.jpg

  # Analyze with landmark metrics overlay
  python -m kyc.face.cli analyze photo.jpg --output out.jpg --metrics

  # Enroll a person from a folder of images
  python -m kyc.face.cli enroll Alice photos/alice/ --db mydb

  # Recognize faces in a photo against enrolled database
  python -m kyc.face.cli recognize photo.jpg --db mydb --output out.jpg

  # Batch analyze all images in a folder
  python -m kyc.face.cli batch photos/ --output-dir results/

  # Export results as JSON
  python -m kyc.face.cli analyze photo.jpg --json result.json
"""

import argparse
import cv2
import json
import numpy as np
import sys
from collections import defaultdict
from pathlib import Path

# Allow running as script from any directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from kyc.face import FacePlatform
    from kyc.face.image_loader import load_image
    from kyc.face.landmarks import MediaPipeLandmarkDetector, MP_LANDMARK_GROUPS
    from kyc.face.recognizer import SFaceSearcher
    from kyc.face.visualizer import save_image
except ModuleNotFoundError:
    from face import FacePlatform
    from face.image_loader import load_image
    from face.landmarks import MediaPipeLandmarkDetector, MP_LANDMARK_GROUPS
    from face.recognizer import SFaceSearcher
    from face.visualizer import save_image

DEFAULT_LBF_MODEL = str(Path(__file__).parent.parent / "models" / "lbfmodel.yaml")
DEFAULT_MEDIAPIPE_MODEL = str(Path(__file__).parent.parent / "models" / "face_landmarker.task")
DEFAULT_YUNET_MODEL = str(Path(__file__).parent.parent / "models" / "face_detection_yunet_2023mar.onnx")
DEFAULT_SFACE_MODEL = str(Path(__file__).parent.parent / "models" / "face_recognition_sface_2021dec.onnx")
DEFAULT_AGE_MODEL = str(Path(__file__).parent.parent / "models" / "age_net.caffemodel")
DEFAULT_AGE_PROTO = str(Path(__file__).parent.parent / "models" / "age_deploy.prototxt")
DEFAULT_GENDER_MODEL = str(Path(__file__).parent.parent / "models" / "gender_net.caffemodel")
DEFAULT_GENDER_PROTO = str(Path(__file__).parent.parent / "models" / "gender_deploy.prototxt")
DEFAULT_DEMOGRAPHIC_CALIBRATION = str(Path(__file__).parent.parent / "models" / "demographic_calibration.json")


def make_platform(args) -> FacePlatform:
    lbf = getattr(args, "lbf_model", DEFAULT_LBF_MODEL)
    mediapipe = getattr(args, "mediapipe_model", DEFAULT_MEDIAPIPE_MODEL)
    yunet = getattr(args, "yunet_model", DEFAULT_YUNET_MODEL)
    age_model = getattr(args, "age_model", DEFAULT_AGE_MODEL)
    age_proto = getattr(args, "age_proto", DEFAULT_AGE_PROTO)
    gender_model = getattr(args, "gender_model", DEFAULT_GENDER_MODEL)
    gender_proto = getattr(args, "gender_proto", DEFAULT_GENDER_PROTO)
    demographic_calibration = getattr(args, "demographic_calibration", DEFAULT_DEMOGRAPHIC_CALIBRATION)
    db = getattr(args, "db", None)
    detection_mode = args.detection_mode
    yunet_model_path = yunet if Path(yunet).exists() else None
    if detection_mode == "auto":
        detection_mode = "yunet" if yunet_model_path else "multiscale"

    return FacePlatform(
        lbf_model_path=lbf if Path(lbf).exists() else None,
        mediapipe_model_path=mediapipe if Path(mediapipe).exists() else None,
        yunet_model_path=yunet_model_path,
        recognizer_db_path=db,
        detection_mode=detection_mode,
        landmark_mode=args.landmark_mode,
        min_face_size=args.min_face_size,
        recognition_enabled=True,
        demographic_enabled=not getattr(args, "no_demographics", False),
        age_model_path=age_model if Path(age_model).exists() else None,
        age_proto_path=age_proto if Path(age_proto).exists() else None,
        gender_model_path=gender_model if Path(gender_model).exists() else None,
        gender_proto_path=gender_proto if Path(gender_proto).exists() else None,
        demographic_calibration_path=demographic_calibration if Path(demographic_calibration).exists() else None,
    )


def image_paths_from_target(target: str, query: Path | None = None) -> list[str]:
    path = Path(target)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
    if path.is_dir():
        paths = [p for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    elif path.is_file() and path.suffix.lower() in exts:
        paths = [path]
    elif path.is_file():
        base = path.parent
        paths = []
        for line in path.read_text().splitlines():
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            candidate = Path(item)
            if not candidate.is_absolute():
                candidate = base / candidate
                if not candidate.exists():
                    candidate = Path(item)
            if candidate.suffix.lower() in exts:
                paths.append(candidate)
    else:
        raise FileNotFoundError(f"Image target not found: {target}")

    if query is not None:
        query_resolved = query.resolve()
        paths = [p for p in paths if p.resolve() != query_resolved]
    return [str(p) for p in paths]


def crop_detections(image, detections, crop_dir: str, source_path: str) -> list[str]:
    out = Path(crop_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(source_path).stem
    saved = []
    for det in detections:
        crop = det.face_roi(image, padding=0.15)
        path = out / f"{stem}_face{det.index + 1}.jpg"
        save_image(crop, str(path))
        saved.append(str(path))
    return saved


def detection_json_path(output_dir: Path, image_path: str) -> Path:
    safe_name = Path(image_path).name
    return output_dir / f"{safe_name}.faces.json"


def overlay_path(output_dir: Path, image_path: str) -> Path:
    return output_dir / f"{Path(image_path).stem}.overlay.jpg"


def draw_detection_overlay(image, detections, landmarks, overlay: str):
    canvas = image.copy()
    draw_boxes = overlay in ("boxes", "both")
    draw_points = overlay in ("points", "both")

    if draw_boxes:
        for det in detections:
            x, y, w, h = det.bbox
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 220, 120), 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"Face {det.index + 1} {det.confidence:.2f}",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 120),
                2,
                cv2.LINE_AA,
            )

    if draw_points:
        for lm in landmarks:
            for point in lm.points.astype(int):
                cv2.circle(canvas, tuple(point), 1 if len(lm.points) > 100 else 2, (0, 200, 255), -1, cv2.LINE_AA)

    return canvas


def draw_search_overlay(image, matches):
    canvas = image.copy()
    for match in matches:
        x, y, w, h = [int(round(v)) for v in match.bbox]
        color = (0, 220, 120) if match.is_match else (0, 0, 230)
        label = f"{'MATCH' if match.is_match else 'no'} {match.cosine:.2f}"
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return canvas


def crop_search_matches(crop_dir: str, matches) -> list[str]:
    out = Path(crop_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    for match in matches:
        if not match.is_match:
            continue
        image = load_image(match.image_path)
        x, y, w, h = [int(round(v)) for v in match.bbox]
        ih, iw = image.shape[:2]
        pad_x = int(w * 0.15)
        pad_y = int(h * 0.15)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(iw, x + w + pad_x)
        y2 = min(ih, y + h + pad_y)
        crop = image[y1:y2, x1:x2]
        path = out / f"{Path(match.image_path).stem}_face{match.face_index + 1}_match.jpg"
        save_image(crop, str(path))
        saved.append(str(path))
    return saved


def landmark_bounds(landmark):
    pts = landmark.points
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def bbox_center(bbox):
    return np.array([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2], dtype=float)


def choose_landmark_for_bbox(landmarks, bbox):
    if not landmarks:
        return None
    center = bbox_center(bbox)
    return min(landmarks, key=lambda lm: float(np.linalg.norm(bbox_center(landmark_bounds(lm)) - center)))


LANDMARK_REGION_SPECS = [
    {
        "key": "left_eye",
        "label": "left eye",
        "feature": "eye shape/position",
        "indices": MP_LANDMARK_GROUPS["left_eye"],
    },
    {
        "key": "right_eye",
        "label": "right eye",
        "feature": "eye shape/position",
        "indices": MP_LANDMARK_GROUPS["right_eye"],
    },
    {
        "key": "left_eyebrow",
        "label": "left eyebrow",
        "feature": "brow height/shape",
        "indices": MP_LANDMARK_GROUPS["left_eyebrow"],
    },
    {
        "key": "right_eyebrow",
        "label": "right eyebrow",
        "feature": "brow height/shape",
        "indices": MP_LANDMARK_GROUPS["right_eyebrow"],
    },
    {
        "key": "nose",
        "label": "nose",
        "feature": "nose bridge/tip/width",
        "indices": MP_LANDMARK_GROUPS["nose"],
    },
    {
        "key": "mouth_outer",
        "label": "outer mouth",
        "feature": "mouth outline/expression",
        "indices": MP_LANDMARK_GROUPS["lips_outer"],
    },
    {
        "key": "mouth_inner",
        "label": "inner mouth",
        "feature": "mouth opening",
        "indices": MP_LANDMARK_GROUPS["lips_inner"],
    },
    {
        "key": "chin",
        "label": "chin",
        "feature": "lower face/chin position",
        "indices": [152, 148, 176, 149, 150, 136, 172],
    },
    {
        "key": "face_oval",
        "label": "face outline",
        "feature": "jawline/cheeks/overall face shape",
        "indices": MP_LANDMARK_GROUPS["face_oval"],
    },
]


def landmark_points_for_indices(landmark, indices):
    if landmark.points is None:
        return None
    valid = [idx for idx in indices if idx < len(landmark.points)]
    if not valid:
        return None
    return landmark.points[valid].astype(float)


def normalized_landmark_scale(landmark):
    return max(landmark.eye_distance() or max(np.ptp(landmark.points[:, 0]), 1.0), 1e-6)


def offset_description(offset):
    horizontal = ""
    vertical = ""
    if abs(offset[0]) >= 0.08:
        horizontal = "right" if offset[0] > 0 else "left"
    if abs(offset[1]) >= 0.08:
        vertical = "lower" if offset[1] > 0 else "higher"

    if horizontal and vertical:
        return f"{vertical} and {horizontal}"
    if vertical:
        return vertical
    if horizontal:
        return horizontal
    return "similarly positioned"


def normalized_region_difference(query_lm, target_lm, region):
    q = landmark_points_for_indices(query_lm, region["indices"])
    t = landmark_points_for_indices(target_lm, region["indices"])
    if q is None or t is None:
        return None

    n = min(len(q), len(t))
    if n == 0:
        return None

    q = q[:n]
    t = t[:n]
    q_center = query_lm.points.mean(axis=0)
    t_center = target_lm.points.mean(axis=0)
    q_scale = normalized_landmark_scale(query_lm)
    t_scale = normalized_landmark_scale(target_lm)
    qn = (q - q_center) / max(q_scale, 1e-6)
    tn = (t - t_center) / max(t_scale, 1e-6)
    offset = tn.mean(axis=0) - qn.mean(axis=0)
    distance = float(np.mean(np.linalg.norm(qn - tn, axis=1)))
    direction = offset_description(offset)
    return {
        "group": region["key"],
        "label": region["label"],
        "feature": region["feature"],
        "distance": round(distance, 4),
        "position": direction,
        "summary": f"{region['label']} ({region['feature']}) is {direction}",
    }


def explain_landmarks(query_path, matches, mediapipe_model_path, limit):
    model = Path(mediapipe_model_path)
    if not model.exists():
        return {}

    detector = MediaPipeLandmarkDetector(str(model), num_faces=10)
    explanations = {}
    try:
        query_image = load_image(str(query_path))
        query_landmarks = detector.detect(query_image)
        if not query_landmarks:
            return {}
        query_landmark = max(query_landmarks, key=lambda lm: landmark_bounds(lm)[2] * landmark_bounds(lm)[3])

        by_image = defaultdict(list)
        for match in matches[:limit]:
            by_image[match.image_path].append(match)

        for image_path, image_matches in by_image.items():
            image = load_image(image_path)
            target_landmarks = detector.detect(image)
            for match in image_matches:
                target_landmark = choose_landmark_for_bbox(target_landmarks, match.bbox)
                key = (match.image_path, match.face_index)
                if target_landmark is None:
                    explanations[key] = {
                        "available": False,
                        "note": "No MediaPipe landmarks found for this detected face.",
                    }
                    continue

                distances = []
                for region in LANDMARK_REGION_SPECS:
                    difference = normalized_region_difference(query_landmark, target_landmark, region)
                    if difference is not None:
                        distances.append(difference)

                distances.sort(key=lambda item: item["distance"], reverse=True)
                closest = sorted(distances, key=lambda item: item["distance"])[:3]
                most_different = distances[:3]
                if match.is_match:
                    summary = (
                        "Matched by SFace; closest facial landmark regions are "
                        + ", ".join(item["label"] for item in closest)
                        + "."
                    )
                else:
                    summary = (
                        "Rejected by SFace; largest visible landmark differences are around "
                        + ", ".join(item["label"] for item in most_different)
                        + "."
                    )
                explanations[key] = {
                    "available": True,
                    "note": "Landmark differences are normalized by eye distance and are explanatory only; SFace cosine decides identity.",
                    "summary": summary,
                    "most_different": most_different,
                    "most_similar": closest,
                    "not_measured": [
                        {
                            "feature": "mole / skin mark",
                            "reason": "This MediaPipe landmark comparison measures facial geometry only, not texture-level marks.",
                        }
                    ],
                }
    finally:
        detector.close()

    return explanations


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_analyze(args):
    platform = make_platform(args)
    try:
        result = platform.analyze(
            args.image,
            save_annotated=args.output,
            draw_metrics=args.metrics,
        )
        result.print_summary()
        if args.json:
            with open(args.json, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"[✓] JSON saved: {args.json}")
        if args.output:
            print(f"[✓] Annotated image saved: {args.output}")
    finally:
        platform.close()


def cmd_enroll(args):
    platform = make_platform(args)
    try:
        folder = Path(args.folder)
        if folder.is_dir():
            n = platform.enroll_from_folder(args.label, str(folder))
            print(f"[✓] Enrolled '{args.label}' from {n} faces in {folder}")
        else:
            platform.enroll_from_image(args.label, str(folder))
            print(f"[✓] Enrolled '{args.label}' from {folder}")
        platform.save_database(args.db)
    finally:
        platform.close()


def cmd_recognize(args):
    platform = make_platform(args)
    try:
        if not platform.known_faces:
            print("[!] No known faces in database. Use 'enroll' first.")
            return
        print(f"[i] Known faces: {platform.known_faces}")
        result = platform.analyze(
            args.image,
            save_annotated=args.output,
            draw_metrics=args.metrics,
        )
        result.print_summary()
        if args.json:
            with open(args.json, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"[✓] JSON saved: {args.json}")
    finally:
        platform.close()


def cmd_batch(args):
    platform = make_platform(args)
    try:
        folder = Path(args.folder)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
        paths = [str(f) for f in folder.iterdir() if f.suffix.lower() in exts]
        if not paths:
            print(f"[!] No images found in {folder}")
            return
        print(f"[i] Processing {len(paths)} images...")
        results = platform.analyze_batch(
            paths,
            output_folder=args.output_dir,
            draw_metrics=args.metrics,
        )
        print(f"\n[✓] Batch complete: {len(results)} images processed")
        total_faces = sum(r.num_faces for r in results)
        print(f"    Total faces detected: {total_faces}")
        for r in results:
            r.print_summary()
    finally:
        platform.close()


def cmd_info(args):
    try:
        from kyc.face.image_loader import image_info
    except ModuleNotFoundError:
        from face.image_loader import image_info
    img = load_image(args.image)
    info = image_info(img)
    print(f"\nImage: {args.image}")
    for k, v in info.items():
        print(f"  {k}: {v}")


def cmd_search(args):
    query = Path(args.query)
    paths = image_paths_from_target(args.target, query=query)
    if not paths:
        print(f"[!] No searchable images found in {args.target}")
        return

    searcher = SFaceSearcher(
        yunet_model_path=args.yunet_model,
        sface_model_path=args.sface_model,
        cosine_threshold=args.threshold,
        detect_threshold=args.detect_threshold,
    )
    matches = searcher.search(str(query), paths)
    explanations = explain_landmarks(query, matches, args.mediapipe_model, args.explain_limit) if args.verbose else {}

    payload = {
        "query": str(query),
        "target": str(args.target),
        "threshold": args.threshold,
        "matches": [match.to_dict() for match in matches],
    }
    if args.verbose:
        for match_data in payload["matches"]:
            key = (match_data["image_path"], match_data["face_index"])
            if key in explanations:
                match_data["landmark_explanation"] = explanations[key]

    print(f"[i] Query: {query}")
    print(f"[i] Search target: {args.target}")
    print(f"[i] Threshold: cosine >= {args.threshold:.3f}")
    print("\nRanked matches:")
    for match in matches:
        verdict = "MATCH" if match.is_match else "no_match"
        margin = match.cosine - args.threshold
        print(
            f"{verdict:8} cosine={match.cosine:.4f} "
            f"margin={margin:+.4f} l2={match.l2:.4f} face={match.face_index} "
            f"det={match.detection_confidence:.3f} "
            f"bbox={[round(v, 1) for v in match.bbox]} "
            f"{match.image_path}"
        )
        if args.verbose:
            reason = (
                f"accepted because cosine is {margin:.4f} above threshold"
                if match.is_match
                else f"rejected because cosine is {abs(margin):.4f} below threshold"
            )
            print(f"  decision: {reason}; landmark details below are supporting context")
            explanation = explanations.get((match.image_path, match.face_index))
            if explanation and explanation.get("available"):
                print(f"  landmarks: {explanation['summary']}")
                print("  closest landmarks: " + ", ".join(
                    f"{item['label']} {item['distance']:.4f} ({item['feature']}, {item['position']})"
                    for item in explanation["most_similar"]
                ))
                print("  different landmarks: " + ", ".join(
                    f"{item['label']} {item['distance']:.4f} ({item['feature']}, {item['position']})"
                    for item in explanation["most_different"]
                ))
                print("  not measured: mole / skin mark (landmarks compare geometry, not texture)")
            elif explanation:
                print(f"  landmarks: {explanation['note']}")

    if args.overlay_dir:
        overlay_dir = Path(args.overlay_dir)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        grouped = defaultdict(list)
        for match in matches:
            if args.overlay_all or match.is_match:
                grouped[match.image_path].append(match)
        for image_path, image_matches in grouped.items():
            image = load_image(image_path)
            overlay = draw_search_overlay(image, image_matches)
            out_path = overlay_dir / f"{Path(image_path).stem}_search.jpg"
            save_image(overlay, str(out_path))
        print(f"\n[✓] Search overlays saved: {overlay_dir}")

    if args.crop_dir:
        crops = crop_search_matches(args.crop_dir, matches)
        payload["crops"] = crops
        print(f"[✓] Match crops saved: {args.crop_dir} ({len(crops)} files)")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[✓] JSON saved: {args.json}")


def cmd_detect(args):
    platform = make_platform(args)
    try:
        image = load_image(args.image)
        result = platform.analyze(args.image, return_annotated=False)
        crops = []
        if args.crop_dir:
            crops = crop_detections(image, result.faces, args.crop_dir, args.image)
            print(f"[✓] Crops saved: {args.crop_dir} ({len(crops)} files)")

        if args.output and args.overlay != "none":
            overlay = draw_detection_overlay(image, result.faces, result.landmarks, args.overlay)
            save_image(overlay, args.output)
            print(f"[✓] Overlay saved: {args.output}")

        payload = result.to_dict()
        payload["crops"] = crops
        print(f"[i] Faces detected: {result.num_faces}")
        for face in result.faces:
            print(f"face={face.index} conf={face.confidence:.3f} bbox={face.bbox}")

        if args.json:
            with open(args.json, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"[✓] JSON saved: {args.json}")
    finally:
        platform.close()


def cmd_detect_all(args):
    output_dir = Path(args.output_dir)
    points_dir = output_dir / "points"
    overlays_dir = output_dir / "overlays"
    crops_dir = output_dir / "crops"
    points_dir.mkdir(parents=True, exist_ok=True)
    if args.overlay != "none":
        overlays_dir.mkdir(parents=True, exist_ok=True)
    if args.crop:
        crops_dir.mkdir(parents=True, exist_ok=True)

    paths = image_paths_from_target(args.target)
    platform = make_platform(args)
    summary = []
    try:
        for index, image_path in enumerate(paths, 1):
            try:
                image = load_image(image_path)
                result = platform.analyze(image_path, return_annotated=False)
                payload = result.to_dict()

                crops = []
                if args.crop:
                    image_crop_dir = crops_dir / Path(image_path).stem
                    crops = crop_detections(image, result.faces, str(image_crop_dir), image_path)
                    payload["crops"] = crops

                if args.overlay != "none":
                    overlay = draw_detection_overlay(image, result.faces, result.landmarks, args.overlay)
                    save_image(overlay, str(overlay_path(overlays_dir, image_path)))

                json_path = detection_json_path(points_dir, image_path)
                with open(json_path, "w") as f:
                    json.dump(payload, f, indent=2)

                item = {
                    "image_path": image_path,
                    "faces": result.num_faces,
                    "points_json": str(json_path),
                    "crops": crops,
                }
                summary.append(item)
                print(f"[{index}/{len(paths)}] faces={result.num_faces} {image_path}")
            except Exception as exc:
                item = {"image_path": image_path, "error": str(exc)}
                summary.append(item)
                print(f"[{index}/{len(paths)}] error {image_path}: {exc}")
    finally:
        platform.close()

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"target": args.target, "items": summary}, f, indent=2)
    print(f"\n[✓] Face points saved under: {points_dir}")
    if args.overlay != "none":
        print(f"[✓] Overlays saved under: {overlays_dir}")
    if args.crop:
        print(f"[✓] Crops saved under: {crops_dir}")
    print(f"[✓] Summary saved: {summary_path}")


# ── Argument parser ────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="face",
        description="Face Recognition & Landmark Extraction Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--lbf-model", default=DEFAULT_LBF_MODEL,
                        help="Path to LBF landmark model (lbfmodel.yaml)")
    parser.add_argument("--mediapipe-model", default=DEFAULT_MEDIAPIPE_MODEL,
                        help="Path to MediaPipe face_landmarker.task model")
    parser.add_argument("--landmark-mode", default="mediapipe",
                        choices=("auto", "mediapipe", "lbf", "region"),
                        help="Landmark backend to use; mediapipe enforces 478-point landmarks")
    parser.add_argument("--min-face-size", type=int, default=80,
                        help="Minimum face size in pixels for Haar/multiscale detection")
    parser.add_argument("--yunet-model", default=DEFAULT_YUNET_MODEL,
                        help="Path to YuNet ONNX model for face search")
    parser.add_argument("--sface-model", default=DEFAULT_SFACE_MODEL,
                        help="Path to SFace ONNX model for face search")
    parser.add_argument("--detection-mode", default="auto",
                        choices=("auto", "yunet", "multiscale", "haar"),
                        help="Face detector for analyze/detect/recognize")
    parser.add_argument("--age-model", default=DEFAULT_AGE_MODEL,
                        help="Path to age_net.caffemodel")
    parser.add_argument("--age-proto", default=DEFAULT_AGE_PROTO,
                        help="Path to deploy_age.prototxt")
    parser.add_argument("--gender-model", default=DEFAULT_GENDER_MODEL,
                        help="Path to gender_net.caffemodel")
    parser.add_argument("--gender-proto", default=DEFAULT_GENDER_PROTO,
                        help="Path to deploy_gender.prototxt")
    parser.add_argument("--demographic-calibration", default=DEFAULT_DEMOGRAPHIC_CALIBRATION,
                        help="Path to demographic calibration JSON")
    parser.add_argument("--no-demographics", action="store_true",
                        help="Disable age/gender demographic model inference")
    sub = parser.add_subparsers(dest="command", required=True)

    # analyze
    p = sub.add_parser("analyze", help="Detect faces and extract landmarks")
    p.add_argument("image", help="Input image path")
    p.add_argument("--output", "-o", help="Save annotated image here")
    p.add_argument("--json", "-j", help="Save JSON results here")
    p.add_argument("--metrics", action="store_true", help="Draw landmark metrics")
    p.add_argument("--db", help="Recognizer database prefix for identification")

    # enroll
    p = sub.add_parser("enroll", help="Enroll a person into the face database")
    p.add_argument("label", help="Person name/label")
    p.add_argument("folder", help="Image file or folder of face images")
    p.add_argument("--db", required=True, help="Recognizer database prefix")

    # recognize
    p = sub.add_parser("recognize", help="Identify faces against enrolled database")
    p.add_argument("image", help="Input image path")
    p.add_argument("--db", required=True, help="Recognizer database prefix")
    p.add_argument("--output", "-o", help="Save annotated image here")
    p.add_argument("--json", "-j", help="Save JSON results here")
    p.add_argument("--metrics", action="store_true")

    # batch
    p = sub.add_parser("batch", help="Analyze all images in a folder")
    p.add_argument("folder", help="Folder containing images")
    p.add_argument("--output-dir", "-o", help="Folder to save annotated images")
    p.add_argument("--metrics", action="store_true")
    p.add_argument("--db", help="Recognizer database prefix")

    # info
    p = sub.add_parser("info", help="Show image metadata")
    p.add_argument("image", help="Input image path")

    # detect
    p = sub.add_parser("detect", help="Detect faces, draw overlays, and crop faces")
    p.add_argument("image", help="Input image path")
    p.add_argument("--overlay", choices=("none", "boxes", "points", "both"), default="boxes",
                   help="Overlay style to save when --output is used")
    p.add_argument("--output", "-o", help="Save overlay image here")
    p.add_argument("--crop-dir", help="Save detected face crops into this folder")
    p.add_argument("--json", "-j", help="Save JSON detection results here")

    # detect-all
    p = sub.add_parser("detect-all", help="Detect faces in a folder or image-list and store landmark points")
    p.add_argument("target", help="Folder, single image, or text file of image paths to process")
    p.add_argument("--output-dir", "-o", required=True,
                   help="Directory where points JSON, overlays, crops, and summary are saved")
    p.add_argument("--overlay", choices=("none", "boxes", "points", "both"), default="both",
                   help="Overlay style to save")
    p.add_argument("--crop", action="store_true", help="Crop detected faces")

    # search
    p = sub.add_parser("search", help="Search a folder or image-list file for faces matching a query image")
    p.add_argument("query", help="Query image containing the face to search for")
    p.add_argument("target", help="Folder, single image, or text file of image paths to search")
    p.add_argument("--threshold", type=float, default=0.363,
                   help="SFace cosine threshold for match")
    p.add_argument("--detect-threshold", type=float, default=0.45,
                   help="YuNet face detection threshold")
    p.add_argument("--overlay-dir", help="Save square overlays for matched images")
    p.add_argument("--overlay-all", action="store_true",
                   help="When saving overlays, include non-matching detected faces too")
    p.add_argument("--crop-dir", help="Save cropped matching faces into this folder")
    p.add_argument("--verbose", action="store_true",
                   help="Explain match/no-match margins and landmark-region differences")
    p.add_argument("--explain-limit", type=int, default=10,
                   help="Maximum ranked faces to explain with MediaPipe landmarks")
    p.add_argument("--json", "-j", help="Save JSON results here")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "analyze":   cmd_analyze,
        "enroll":    cmd_enroll,
        "recognize": cmd_recognize,
        "batch":     cmd_batch,
        "info":      cmd_info,
        "detect":    cmd_detect,
        "detect-all": cmd_detect_all,
        "search":    cmd_search,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
