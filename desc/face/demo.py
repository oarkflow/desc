#!/usr/bin/env python3
"""
Demo & test script for face_platform.
Downloads a public domain face image and runs the full pipeline:
  - Face detection
  - 68-point landmark extraction
  - Visualization
"""

import sys
import os
import json
import urllib.request
import numpy as np
import cv2
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from face_platform import FacePlatform, load_image

# ── Paths ─────────────────────────────────────────────────────────────────────
LBF_MODEL = str(Path(__file__).parent.parent / "lbfmodel.yaml")
OUTPUT_DIR = Path(__file__).parent.parent / "face_platform" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_synthetic_test_image() -> np.ndarray:
    """
    Create a synthetic face-like image for testing pipeline components
    when no real image is available.
    """
    img = np.ones((480, 640, 3), dtype=np.uint8) * 220

    # Gradient background
    for y in range(480):
        shade = int(180 + 40 * y / 480)
        img[y, :] = [shade - 20, shade - 10, shade]

    # Face oval
    cv2.ellipse(img, (320, 240), (100, 130), 0, 0, 360, (210, 185, 160), -1)
    cv2.ellipse(img, (320, 240), (100, 130), 0, 0, 360, (160, 130, 110), 2)

    # Eyes
    cv2.ellipse(img, (285, 210), (22, 14), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (355, 210), (22, 14), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, (285, 212), 9, (60, 40, 20), -1)
    cv2.circle(img, (355, 212), 9, (60, 40, 20), -1)
    cv2.circle(img, (285, 212), 4, (10, 10, 10), -1)
    cv2.circle(img, (355, 212), 4, (10, 10, 10), -1)
    cv2.circle(img, (288, 208), 2, (255, 255, 255), -1)
    cv2.circle(img, (358, 208), 2, (255, 255, 255), -1)

    # Eyebrows
    cv2.ellipse(img, (285, 193), (25, 7), -10, 200, 340, (100, 70, 50), 3)
    cv2.ellipse(img, (355, 193), (25, 7), 10, 200, 340, (100, 70, 50), 3)

    # Nose
    pts_nose = np.array([[320, 225], [308, 258], [332, 258]], np.int32)
    cv2.polylines(img, [pts_nose], True, (170, 145, 125), 2)
    cv2.ellipse(img, (320, 262), (18, 8), 0, 0, 180, (170, 140, 120), 2)

    # Mouth
    cv2.ellipse(img, (320, 285), (28, 12), 0, 0, 180, (170, 100, 100), 3)
    cv2.line(img, (292, 285), (348, 285), (160, 90, 90), 2)

    # Ears
    cv2.ellipse(img, (220, 240), (16, 26), 0, 80, 280, (200, 175, 150), -1)
    cv2.ellipse(img, (420, 240), (16, 26), 0, -100, 100, (200, 175, 150), -1)

    # Hair
    cv2.ellipse(img, (320, 150), (115, 95), 0, 180, 360, (80, 55, 30), -1)
    cv2.ellipse(img, (320, 150), (115, 95), 0, 180, 360, (70, 45, 20), 3)

    cv2.putText(img, "Synthetic Test Face", (160, 440),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 100), 2)

    return img


def try_download_test_image() -> str:
    """Try to get a real test image (public domain)."""
    # Using a simple generated face from a known public endpoint
    test_path = str(OUTPUT_DIR / "test_input.jpg")

    # Try OpenCV's built-in Lena image (bundled in some builds)
    try:
        lena = cv2.imread(cv2.samples.findFile("lena.jpg", raise_error=False) or "")
        if lena is not None:
            cv2.imwrite(test_path, lena)
            print("[i] Using bundled Lena test image")
            return test_path
    except Exception:
        pass

    # Use synthetic image
    print("[i] Using synthetic test image")
    img = create_synthetic_test_image()
    cv2.imwrite(test_path, img)
    return test_path


def run_demo():
    print("\n" + "="*60)
    print("  FACE PLATFORM — DEMO & TEST")
    print("="*60)

    # ── Initialize platform ────────────────────────────────────────────────
    print("\n[1] Initializing FacePlatform...")
    use_lbf = Path(LBF_MODEL).exists()
    print(f"    LBF landmark model: {'FOUND ✓' if use_lbf else 'NOT FOUND — using region fallback'}")

    platform = FacePlatform(
        lbf_model_path=LBF_MODEL if use_lbf else None,
        detection_mode="multiscale",
        recognition_enabled=True,
    )

    # ── Test image ─────────────────────────────────────────────────────────
    print("\n[2] Loading test image...")
    test_image_path = try_download_test_image()
    print(f"    Path: {test_image_path}")

    # ── Analysis ───────────────────────────────────────────────────────────
    print("\n[3] Running face analysis...")
    annotated_path = str(OUTPUT_DIR / "annotated_result.jpg")
    result = platform.analyze(
        test_image_path,
        save_annotated=annotated_path,
        draw_metrics=True,
        return_annotated=True,
    )

    result.print_summary()

    # ── Save JSON ──────────────────────────────────────────────────────────
    json_path = str(OUTPUT_DIR / "analysis_result.json")
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"[✓] JSON result: {json_path}")
    print(f"[✓] Annotated image: {annotated_path}")

    # ── Enrollment simulation ──────────────────────────────────────────────
    print("\n[4] Testing enrollment pipeline...")
    img = load_image(test_image_path)
    dets = platform.detector.detect(img)

    if dets:
        roi = dets[0].face_roi(img)
        # Augment: brightness variations
        augmented = [roi]
        for delta in [-30, -15, 15, 30]:
            aug = np.clip(roi.astype(np.int32) + delta, 0, 255).astype(np.uint8)
            augmented.append(aug)

        platform.enroll("TestPerson", augmented)
        print(f"    Enrolled 'TestPerson' with {len(augmented)} augmented images")

        # Test prediction
        test_roi = dets[0].face_roi(img)
        rec = platform.recognizer.predict(test_roi)
        print(f"    Recognition test → {rec.label}  confidence={rec.confidence:.2%}  [{rec.method}]")

        # Save DB
        db_path = str(OUTPUT_DIR / "test_db")
        platform.save_database(db_path)
        print(f"    Database saved: {db_path}*")

        # Re-analyze with recognition
        print("\n[5] Re-analyzing with recognition enabled...")
        result2 = platform.analyze(
            test_image_path,
            save_annotated=str(OUTPUT_DIR / "recognized_result.jpg"),
            draw_metrics=True,
        )
        result2.print_summary()
    else:
        print("    [Note] No face detected in synthetic image for enrollment test")
        print("    → Use a real face photograph for full recognition pipeline")

    # ── Landmark detail report ─────────────────────────────────────────────
    if result.landmarks:
        print("\n[6] Landmark detail report:")
        lm = result.landmarks[0]
        print(f"    Mode          : {lm.mode}")
        print(f"    Total points  : {len(lm.points)}")
        if lm.groups:
            print("    Groups:")
            for grp, pts in lm.groups.items():
                print(f"      {grp:<20} {len(pts)} pts  "
                      f"center=({pts.mean(axis=0)[0]:.0f}, {pts.mean(axis=0)[1]:.0f})")
        metrics = {
            "eye_distance_px": lm.eye_distance(),
            "face_width_px":   lm.face_width(),
            "mouth_open_ratio": lm.mouth_open_ratio(),
            "yaw_estimate_deg": lm.yaw_estimate(),
        }
        print("    Metrics:")
        for k, v in metrics.items():
            print(f"      {k:<25} {f'{v:.2f}' if v is not None else 'N/A'}")

    print("\n" + "="*60)
    print("  DEMO COMPLETE")
    print(f"  Output files in: {OUTPUT_DIR}")
    print("="*60 + "\n")

    return result


if __name__ == "__main__":
    run_demo()
