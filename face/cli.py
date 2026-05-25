#!/usr/bin/env python3
"""
face CLI — Face Recognition & Landmark Extraction
Usage examples:
  # Analyze a single image (detection + landmarks)
  python -m face.cli analyze photo.jpg --output out.jpg

  # Analyze with landmark metrics overlay
  python -m face.cli analyze photo.jpg --output out.jpg --metrics

  # Enroll a person from a folder of images
  python -m face.cli enroll Alice photos/alice/ --db mydb

  # Recognize faces in a photo against enrolled database
  python -m face.cli recognize photo.jpg --db mydb --output out.jpg

  # Batch analyze all images in a folder
  python -m face.cli batch photos/ --output-dir results/

  # Export results as JSON
  python -m face.cli analyze photo.jpg --json result.json
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running as script from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from face import FacePlatform

DEFAULT_LBF_MODEL = str(Path(__file__).parent.parent / "lbfmodel.yaml")


def make_platform(args) -> FacePlatform:
    lbf = getattr(args, "lbf_model", DEFAULT_LBF_MODEL)
    db = getattr(args, "db", None)

    return FacePlatform(
        lbf_model_path=lbf if Path(lbf).exists() else None,
        recognizer_db_path=db,
        detection_mode="multiscale",
        min_face_size=args.min_face_size,
        recognition_enabled=True,
    )


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_analyze(args):
    platform = make_platform(args)
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


def cmd_enroll(args):
    platform = make_platform(args)
    folder = Path(args.folder)
    if folder.is_dir():
        n = platform.enroll_from_folder(args.label, str(folder))
        print(f"[✓] Enrolled '{args.label}' from {n} faces in {folder}")
    else:
        platform.enroll_from_image(args.label, str(folder))
        print(f"[✓] Enrolled '{args.label}' from {folder}")
    platform.save_database(args.db)


def cmd_recognize(args):
    platform = make_platform(args)
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


def cmd_batch(args):
    platform = make_platform(args)
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


def cmd_info(args):
    from face_platform.image_loader import load_image, image_info
    img = load_image(args.image)
    info = image_info(img)
    print(f"\nImage: {args.image}")
    for k, v in info.items():
        print(f"  {k}: {v}")


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
    parser.add_argument("--min-face-size", type=int, default=80,
                        help="Minimum face size in pixels for Haar/multiscale detection")
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
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
