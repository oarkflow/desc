import argparse
import os
from pathlib import Path

import cv2

from check_common import default_test_image, existing_path, print_json, require


def default_face_image():
    path = Path("/tmp/kyc_identity_selfie.jpg")
    if path.exists():
        return path
    try:
        from PIL import Image
        from skimage import data

        Image.fromarray(data.astronaut()).save(path, format="JPEG")
        return path
    except Exception:
        return default_test_image("citizenship.jpg")


def main():
    parser = argparse.ArgumentParser(description="Check ONNX anti-spoofing model availability and inference.")
    parser.add_argument("--image", type=existing_path, default=default_face_image())
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--allow-missing-model", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--input-size", default=None, help="Width,height, for example 80,80.")
    args = parser.parse_args()

    if args.input_size:
        os.environ["ANTI_SPOOF_INPUT_SIZE"] = args.input_size
    os.environ["ANTI_SPOOF_ENABLED"] = "true"

    from kyc.core.liveness import AntiSpoofingProvider
    from kyc.ocr.service import ObjectDetectionService, summarize_object_anti_spoofing

    image = cv2.imread(str(args.image))
    require(image is not None, f"Could not read anti-spoofing image: {args.image}")

    provider = AntiSpoofingProvider(model_path=args.model, threshold=args.threshold)
    result = provider.analyze(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    objects, object_summary = ObjectDetectionService().detect(image_rgb, [])
    document_result = summarize_object_anti_spoofing(objects)
    if result.get("status") == "needs_manual_review" and args.allow_missing_model:
        print_json(
            {
                "check": "anti_spoof",
                "status": "warn",
                "image": str(args.image),
                "result": result,
                "document_face_result": document_result,
                "object_summary": object_summary,
            }
        )
        return

    require(result.get("available"), result.get("reason") or "Anti-spoofing model is not available.")
    require(result.get("status") in {"live", "spoof"}, f"Unexpected anti-spoofing status: {result.get('status')}")

    print_json(
        {
            "check": "anti_spoof",
            "status": "pass",
            "image": str(args.image),
            "result": result,
            "document_face_result": document_result,
            "object_summary": object_summary,
            "objects": objects,
        }
    )


if __name__ == "__main__":
    main()
