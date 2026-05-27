import argparse
import os
from pathlib import Path

import cv2

from check_common import default_test_image, existing_path, print_json, require


def main():
    parser = argparse.ArgumentParser(description="Check ONNX anti-spoofing model availability and inference.")
    parser.add_argument("--image", type=existing_path, default=default_test_image("citizenship.jpg"))
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--allow-missing-model", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--input-size", default=None, help="Width,height, for example 80,80.")
    args = parser.parse_args()

    if args.input_size:
        os.environ["ANTI_SPOOF_INPUT_SIZE"] = args.input_size
    os.environ["ANTI_SPOOF_ENABLED"] = "true"

    from kyc.core.liveness import AntiSpoofingProvider

    image = cv2.imread(str(args.image))
    require(image is not None, f"Could not read anti-spoofing image: {args.image}")

    provider = AntiSpoofingProvider(model_path=args.model, threshold=args.threshold)
    result = provider.analyze(image)
    if result.get("status") == "needs_manual_review" and args.allow_missing_model:
        print_json(
            {
                "check": "anti_spoof",
                "status": "warn",
                "image": str(args.image),
                "result": result,
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
        }
    )


if __name__ == "__main__":
    main()
