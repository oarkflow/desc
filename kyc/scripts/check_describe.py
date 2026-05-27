import argparse
from pathlib import Path

from fastapi.testclient import TestClient

from check_common import default_test_image, existing_path, image_mime, print_json, require


def main():
    parser = argparse.ArgumentParser(description="Check image describe, OCR text, tags, and tamper summary.")
    parser.add_argument("--image", type=existing_path, default=default_test_image("citizenship.jpg"))
    parser.add_argument("--min-tags", type=int, default=1)
    parser.add_argument("--allow-empty-text", action="store_true")
    args = parser.parse_args()

    from kyc.ocr.service import app

    client = TestClient(app)
    with Path(args.image).open("rb") as handle:
        response = client.post(
            "/describe",
            files={"file": (Path(args.image).name, handle, image_mime(args.image))},
        )

    require(response.status_code == 200, f"Describe request failed with HTTP {response.status_code}: {response.text[:500]}")
    payload = response.json()
    require(payload.get("caption"), "Describe response did not include a caption.")
    require(len(payload.get("tags") or []) >= args.min_tags, f"Describe returned fewer than {args.min_tags} tags.")
    require(payload.get("tamper") is not None, "Describe response did not include tamper output.")
    if not args.allow_empty_text:
        require(payload.get("text") is not None, "Describe response did not include OCR text.")

    print_json(
        {
            "check": "describe",
            "status": "pass",
            "image": str(args.image),
            "caption": payload.get("caption"),
            "object_count": payload.get("object_count"),
            "tags": payload.get("tags"),
            "text": payload.get("text"),
            "text_languages": payload.get("text_languages"),
            "tamper": payload.get("tamper"),
        }
    )


if __name__ == "__main__":
    main()
