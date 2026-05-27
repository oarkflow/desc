import argparse
from pathlib import Path

from fastapi.testclient import TestClient

from check_common import default_test_image, existing_path, image_mime, print_json, require


def main():
    parser = argparse.ArgumentParser(description="Check document OCR and key-value extraction through the unified FastAPI app.")
    parser.add_argument("--image", type=existing_path, default=default_test_image("national-id.webp"))
    parser.add_argument("--document-type", default=None)
    parser.add_argument("--lang", default=None)
    parser.add_argument("--values-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--accuracy-mode", default="accurate", choices=["fast", "accurate"])
    parser.add_argument("--retry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-values", type=int, default=1)
    parser.add_argument("--require-key", action="append", default=[])
    parser.add_argument("--allow-missing-model", action="store_true")
    args = parser.parse_args()

    from kyc.ocr.service import app

    params = {
        "values_only": str(args.values_only).lower(),
        "accuracy_mode": args.accuracy_mode,
        "retry": str(args.retry).lower(),
        "include_stats": "true",
    }
    if args.document_type:
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

    if response.status_code != 200 and args.allow_missing_model:
        print_json(
            {
                "check": "ocr",
                "status": "warn",
                "image": str(args.image),
                "http_status": response.status_code,
                "reason": response.text[:500],
            }
        )
        return

    require(response.status_code == 200, f"OCR request failed with HTTP {response.status_code}: {response.text[:500]}")
    payload = response.json()
    values = payload.get("values") or payload.get("response", {}).get("values") or {}
    missing = [key for key in args.require_key if not values.get(key)]
    require(len(values) >= args.min_values, f"OCR extracted {len(values)} values, expected at least {args.min_values}.")
    require(not missing, f"OCR missing required keys: {', '.join(missing)}")

    print_json(
        {
            "check": "ocr",
            "status": "pass",
            "image": str(args.image),
            "document_type": payload.get("document_type") or payload.get("meta", {}).get("document_type"),
            "value_count": len(values),
            "values": values,
            "meta": payload.get("meta", {}),
        }
    )


if __name__ == "__main__":
    main()
