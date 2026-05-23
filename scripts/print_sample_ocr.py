import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kyc import OCRService


def main():
    sample_path = ROOT / "sample.jpg"
    if not sample_path.exists():
        print("\nSample OCR JSON: skipped, sample.jpg not found")
        return

    result = OCRService().extract(sample_path)
    output = {
        "source": str(sample_path),
        "risk_status": result.get("risk_status"),
        "confidence": result.get("confidence"),
        "languages": result.get("languages", []),
        "language_status": result.get("language_status"),
        "structured_fields": result.get("normalized", {}).get("structured_fields", {}),
    }
    print("\nSample OCR structured JSON:")
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
