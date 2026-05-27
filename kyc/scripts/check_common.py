import argparse
import json
import mimetypes
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
KYC_ROOT = Path(__file__).resolve().parents[1]
TESTDATA_DIR = KYC_ROOT / "testdata"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("OCR_CACHE_DIR", str(KYC_ROOT / ".ocr_cache"))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def existing_path(value):
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"not found: {path}")
    return path


def image_mime(path):
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "image/jpeg"


def print_json(payload):
    print(json.dumps(payload, indent=2, sort_keys=True))


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def default_test_image(name="citizenship.jpg"):
    return TESTDATA_DIR / name
