#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KYC_DIR="$ROOT_DIR/kyc"
PYTHON="${KYC_TEST_PYTHON:-$KYC_DIR/.ocr-venv/bin/python}"
OCR_SMOKE_IMAGE="${OCR_SMOKE_IMAGE:-testdata/nepali-citizenship-card.png.webp}"
OCR_SMOKE_OUTPUT="${OCR_SMOKE_OUTPUT:-/tmp/kyc_values_only_smoke.json}"

section() {
  printf '\n==> %s\n' "$1"
}

require_file() {
  if [ ! -e "$1" ]; then
    printf 'Missing required file: %s\n' "$1" >&2
    return 1
  fi
}

section "Checking test runtime"
require_file "$PYTHON"
require_file "$KYC_DIR/models/face_landmarker.task"
"$PYTHON" --version

export MPLCONFIGDIR="${MPLCONFIGDIR:-$KYC_DIR/.cache/matplotlib}"
export OCR_CACHE_DIR="${OCR_CACHE_DIR:-$KYC_DIR/.ocr_cache}"
mkdir -p "$MPLCONFIGDIR" "$OCR_CACHE_DIR"

section "Compiling Python modules"
"$PYTHON" -m py_compile \
  "$KYC_DIR/index.py" \
  "$KYC_DIR/blink.py" \
  "$KYC_DIR/kyc.py" \
  "$KYC_DIR"/core/*.py \
  "$KYC_DIR/ocr_service.py" \
  "$KYC_DIR"/ocr/*.py

section "Validating Docker Compose"
docker compose -f "$KYC_DIR/docker-compose.yml" config >/tmp/kyc_docker_compose_config.yml

section "Running KYC app unit tests"
(
  cd "$ROOT_DIR"
  "$PYTHON" -m unittest kyc.tests.test_kyc kyc.tests.test_index
)

section "Running OCR unit tests"
(
  cd "$ROOT_DIR"
  "$PYTHON" -m unittest kyc.tests.test_ocr_service
)

section "Running gateway Go tests"
(
  cd "$KYC_DIR"
  go test ./cmd/gateway
)

section "Running OCR values-only CLI smoke"
(
  cd "$KYC_DIR"
  "$PYTHON" ocr_service.py --image "$OCR_SMOKE_IMAGE" --values-only >"$OCR_SMOKE_OUTPUT"
)
"$PYTHON" -m json.tool "$OCR_SMOKE_OUTPUT" >/tmp/kyc_values_only_smoke.pretty.json
"$PYTHON" - "$OCR_SMOKE_OUTPUT" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)

if not isinstance(payload, dict):
    raise SystemExit("values-only output must be a JSON object")

for forbidden in ("document_type", "meta", "objects", "object_summary", "tamper", "tamper_score", "status", "flags"):
    if forbidden in payload:
        raise SystemExit(f"values-only output contains non-value key: {forbidden}")

required_any = {"citizenship_number", "full_name", "date_of_birth"}
missing = sorted(required_any - payload.keys())
if missing:
    raise SystemExit(f"values-only smoke missing expected keys: {', '.join(missing)}")

print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
PY

section "Checking Git diff whitespace"
(
  cd "$ROOT_DIR"
  git diff --check
)

section "All KYC checks passed"
