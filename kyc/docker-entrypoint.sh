#!/usr/bin/env sh
set -eu

if [ "$#" -eq 0 ]; then
  set -- --serve
fi

exec python -m kyc.ocr_service "$@"
