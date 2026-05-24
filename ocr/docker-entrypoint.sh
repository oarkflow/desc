#!/usr/bin/env sh
set -eu

if [ "$#" -eq 0 ]; then
  set -- --serve
fi

exec python /app/ocr_service.py "$@"
