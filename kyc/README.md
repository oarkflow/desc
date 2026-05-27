# KYC OCR Gateway

The gateway is a small Go HTTP service that fronts the Python OCR service. It adds API-key authentication, request size limits, queue/concurrency control, round-robin upstream selection, default OCR query options, health checks, and Prometheus-style metrics.

By default Docker Compose exposes the gateway on `http://localhost:8000` and keeps the OCR service private on the Docker network.

## Requirements

- Docker and Docker Compose for the recommended runtime.
- A configured `GATEWAY_API_KEY` for protected deployments.
- Input files must be images accepted by the OCR service, sent as `multipart/form-data`.
- Default maximum request body size is `15 MB`, controlled by `OCR_MAX_FILE_MB`.
- CPU mode works by default. GPU mode requires a Docker runtime with GPU support and the `docker-compose.gpu.yml` override.

## Run

```sh
GATEWAY_API_KEY=change-me docker compose up --build
```

GPU mode:

```sh
GATEWAY_API_KEY=change-me docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Health check:

```sh
curl http://localhost:8000/healthz
```

## Complete Local Examples

Run the Python OCR/KYC HTTP server locally:

```sh
make -C kyc run PORT=8001
```

Run the Go gateway against that local OCR server from a second terminal:

```sh
cd kyc
GATEWAY_PORT=8000 \
GATEWAY_OCR_UPSTREAM=http://127.0.0.1:8001 \
GATEWAY_API_KEY=change-me \
go run ./cmd/gateway
```

If port `8000` is already in use, pick another port, for example `GATEWAY_PORT=8002`, and replace `localhost:8000` in the curl examples with `localhost:8002`.

Open the gateway identity workbench for browser testing:

```text
http://localhost:8000/identity?api_key=change-me
```

The workbench is served by the Go gateway and calls the Python OCR/KYC service through the gateway. It supports document upload, portrait upload, document OCR/object/tamper checks, face analysis, anti-spoofing, and webcam liveness challenge frames.

Run the complete real-model image matrix and write reports:

```sh
make -C kyc image-matrix
```

Generated reports:

```text
test-results/image-matrix/report.md
test-results/image-matrix/results.json
kyc/reports/image-feature-test-report.md
```

### CLI Checks

Run all local capability checks:

```sh
cd kyc
.venv/bin/python scripts/check_all.py
```

Run any individual identity-suite feature through one dispatcher:

```sh
cd kyc

# Document fields + document face/photo detection + document anti-spoof
.venv/bin/python scripts/identity_suite.py document \
  --image testdata/national-id.webp \
  --json ../test-results/document-identity/national-id.json

# Citizenship card fields + document face/photo detection + document anti-spoof
.venv/bin/python scripts/identity_suite.py document \
  --image testdata/nepali-citizenship-card.png.webp \
  --json ../test-results/document-identity/citizenship-card.json

# Add summaries or full OCR evidence only when needed
.venv/bin/python scripts/identity_suite.py document \
  --image testdata/nepali-citizenship-card.png.webp \
  --result-mode summary \
  --crop-faces \
  --json ../test-results/document-identity/citizenship-card-summary.json

.venv/bin/python scripts/identity_suite.py document \
  --image testdata/nepali-citizenship-card.png.webp \
  --result-mode full \
  --json ../test-results/document-identity/citizenship-card-full.json

# Force a profile only when debugging a specific extractor
.venv/bin/python scripts/identity_suite.py document \
  --image testdata/nepali-citizenship-card.png.webp \
  --document-type nepali_citizenship_mixed_language \
  --force-document-type \
  --json ../test-results/document-identity/citizenship-card-forced.json

# Document fields + document face match against a selfie/reference face
.venv/bin/python scripts/identity_suite.py document \
  --image testdata/national-id.webp \
  --match-face /tmp/kyc_identity_selfie.jpg \
  --match-landmarks \
  --json ../test-results/document-identity/national-id-match.json

# Face detection with landmarks and overlay
.venv/bin/python scripts/identity_suite.py face-detect \
  testdata/citizenship.jpg \
  --overlay both \
  --output ../test-results/face/document-face.overlay.jpg \
  --crop-dir ../test-results/face/crops \
  --json ../test-results/face/document-face.json

# Face enrollment and recognition
.venv/bin/python scripts/identity_suite.py face-enroll alice /path/to/alice-images \
  --db ../test-results/face/known-faces

.venv/bin/python scripts/identity_suite.py face-recognize /path/to/query.jpg \
  --db ../test-results/face/known-faces \
  --json ../test-results/face/recognition.json

# Face search over a folder or image-list file
.venv/bin/python scripts/identity_suite.py face-search /path/to/query.jpg /path/to/search-folder \
  --verbose \
  --overlay-dir ../test-results/face/search-overlays \
  --crop-dir ../test-results/face/search-crops \
  --json ../test-results/face/search.json

# Liveness and face anti-spoofing
.venv/bin/python scripts/identity_suite.py liveness --image /tmp/kyc_identity_selfie.jpg
.venv/bin/python scripts/identity_suite.py anti-spoof --image /tmp/kyc_identity_selfie.jpg

# Image description/tamper, KYC flow, and full matrix
.venv/bin/python scripts/identity_suite.py describe --image testdata/citizenship.jpg --allow-empty-text
.venv/bin/python scripts/identity_suite.py kyc-flow --document testdata/national-id.webp
.venv/bin/python scripts/identity_suite.py matrix
```

Run the underlying scripts directly when you want focused checks:

```sh
cd kyc

# Document fields, document face/photo crops, document anti-spoof, and optional selfie match
.venv/bin/python scripts/check_document_identity.py \
  --image testdata/national-id.webp \
  --match-face /tmp/kyc_identity_selfie.jpg \
  --match-landmarks \
  --json ../test-results/document-identity/national-id-match.json

# OCR and document field extraction
.venv/bin/python scripts/check_ocr.py \
  --image testdata/national-id.webp \
  --document-type nepali_national_id \
  --values-only \
  --accuracy-mode fast \
  --no-retry

# Document/image description, object tags, OCR text, and tamper summary
.venv/bin/python scripts/check_describe.py \
  --image testdata/citizenship.jpg \
  --allow-empty-text

# Face detection, 478-point landmarks, recognition, and demographics
.venv/bin/python scripts/check_face.py \
  --detection-mode yunet \
  --landmark-mode mediapipe \
  --require-demographics

# Face/liveness frame state with anti-spoofing result embedded
.venv/bin/python scripts/check_liveness.py \
  --challenge look_center

# Direct ONNX face anti-spoofing inference
.venv/bin/python scripts/check_anti_spoof.py

# Full KYC session flow with document, selfie, and liveness records
.venv/bin/python scripts/check_kyc_flow.py \
  --document testdata/national-id.webp
```

Face CLI examples from the repository root:

```sh
# Analyze one image and save JSON/overlay outputs
kyc/.venv/bin/python -m kyc.face.cli \
  --detection-mode yunet \
  --landmark-mode mediapipe \
  --mediapipe-model kyc/models/face_landmarker.task \
  analyze kyc/testdata/citizenship.jpg \
  --json test-results/face.json \
  --output test-results/face.overlay.jpg

# Batch detect every supported image in a folder
kyc/.venv/bin/python -m kyc.face.cli \
  --landmark-mode mediapipe \
  --mediapipe-model kyc/models/face_landmarker.task \
  detect-all test-results/image-matrix/fixtures \
  --output-dir test-results/face-detect-all \
  --overlay both \
  --crop
```

### Curl Requests

Gateway OCR:

```sh
curl -sS -X POST "http://localhost:8000/ocr?values_only=false&include_stats=true&detect_objects=true&accuracy_mode=fast&retry=true" \
  -H "X-API-Key: change-me" \
  -F "file=@kyc/testdata/national-id.webp"
```

Gateway describe/tamper:

```sh
curl -sS -X POST "http://localhost:8000/describe" \
  -H "X-API-Key: change-me" \
  -F "file=@kyc/testdata/citizenship.jpg"
```

Gateway health and metrics:

```sh
curl -sS http://localhost:8000/healthz
curl -sS -H "X-API-Key: change-me" http://localhost:8000/metrics
```

Gateway admin config and preview:

```sh
curl -sS -H "X-API-Key: change-me" http://localhost:8000/admin/api/config

curl -sS -X POST "http://localhost:8000/admin/api/preview" \
  -H "X-API-Key: change-me" \
  -F "document_type=nepali_national_id" \
  -F "values_only=false" \
  -F "include_stats=true" \
  -F "file=@kyc/testdata/national-id.webp"
```

Gateway identity workbench page and static asset smoke checks:

```sh
curl -sS "http://localhost:8000/identity?api_key=change-me"

curl -sS "http://localhost:8000/identity/static/identity.js" \
  -H "X-API-Key: change-me"
```

Gateway document OCR with automatic document type detection:

```sh
curl -sS -X POST "http://localhost:8000/ocr?values_only=true&include_stats=true" \
  -H "X-API-Key: change-me" \
  -F "file=@kyc/testdata/national-id.webp"
```

Gateway document OCR with full evidence, object detection, document face/photo detection, document anti-spoofing, and tamper summary:

```sh
curl -sS -X POST "http://localhost:8000/ocr?values_only=false&detect_objects=true&accuracy_mode=accurate&retry=true" \
  -H "X-API-Key: change-me" \
  -F "file=@kyc/testdata/national-id.webp"
```

Gateway portrait/face analysis with anti-spoofing:

```sh
curl -sS -X POST "http://localhost:8000/identity/api/portrait" \
  -H "X-API-Key: change-me" \
  -F "file=@/tmp/kyc_identity_selfie.jpg"
```

Gateway liveness frame challenge:

```sh
curl -sS -X POST "http://localhost:8000/identity/api/liveness/frame?session_id=test-session&challenge=blink&challenge=turn_left&challenge=turn_right&challenge=look_center" \
  -H "X-API-Key: change-me" \
  -F "file=@/tmp/kyc_identity_selfie.jpg"
```

Gateway liveness completion:

```sh
curl -sS -X POST "http://localhost:8000/identity/api/liveness/complete?session_id=test-session&challenge=blink&challenge=turn_left&challenge=turn_right&challenge=look_center" \
  -H "X-API-Key: change-me"
```

Python OCR/KYC server direct requests:

```sh
# Create a KYC session
curl -sS -X POST http://localhost:8001/api/kyc/sessions > /tmp/kyc-session.json

# Extract session fields with Python/jq/etc. Example with jq:
SESSION_ID="$(jq -r .session_id /tmp/kyc-session.json)"
SESSION_TOKEN="$(jq -r .session_token /tmp/kyc-session.json)"

# Save applicant profile
curl -sS -X POST "http://localhost:8001/api/kyc/sessions/${SESSION_ID}/profile" \
  -H "X-Session-Token: ${SESSION_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"full_name":"Matrix Test","date_of_birth":"1990-01-01","nationality":"Nepal","address":"Kathmandu","document_type":"national_id","document_number":"123-456-7890"}'

# Upload document
curl -sS -X POST "http://localhost:8001/api/kyc/sessions/${SESSION_ID}/documents" \
  -H "X-Session-Token: ${SESSION_TOKEN}" \
  -F "document_type=national_id" \
  -F "side=front" \
  -F "file=@kyc/testdata/national-id.webp"

# Upload selfie
curl -sS -X POST "http://localhost:8001/api/kyc/sessions/${SESSION_ID}/selfie" \
  -H "X-Session-Token: ${SESSION_TOKEN}" \
  -F "file=@/tmp/kyc_identity_selfie.jpg"

# Process one liveness frame and complete the challenge
curl -sS -X POST "http://localhost:8001/api/kyc/sessions/${SESSION_ID}/liveness/frame" \
  -H "X-Session-Token: ${SESSION_TOKEN}" \
  -F "file=@/tmp/kyc_identity_selfie.jpg"

curl -sS -X POST "http://localhost:8001/api/kyc/sessions/${SESSION_ID}/liveness/complete" \
  -H "X-Session-Token: ${SESSION_TOKEN}"

# Fetch the final case
curl -sS "http://localhost:8001/api/kyc/sessions/${SESSION_ID}" \
  -H "X-Session-Token: ${SESSION_TOKEN}"
```

## Anti-Spoofing And Tamper Coverage

Face anti-spoofing is implemented for both live face images and document images through `kyc.core.liveness.AntiSpoofingProvider`. For liveness frames and videos, the ONNX model runs on the detected face crop and returns `anti_spoofing.status` as `live`, `spoof`, or `needs_manual_review`; liveness marks the session risk as `fail` when the model returns `spoof`.

Document anti-spoofing runs when OCR object detection finds `face`, `photo`, or `portrait` regions on an uploaded document. Each matching object gets an `anti_spoofing` result, and the aggregate appears at `object_summary.anti_spoofing` and `tamper.checks.object_summary.anti_spoofing`. Document protection still also includes OCR/describe tamper analysis: `tamper`, `tamper_score`, `flags`, expected object checks, protected-region checks, field validation, and object summaries. Use `/ocr?values_only=false&detect_objects=true` for document OCR plus anti-spoof/tamper/object evidence.

## HTTP Endpoints

### `GET /admin`

Serves the gateway-hosted OCR admin console for managing document type YAML, global OCR settings, data files, and upload previews.

If `GATEWAY_API_KEY` is set, open the UI with the key once to set an admin cookie:

```sh
http://localhost:8000/admin?api_key=change-me
```

The UI can:

- List, create, edit, duplicate, and soft-delete document types in `config/document_types`.
- Edit `config/document_profiles.yaml`, including extraction rules.
- Edit `.yaml`, `.yml`, and `.txt` files in `data`.
- Upload a document preview through multipart form upload and view OCR JSON plus image overlays.
- Validate YAML through the OCR service before saving.
- Save files atomically with timestamped backups and request OCR config reloads.

### `GET /identity`

Serves the gateway-hosted identity workbench for browser testing. If `GATEWAY_API_KEY` is set, open it with the key once to set a cookie:

```sh
http://localhost:8000/identity?api_key=change-me
```

The workbench provides document upload, portrait upload, document OCR/object/tamper checks, face analysis, anti-spoofing, and webcam liveness challenge capture.

### `POST /identity/api/portrait`

Proxies portrait face analysis to the Python OCR/KYC service. The response includes face count, boxes/landmarks where available, demographics where available, and anti-spoofing status.

```sh
curl -sS -X POST "http://localhost:8000/identity/api/portrait" \
  -H "X-API-Key: change-me" \
  -F "file=@/tmp/kyc_identity_selfie.jpg"
```

### `POST /identity/api/liveness/frame`

Proxies one captured webcam frame to the Python liveness service. Send the same `session_id` for all frames in a single browser/test run. Repeating `challenge` query parameters defines the expected challenge sequence.

```sh
curl -sS -X POST "http://localhost:8000/identity/api/liveness/frame?session_id=test-session&challenge=blink&challenge=turn_left&challenge=turn_right&challenge=look_center" \
  -H "X-API-Key: change-me" \
  -F "file=@/tmp/kyc_identity_selfie.jpg"
```

### `POST /identity/api/liveness/complete`

Completes a liveness session and returns the aggregate challenge result.

```sh
curl -sS -X POST "http://localhost:8000/identity/api/liveness/complete?session_id=test-session&challenge=blink&challenge=turn_left&challenge=turn_right&challenge=look_center" \
  -H "X-API-Key: change-me"
```

### `GET /healthz`

Returns gateway health. This endpoint does not require an API key.

Response:

```json
{
  "status": "ok"
}
```

### `GET /metrics`

Returns gateway metrics in Prometheus text format. If `GATEWAY_API_KEY` is set, the request must include `X-API-Key`.

Request:

```sh
curl -H "X-API-Key: change-me" http://localhost:8000/metrics
```

Response content type:

```text
text/plain; version=0.0.4; charset=utf-8
```

Exported metrics include:

- `kyc_gateway_requests_total`
- `kyc_gateway_ocr_requests_total`
- `kyc_gateway_ocr_success_total`
- `kyc_gateway_ocr_errors_total`
- `kyc_gateway_auth_failures_total`
- `kyc_gateway_queue_rejected_total`
- `kyc_gateway_request_too_large_total`
- `kyc_gateway_upstream_errors_total`
- `kyc_gateway_active_ocr`
- `kyc_gateway_queued_ocr`
- `kyc_gateway_ocr_latency_ms_total`
- `kyc_gateway_queue_wait_ms_total`
- `kyc_gateway_responses_total{status="..."}`

### `POST /ocr`

Proxies an OCR request to the upstream OCR service. If `GATEWAY_API_KEY` is set, the request must include `X-API-Key`.

Headers:

- `X-API-Key: <key>` when gateway auth is enabled.
- `Content-Type: multipart/form-data`.
- `Accept` is forwarded upstream when provided.

Multipart fields:

- `file`: required image upload.

Query parameters forwarded to the OCR service:

| Parameter | Default at gateway | Description |
| --- | --- | --- |
| `document_type` | auto-detect upstream | Optional forced document profile override. Omit this for normal use so OCR detects the profile from document cues. Known profiles include `nepali_citizenship_old_front`, `nepali_citizenship_front_back`, `nepali_citizenship_mixed_language`, `nepali_national_id`, and `generic_devanagari_document`. |
| `lang` | OCR service default, usually `ne` | OCR language override. |
| `accuracy_mode` | `accurate` | Gateway injects this when omitted. OCR supports profile-dependent behavior for `fast` and `accurate`. |
| `retry` | `true` | Gateway injects this when omitted. Set `false` for lower-latency requests. |
| `values_only` | `true` | Gateway injects this when omitted. Returns a lightweight `{document_type, values, meta}` response. Set `false` for the full OCR payload. |
| `fields_only` | `false` | Returns structured field details without raw OCR items when true and `values_only=false`. |
| `include_stats` | `false` | Adds processing metadata when `values_only=true`. |
| `upscale` | `true` | Image preprocessing toggle. |
| `denoise` | `false` | Image preprocessing toggle. |
| `threshold` | `false` | Image preprocessing toggle. |
| `crop_border` | `true` | Image preprocessing toggle. |
| `enhance` | `true` | Image preprocessing toggle. |
| `clean_background` | OCR service setting | Image preprocessing toggle. |

Minimal request:

```sh
curl -X POST "http://localhost:8000/ocr" \
  -H "X-API-Key: change-me" \
  -F "file=@testdata/national-id.webp"
```

Full response request:

```sh
curl -X POST "http://localhost:8000/ocr?values_only=false&accuracy_mode=accurate&retry=true" \
  -H "X-API-Key: change-me" \
  -F "file=@testdata/national-id.webp"
```

Default `values_only=true` response:

```json
{
  "document_type": "nepali_national_id",
  "values": {
    "nid_number": "123-456-7890"
  },
  "meta": {
    "document_type": "nepali_national_id",
    "document_type_confidence": 0.3333
  }
}
```

`values_only=true&include_stats=true` response:

```json
{
  "document_type": "nepali_national_id",
  "values": {
    "nid_number": "123-456-7890"
  },
  "meta": {
    "document_type": "nepali_national_id",
    "document_type_confidence": 0.3333,
    "device": "cpu",
    "gpu": false,
    "processing_ms": 1200,
    "resource_usage": {
      "wall_ms": 1200,
      "cpu_ms": 1800,
      "max_rss_mb": 512.4
    }
  }
}
```

`values_only=false` response shape:

```json
{
  "request_id": "uuid",
  "filename": "national-id.webp",
  "mime_type": "image/webp",
  "file_size_bytes": 123456,
  "width": 1200,
  "height": 800,
  "processing_ms": 1200,
  "document_type": "nepali_national_id",
  "document_type_confidence": 0.3333,
  "full_text": "recognized text",
  "values": {
    "nid_number": "123-456-7890"
  },
  "fields": {
    "nid_number": {
      "value": "123-456-7890",
      "confidence": 0.94,
      "source_text": "ID No 123-456-7890",
      "raw_value": "123-456-7890",
      "normalized_value": "123-456-7890",
      "requires_review": false,
      "evidence": [],
      "details": {}
    }
  },
  "items": [
    {
      "text": "ID No 123-456-7890",
      "confidence": 0.94,
      "box": [[0, 0], [100, 0], [100, 20], [0, 20]],
      "source_pass": "english"
    }
  ],
  "meta": {
    "engine": "paddleocr",
    "lang": "ne",
    "device": "cpu",
    "document_type": "nepali_national_id",
    "document_type_confidence": 1,
    "gpu": false,
    "preprocessing": {
      "upscale": true,
      "denoise": false,
      "threshold": false,
      "crop_border": true,
      "enhance": true,
      "clean_background": false
    }
  }
}
```

## Error Responses

Gateway-generated errors are JSON:

```json
{
  "error": "unauthorized"
}
```

Common gateway statuses:

| Status | Cause |
| --- | --- |
| `400` | Request is not `multipart/form-data` or the body cannot be read. |
| `401` | `GATEWAY_API_KEY` is set and `X-API-Key` is missing or incorrect. |
| `404` | Unknown route. |
| `413` | Request body is larger than `OCR_MAX_FILE_MB`. |
| `429` | OCR concurrency and queue are full. |
| `502` | Gateway could not create or complete an upstream OCR request. |
| `504` | Upstream OCR request timed out. |

The gateway passes upstream OCR response headers, status codes, and bodies through for completed upstream requests.

### `POST /describe`

Proxies an image description request to the Python OCR service. If `GATEWAY_API_KEY` is set, the request must include `X-API-Key`. The implementation lives under `kyc/describe` and returns detected objects, a rule-based caption, optional Tesseract OCR text, tags, dimensions, and a lightweight tamper heuristic.

```sh
curl -X POST "http://localhost:8000/describe" \
  -H "X-API-Key: change-me" \
  -F "file=@testdata/citizenship.jpg"
```

### Face Analysis

The face platform now lives under `kyc/face`. Use `python -m kyc.face.cli ...` from the repository root.

## Admin API

All admin API routes are under `/admin/api` and require the same gateway API key when `GATEWAY_API_KEY` is configured.

| Endpoint | Purpose |
| --- | --- |
| `GET /admin/api/config` | Load document type summaries, data file summaries, and global profile YAML. |
| `GET /admin/api/document-types` | List document type YAML files. |
| `GET /admin/api/document-types/{id}` | Read one document type YAML file. |
| `POST /admin/api/document-types` | Create a document type from JSON `{ "name": "...", "content": "..." }`. |
| `PUT /admin/api/document-types/{id}` | Validate and save document type YAML. |
| `POST /admin/api/document-types/{id}/duplicate` | Duplicate a document type using JSON `{ "id": "new_id" }`. |
| `DELETE /admin/api/document-types/{id}` | Move the document type file to backups and remove it from active config. |
| `GET /admin/api/profiles` | Read `document_profiles.yaml`. |
| `PUT /admin/api/profiles` | Validate and save `document_profiles.yaml`. |
| `GET /admin/api/data` | List editable data files. |
| `GET /admin/api/data/{name}` | Read a data file. |
| `PUT /admin/api/data/{name}` | Save a data file. |
| `POST /admin/api/preview` | Multipart preview upload; forwards `file` and OCR options to upstream `/ocr`. |
| `POST /admin/api/reload` | Request OCR config/data cache reload. |

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `GATEWAY_PORT` | `8000` | Gateway listen port inside the container. Compose maps `${GATEWAY_PORT:-8000}` on the host to container port `8000`. |
| `GATEWAY_OCR_UPSTREAM` | `http://ocr:8000` | Single upstream OCR service URL. |
| `GATEWAY_OCR_UPSTREAMS` | value of `GATEWAY_OCR_UPSTREAM` | Comma-separated upstream OCR service URLs for round-robin routing. |
| `GATEWAY_API_KEY` | empty | Enables API-key auth when set. Requests must send `X-API-Key`. |
| `GATEWAY_CONFIG_DIR` | `config` | Directory containing `document_profiles.yaml` and `document_types`. |
| `GATEWAY_DATA_DIR` | `data` | Directory containing editable OCR data files. |
| `GATEWAY_BACKUP_DIR` | `.gateway_backups` | Directory for timestamped backups before admin writes/deletes. |
| `GATEWAY_MAX_ACTIVE` | `OCR_WORKERS` or `1` | Maximum concurrent OCR proxy requests. |
| `GATEWAY_MAX_QUEUE` | `8` | Number of OCR requests allowed to wait for a worker. Use `0` to disable waiting. |
| `GATEWAY_UPSTREAM_TIMEOUT_SECONDS` | `120` | HTTP client timeout for upstream OCR calls. |
| `GATEWAY_SHUTDOWN_TIMEOUT_SECONDS` | `10` | Reserved shutdown timeout setting. |
| `GATEWAY_READ_HEADER_TIMEOUT_SECONDS` | `10` | HTTP server read-header timeout. |
| `GATEWAY_DEFAULT_ACCURACY_MODE` | `accurate` | Injected `accuracy_mode` query value when the client omits it. Use `fast` for lower-latency deployments. |
| `GATEWAY_DEFAULT_RETRY` | `true` | Injected `retry` query value when the client omits it. Use `false` for lower-latency deployments. |
| `OCR_MAX_FILE_MB` | `15` | Maximum gateway request body size and OCR upload size. |

OCR service settings are configured on the `ocr` Compose service. Common values include `OCR_WORKERS`, `OCR_LOG_LEVEL`, `OCR_KEEP_ALIVE`, `OCR_DEVICE`, `OCR_USE_GPU`, `OCR_GPU_ID`, and `OCR_CACHE_DIR`.

Face recognition and liveness model settings use `kyc/models` as the canonical artifact directory:

| Variable | Default | Purpose |
| --- | --- | --- |
| `FACE_RECOGNITION_PROVIDER` | `auto` | Uses InsightFace when its artifact is present, otherwise falls back to the local ONNX provider. Set `insightface` to require InsightFace. |
| `INSIGHTFACE_MODEL_ROOT` | `kyc/models/insightface` | Root directory for InsightFace artifacts such as `models/buffalo_l`. |
| `INSIGHTFACE_MODEL_NAME` | `buffalo_l` | InsightFace model pack name. |
| `INSIGHTFACE_MODEL_URL` | official `buffalo_l` release URL | Optional zip URL override for `scripts/download_models.py`. |
| `FACE_RECOGNITION_MODEL` | `kyc/models/arcface.onnx` | Local ONNX fallback recognition model path. |
| `ANTI_SPOOF_ENABLED` | auto when model exists | Enables ONNX anti-spoofing inference for liveness frames and videos. |
| `ANTI_SPOOF_MODEL_PATH` | `kyc/models/anti_spoof.onnx` | Vetted anti-spoofing ONNX artifact path. |
| `ANTI_SPOOF_MODEL_URL` | MiniFASNet-V2 ONNX URL | Optional setup-time URL override used by `scripts/download_models.py` to fetch the anti-spoofing artifact. |
| `ANTI_SPOOF_MODEL_SHA256` | default artifact checksum | Expected SHA-256 checksum for the anti-spoofing download. |
| `ANTI_SPOOF_LIVE_THRESHOLD` | `0.65` | Minimum live probability required to avoid a spoof result. |
| `ANTI_SPOOF_INPUT_SIZE` | `80,80` | Anti-spoofing model input size, configurable for the chosen artifact. |
| `ANTI_SPOOF_LIVE_INDEX` | `0` | Class index treated as live after softmax. |
| `ANTI_SPOOF_CROP_SCALE` | `2.7` | Face crop scale used before resizing for anti-spoofing inference. |
| `ANTI_SPOOF_COLOR_ORDER` | `bgr` | Input channel order for anti-spoofing preprocessing; set `rgb` for models that require RGB. |

## Operational Notes

- The gateway reads the full request body up to `OCR_MAX_FILE_MB` before forwarding it upstream.
- Accuracy-first defaults are intentionally conservative for single VPS/container deployments: one active OCR request, a small queue, `accurate` mode, retry enabled, and a 120 second upstream timeout.
- For a faster optional profile, set `GATEWAY_DEFAULT_ACCURACY_MODE=fast` and `GATEWAY_DEFAULT_RETRY=false`.
- Upstream routing is round-robin when `GATEWAY_OCR_UPSTREAMS` contains more than one URL.
- The gateway forwards `Content-Type`, `Accept`, and `User-Agent` request headers.
- Do not expose the OCR service port directly in production; route traffic through the gateway.
- Set `GATEWAY_MAX_ACTIVE` close to the number of OCR workers or available GPU/CPU capacity to avoid overloading the OCR service.
