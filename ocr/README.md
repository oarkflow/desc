# KYC OCR Gateway

The gateway is a small Go HTTP service that fronts the Python OCR service. It adds API-key authentication, request size limits, queue/concurrency control, round-robin upstream selection, default OCR query options, health checks, and Prometheus-style metrics.

By default Docker Compose exposes the gateway on `http://localhost:8000` and keeps the OCR service private on the Docker network.

## Requirements

- Docker and Docker Compose for the recommended runtime.
- A configured `GATEWAY_API_KEY` for protected deployments.
- Input files must be images or PDFs accepted by the OCR service, sent as `multipart/form-data`.
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

The Document types editor includes a Regions section. Upload or drop an image, draw normalized field/photo/logo/hologram/stamp/signature/object regions, run an OCR overlay, and save the generated multi-region config with the same document type YAML.

`GET /region-editor` redirects to `/admin` for compatibility with older links.

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

- `file`: required image or PDF upload. PDFs are rasterized page-by-page and capped by `OCR_MAX_PDF_PAGES`.

Query parameters forwarded to the OCR service:

| Parameter | Default at gateway | Description |
| --- | --- | --- |
| `document_type` | auto-detect upstream | Optional document profile hint. Known profiles include `nepali_citizenship_old_front`, `nepali_citizenship_front_back`, `nepali_citizenship_mixed_language`, `nepali_national_id`, and `generic_devanagari_document`. |
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
curl -X POST "http://localhost:8000/ocr?document_type=nepali_national_id&values_only=false&accuracy_mode=accurate&retry=true" \
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
  "tamper_score": 0.18,
  "status": "suspicious",
  "flags": [
    {
      "code": "expected_object_missing",
      "message": "Expected object is missing: face",
      "severity": "high",
      "score": 0.3,
      "evidence": {
        "label": "face"
      }
    }
  ],
  "manual_review_required": true,
  "tamper": {
    "tamper_score": 0.18,
    "status": "suspicious",
    "flags": [],
    "manual_review_required": true,
    "checks": {
      "aggregate_strategy": "max_page_score",
      "worst_page": 1
    }
  },
  "meta": {
    "document_type": "nepali_national_id",
    "document_type_confidence": 0.3333,
    "page_count": 1
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
  "tamper_score": 0,
  "status": "genuine",
  "flags": [],
  "manual_review_required": false,
  "tamper": {
    "tamper_score": 0,
    "status": "genuine",
    "flags": [],
    "manual_review_required": false,
    "checks": {
      "aggregate_strategy": "max_page_score",
      "worst_page": 1
    }
  },
  "meta": {
    "document_type": "nepali_national_id",
    "document_type_confidence": 0.3333,
    "page_count": 1,
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
  "tamper_score": 0.18,
  "status": "suspicious",
  "flags": [],
  "manual_review_required": true,
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
  "tamper": {
    "tamper_score": 0.18,
    "status": "suspicious",
    "flags": [],
    "manual_review_required": true,
    "checks": {
      "aggregate_strategy": "max_page_score",
      "worst_page": 1
    }
  },
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
| `413` | PDF has more pages than `OCR_MAX_PDF_PAGES`. |
| `429` | OCR concurrency and queue are full. |
| `502` | Gateway could not create or complete an upstream OCR request. |
| `504` | Upstream OCR request timed out. |

The gateway passes upstream OCR response headers, status codes, and bodies through for completed upstream requests.

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

Document type YAML supports both single regions and multi-region lists. Existing keys such as `anchor_region`, `retry_region`, `tamper.field_regions.<field>`, `tamper.protected_regions.<name>`, and `tamper.expected_objects[].region` remain valid. For variants that need several allowed boxes, use `anchor_regions`, `retry_regions`, list values under tamper field/protected regions, and `tamper.expected_objects[].regions`.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `GATEWAY_PORT` | `8000` | Gateway listen port inside the container. Compose maps `${GATEWAY_PORT:-8000}` on the host to container port `8000`. |
| `GATEWAY_OCR_UPSTREAM` | `http://127.0.0.1:8001` | Single upstream OCR service URL. Docker Compose overrides this to `http://ocr:8000` inside the Compose network. |
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
| `OCR_TAMPER_ENABLED` | `true` | Enables tamper/risk scoring in OCR responses. |
| `OCR_TAMPER_MODE` | `standard` | Use `production` to require configured ML model paths at service startup. |
| `OCR_MAX_PDF_PAGES` | `5` | Maximum PDF pages rasterized per OCR request. |
| `OCR_YOLO_MODEL_PATH` | empty | Optional YOLO model path for document/photo/stamp/signature/seal detection. In Compose, place models under `ocr/models` and use `/app/models/<file>`. |
| `OCR_FACE_MODEL_PATH` | empty | Optional ONNX face model path validated by ONNX Runtime in production tamper mode. |
| `OCR_TAMPER_REVIEW_THRESHOLD` | `0.35` | Default score at or above which the response status becomes `suspicious`. |
| `OCR_TAMPER_REJECT_THRESHOLD` | `0.70` | Default score at or above which the response status becomes `likely_tampered`. |

OCR service settings are configured on the `ocr` Compose service. Common values include `OCR_WORKERS`, `OCR_LOG_LEVEL`, `OCR_KEEP_ALIVE`, `OCR_DEVICE`, `OCR_USE_GPU`, `OCR_GPU_ID`, and `OCR_CACHE_DIR`.

## Operational Notes

- The gateway reads the full request body up to `OCR_MAX_FILE_MB` before forwarding it upstream.
- Accuracy-first defaults are intentionally conservative for single VPS/container deployments: one active OCR request, a small queue, `accurate` mode, retry enabled, and a 120 second upstream timeout.
- For a faster optional profile, set `GATEWAY_DEFAULT_ACCURACY_MODE=fast` and `GATEWAY_DEFAULT_RETRY=false`.
- Upstream routing is round-robin when `GATEWAY_OCR_UPSTREAMS` contains more than one URL.
- The gateway forwards `Content-Type`, `Accept`, and `User-Agent` request headers.
- Tamper detection is a risk scoring layer, not a final yes/no verdict. Automatically approve only clean cases and send suspicious or likely-tampered cases to review.
- Production tamper mode requires `OCR_YOLO_MODEL_PATH` and `OCR_FACE_MODEL_PATH` to point at mounted model files; the service fails startup when they are missing or invalid.
- Do not expose the OCR service port directly in production; route traffic through the gateway.
- Set `GATEWAY_MAX_ACTIVE` close to the number of OCR workers or available GPU/CPU capacity to avoid overloading the OCR service.
