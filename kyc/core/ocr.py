import mimetypes
import os
from pathlib import Path

from kyc.core.registry import DOCUMENT_TYPES

class OCRGatewayClient:
    def __init__(self, base_url=None, api_key=None, timeout_seconds=None):
        self.base_url = (base_url or os.environ.get("OCR_GATEWAY_URL") or "http://localhost:8000").rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OCR_GATEWAY_API_KEY") or os.environ.get("GATEWAY_API_KEY")
        self.timeout_seconds = float(timeout_seconds or os.environ.get("OCR_GATEWAY_TIMEOUT_SECONDS", 90))
        self.send_document_type = os.environ.get("OCR_GATEWAY_SEND_DOCUMENT_TYPE", "").lower() in {"1", "true", "yes"}

    def extract(self, image_path, document_type=None, content_type=None, filename=None):
        try:
            import requests
        except Exception as error:
            raise ValueError("requests is required to call the OCR HTTP gateway") from error

        path = Path(image_path)
        upload_name = filename or path.name
        upload_content_type = content_type or mimetypes.guess_type(upload_name)[0] or "application/octet-stream"
        params = {"values_only": "true"}
        if document_type and self.send_document_type:
            params["document_type"] = document_type
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        try:
            with path.open("rb") as handle:
                response = requests.post(
                    f"{self.base_url}/ocr",
                    params=params,
                    files={"file": (upload_name, handle, upload_content_type)},
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
        except requests.RequestException as error:
            raise ValueError(f"OCR HTTP gateway request failed: {error}") from error

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type.lower():
            try:
                body = response.json()
            except ValueError:
                body = {"raw_body": response.text}
        else:
            body = {"raw_body": response.text}

        if not response.ok:
            message = body.get("error") if isinstance(body, dict) else None
            raise ValueError(message or f"OCR HTTP gateway returned {response.status_code}")

        return {
            "engine": "http_gateway",
            "upstream": self.base_url,
            "status_code": response.status_code,
            "response": body,
        }


class OCRProfileMapper:
    DOCUMENT_TYPE_ALIASES = {
        "nepali_national_id": "national_id",
        "nepali_citizenship_front_back": "national_id",
        "nepali_citizenship_mixed_language": "national_id",
        "nepali_citizenship_old_front": "national_id",
        "generic_devanagari_document": "other_government_id",
        "passport": "passport",
        "driving_license": "driving_license",
        "voter_id": "voter_id",
        "national_id": "national_id",
    }
    FIELD_ALIASES = {
        "full_name": ["full_name", "name", "given_name", "applicant_name"],
        "date_of_birth": ["date_of_birth", "dob", "birth_date"],
        "nationality": ["nationality", "country", "citizenship"],
        "address": ["address", "residential_address", "permanent_address"],
        "document_number": ["document_number", "id_number", "passport_number", "license_number", "citizenship_number", "number"],
        "issue_date": ["issue_date", "issued_date", "date_of_issue"],
        "expiry_date": ["expiry_date", "expiration_date", "date_of_expiry", "valid_until"],
    }

    def map(self, gateway_result, document_type=None):
        response = gateway_result.get("response", {}) if isinstance(gateway_result, dict) else {}
        values = self.extract_values(response)
        suggested = {}
        for profile_field, aliases in self.FIELD_ALIASES.items():
            value = self.first_value(values, aliases)
            if value:
                suggested[profile_field] = str(value).strip()
        resolved_document_type = self.resolve_document_type(gateway_result, fallback=document_type)
        if resolved_document_type:
            suggested["document_type"] = resolved_document_type
        return suggested

    def resolve_document_type(self, gateway_result, fallback=None):
        response = gateway_result.get("response", {}) if isinstance(gateway_result, dict) else {}
        detected = None
        if isinstance(response, dict):
            detected = response.get("document_type")
            meta = response.get("meta")
            if not detected and isinstance(meta, dict):
                detected = meta.get("document_type")
        return self.normalize_document_type(detected) or self.normalize_document_type(fallback)

    def uses_fallback(self, gateway_result):
        response = gateway_result.get("response", {}) if isinstance(gateway_result, dict) else {}
        if not isinstance(response, dict):
            return True
        detected = response.get("document_type")
        meta = response.get("meta")
        if not detected and isinstance(meta, dict):
            detected = meta.get("document_type")
        return self.normalize_document_type(detected) is None

    def normalize_document_type(self, document_type):
        if not document_type:
            return None
        document_type = str(document_type).strip()
        if not document_type or document_type == "unknown":
            return None
        return self.DOCUMENT_TYPE_ALIASES.get(document_type, document_type if document_type in DOCUMENT_TYPES else None)

    def extract_values(self, body):
        if not isinstance(body, dict):
            return {}
        for key in ("values", "fields", "data", "extracted", "result"):
            nested = body.get(key)
            if isinstance(nested, dict):
                return self.flatten_values(nested)
        return self.flatten_values(body)

    def flatten_values(self, values):
        flat = {}
        for key, value in values.items():
            normalized_key = str(key).lower().strip()
            if isinstance(value, dict):
                if "value" in value:
                    flat[normalized_key] = value.get("value")
                elif "text" in value:
                    flat[normalized_key] = value.get("text")
                else:
                    for child_key, child_value in self.flatten_values(value).items():
                        flat[f"{normalized_key}.{child_key}"] = child_value
            else:
                flat[normalized_key] = value
        return flat

    def first_value(self, values, aliases):
        for alias in aliases:
            alias = alias.lower()
            if values.get(alias):
                return values[alias]
        for key, value in values.items():
            if value and any(key.endswith(f".{alias}") for alias in aliases):
                return value
        return None


