import json
import os
import re
from pathlib import Path

class DocumentConfigError(ValueError):
    pass


class DocumentRegistry:
    def __init__(self, config_path=None):
        default_config_path = Path(__file__).resolve().parents[1] / "config" / "document_types.yaml"
        self.config_path = Path(config_path or os.environ.get("DOCUMENT_TYPES_CONFIG", default_config_path))
        self.document_types = self.load(self.config_path)

    def load(self, path):
        if not path.exists():
            raise DocumentConfigError(f"Document config not found: {path}")
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except Exception as error:
                raise DocumentConfigError("PyYAML is required for YAML document config.") from error
            data = yaml.safe_load(path.read_text()) or {}
        elif path.suffix.lower() == ".json":
            data = json.loads(path.read_text())
        else:
            raise DocumentConfigError("Document config must be .yaml, .yml, or .json")
        return self.validate(data)

    def validate(self, data):
        docs = data.get("document_types")
        if not isinstance(docs, list) or not docs:
            raise DocumentConfigError("document_types must be a non-empty list")
        registry = {}
        for doc in docs:
            if not isinstance(doc, dict):
                raise DocumentConfigError("Each document type must be an object")
            doc_id = doc.get("id")
            if not doc_id or not re.fullmatch(r"[a-z0-9_/-]+", doc_id):
                raise DocumentConfigError("Each document type needs a stable id")
            if doc_id in registry:
                raise DocumentConfigError(f"Duplicate document type id: {doc_id}")
            sides = doc.get("sides")
            if not isinstance(sides, list) or not sides or not all(isinstance(side, str) for side in sides):
                raise DocumentConfigError(f"{doc_id}: sides must be a non-empty string list")
            fields = doc.get("fields", [])
            if not isinstance(fields, list):
                raise DocumentConfigError(f"{doc_id}: fields must be a list")
            for field in fields:
                if not field.get("id") or not field.get("label"):
                    raise DocumentConfigError(f"{doc_id}: every field needs id and label")
                side = field.get("side")
                if side and side not in sides:
                    raise DocumentConfigError(f"{doc_id}.{field.get('id')}: side {side} is not in sides")
                rules = field.get("rules", [])
                if not isinstance(rules, list):
                    raise DocumentConfigError(f"{doc_id}.{field.get('id')}: rules must be a list")
                for rule in rules:
                    if rule.get("type") not in {"regex", "anchor", "date_parts", "enum"}:
                        raise DocumentConfigError(f"{doc_id}.{field.get('id')}: unsupported rule type")
            registry[doc_id] = doc
        return registry

    def get(self, doc_id):
        return self.document_types.get(doc_id)

    def public_types(self):
        public = {}
        for doc_id, doc in self.document_types.items():
            public[doc_id] = {
                "label": doc.get("label", doc_id),
                "country": doc.get("country"),
                "category": doc.get("category"),
                "sides": doc.get("sides", []),
                "fields": doc.get("profile_fields", []),
            }
        return public


DOCUMENT_REGISTRY = DocumentRegistry()
DOCUMENT_TYPES = DOCUMENT_REGISTRY.public_types()
