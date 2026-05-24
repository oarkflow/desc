import base64
import hashlib
import json
import mimetypes
import os
import random
import re
import sqlite3
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from kyc.blink import APILivenessDetector


class DocumentConfigError(ValueError):
    pass


class DocumentRegistry:
    def __init__(self, config_path=None):
        default_config_path = Path(__file__).resolve().parent / "config" / "document_types.yaml"
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

CHALLENGE_ACTIONS = ["blink", "turn_left", "turn_right", "look_center"]
DEMO_PROFILE = {
    "full_name": "Aarav Shrestha",
    "date_of_birth": "1994-04-18",
    "nationality": "Nepal",
    "address": "Ward 10, Kathmandu, Nepal",
    "phone": "+977-9800000000",
    "email": "aarav.demo@example.com",
    "document_type": "national_id",
    "document_number": "NP-DEMO-102938",
    "issue_date": "2020-01-15",
    "expiry_date": "2030-01-15",
}


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def row_to_dict(row):
    return dict(row) if row else None


def hash_secret(secret):
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def normalize_vector(vector):
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0:
        return array.tolist()
    return (array / norm).astype(float).tolist()


def cosine_similarity(vector_a, vector_b):
    a = np.asarray(vector_a, dtype=np.float32)
    b = np.asarray(vector_b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def decode_data_url(data_url):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return base64.b64decode(data_url)


class LocalEvidenceStorage:
    def __init__(self, root="evidence"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_bytes(self, session_id, category, filename, data):
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename or "upload.bin")
        session_dir = self.root / session_id / category
        session_dir.mkdir(parents=True, exist_ok=True)
        unique_name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{safe_name}"
        path = session_dir / unique_name
        path.write_bytes(data)
        return {
            "path": str(path),
            "filename": safe_name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }

    def delete_session(self, session_id):
        session_dir = self.root / session_id
        if not session_dir.exists():
            return
        for child in sorted(session_dir.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        session_dir.rmdir()


class KYCRepository:
    def __init__(self, db_path="kyc.db"):
        self.db_path = db_path
        self.init_db()

    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tenants (
                    id TEXT PRIMARY KEY,
                    slug TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    settings_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tenant_api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    key_hash TEXT NOT NULL UNIQUE,
                    label TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(tenant_id) REFERENCES tenants(id)
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT,
                    session_token TEXT,
                    applicant_status TEXT NOT NULL,
                    review_status TEXT NOT NULL,
                    challenge_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    submitted_at TEXT
                );
                CREATE TABLE IF NOT EXISTS profiles (
                    session_id TEXT PRIMARY KEY,
                    tenant_id TEXT,
                    full_name TEXT NOT NULL,
                    date_of_birth TEXT NOT NULL,
                    nationality TEXT NOT NULL,
                    address TEXT NOT NULL,
                    phone TEXT,
                    email TEXT,
                    document_type TEXT NOT NULL,
                    document_number TEXT NOT NULL,
                    issue_date TEXT,
                    expiry_date TEXT,
                    extra_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    tenant_id TEXT,
                    document_type TEXT NOT NULL,
                    side TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    content_type TEXT,
                    sha256 TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    normalized_json TEXT NOT NULL DEFAULT '{}',
                    risk_status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS selfies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    tenant_id TEXT,
                    file_path TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    content_type TEXT,
                    sha256 TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS liveness_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    tenant_id TEXT,
                    file_path TEXT,
                    sha256 TEXT,
                    result_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS face_matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    tenant_id TEXT,
                    score REAL,
                    status TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_id TEXT,
                    vector_json TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    quality_score REAL,
                    face_box_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS face_search_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    query_embedding_id INTEGER,
                    candidate_embedding_id INTEGER,
                    candidate_session_id TEXT,
                    score REAL NOT NULL,
                    rank INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    tenant_id TEXT,
                    event_type TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            self.migrate_schema(conn)
        self.ensure_default_tenant()

    def migrate_schema(self, conn):
        required_columns = {
            "sessions": {"tenant_id": "TEXT", "session_token": "TEXT"},
            "profiles": {"tenant_id": "TEXT"},
            "documents": {"tenant_id": "TEXT"},
            "selfies": {"tenant_id": "TEXT"},
            "liveness_checks": {"tenant_id": "TEXT"},
            "face_matches": {"tenant_id": "TEXT"},
            "audit_events": {"tenant_id": "TEXT"},
        }
        for table, columns in required_columns.items():
            existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            for name, definition in columns.items():
                if name not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_token ON sessions(session_token)")

    def ensure_default_tenant(self):
        slug = os.environ.get("DEFAULT_TENANT_SLUG", "demo")
        tenant = self.get_tenant_by_slug(slug)
        if not tenant:
            tenant = self.create_tenant(slug, os.environ.get("DEFAULT_TENANT_NAME", "Demo Tenant"))
        api_key = os.environ.get("DEFAULT_TENANT_API_KEY", "dev-tenant-key")
        if api_key and not self.find_api_key(api_key):
            self.add_api_key(tenant["id"], api_key, "default")
        self.backfill_default_tenant(tenant["id"])

    def backfill_default_tenant(self, tenant_id):
        with self.connect() as conn:
            for table in ("sessions", "profiles", "documents", "selfies", "liveness_checks", "face_matches", "audit_events"):
                conn.execute(f"UPDATE {table} SET tenant_id = ? WHERE tenant_id IS NULL", (tenant_id,))
            rows = conn.execute("SELECT id FROM sessions WHERE session_token IS NULL OR session_token = ''").fetchall()
            for row in rows:
                conn.execute("UPDATE sessions SET session_token = ? WHERE id = ?", (uuid.uuid4().hex + uuid.uuid4().hex, row["id"]))

    def default_tenant_id(self):
        tenant = self.get_tenant_by_slug(os.environ.get("DEFAULT_TENANT_SLUG", "demo"))
        if not tenant:
            tenant = self.create_tenant(os.environ.get("DEFAULT_TENANT_SLUG", "demo"), os.environ.get("DEFAULT_TENANT_NAME", "Demo Tenant"))
        return tenant["id"]

    def create_tenant(self, slug, name, settings=None, status="active"):
        tenant_id = uuid.uuid4().hex
        ts = now_iso()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO tenants(id, slug, name, status, settings_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (tenant_id, slug, name, status, json.dumps(settings or {}), ts, ts),
            )
        return self.get_tenant(tenant_id)

    def get_tenant(self, tenant_id):
        with self.connect() as conn:
            return row_to_dict(conn.execute("SELECT * FROM tenants WHERE id = ?", (tenant_id,)).fetchone())

    def get_tenant_by_slug(self, slug):
        with self.connect() as conn:
            return row_to_dict(conn.execute("SELECT * FROM tenants WHERE slug = ?", (slug,)).fetchone())

    def add_api_key(self, tenant_id, api_key, label="integration", status="active"):
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO tenant_api_keys(tenant_id, key_hash, label, status, created_at) VALUES (?, ?, ?, ?, ?)",
                (tenant_id, hash_secret(api_key), label, status, now_iso()),
            )

    def find_api_key(self, api_key):
        with self.connect() as conn:
            return row_to_dict(
                conn.execute(
                    """
                    SELECT k.*, t.status AS tenant_status
                    FROM tenant_api_keys k
                    JOIN tenants t ON t.id = k.tenant_id
                    WHERE k.key_hash = ?
                    """,
                    (hash_secret(api_key or ""),),
                ).fetchone()
            )

    def authenticate_api_key(self, api_key):
        key = self.find_api_key(api_key)
        if not key or key["status"] != "active" or key["tenant_status"] != "active":
            raise ValueError("Invalid tenant API key")
        return self.get_tenant(key["tenant_id"])

    def tenant_id_for_session(self, session_id):
        session = self.get_session(session_id)
        if not session:
            raise ValueError("Unknown KYC session")
        return session.get("tenant_id")

    def audit(self, session_id, event_type, details=None):
        tenant_id = self.tenant_id_for_session(session_id)
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO audit_events(session_id, tenant_id, event_type, details_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, tenant_id, event_type, json.dumps(details or {}), now_iso()),
            )

    def create_session(self, tenant_id=None):
        tenant_id = tenant_id or self.default_tenant_id()
        session_id = uuid.uuid4().hex
        session_token = uuid.uuid4().hex + uuid.uuid4().hex
        challenge = random.sample(CHALLENGE_ACTIONS, 3)
        ts = now_iso()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO sessions(id, tenant_id, session_token, applicant_status, review_status, challenge_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, tenant_id, session_token, "draft", "pending_review", json.dumps(challenge), ts, ts),
            )
        self.audit(session_id, "session_created", {"challenge": challenge})
        return self.get_session(session_id)

    def create_demo_session(self, tenant_id=None):
        tenant_id = tenant_id or self.default_tenant_id()
        session_id = uuid.uuid4().hex
        session_token = uuid.uuid4().hex + uuid.uuid4().hex
        challenge = ["look_center", "blink", "turn_left", "turn_right"]
        ts = now_iso()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO sessions(id, tenant_id, session_token, applicant_status, review_status, challenge_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, tenant_id, session_token, "draft", "pending_review", json.dumps(challenge), ts, ts),
            )
        self.save_profile(session_id, DEMO_PROFILE)
        self.audit(session_id, "demo_session_created", {"challenge": challenge, "profile": DEMO_PROFILE})
        return self.get_session(session_id)

    def get_session(self, session_id):
        with self.connect() as conn:
            return row_to_dict(conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone())

    def get_session_by_token(self, session_token):
        with self.connect() as conn:
            return row_to_dict(conn.execute("SELECT * FROM sessions WHERE session_token = ?", (session_token,)).fetchone())

    def verify_session_token(self, session_id, session_token):
        session = self.get_session(session_id)
        if not session or not session_token or session["session_token"] != session_token:
            raise ValueError("Invalid KYC session token")
        return session

    def save_profile(self, session_id, data):
        tenant_id = self.tenant_id_for_session(session_id)
        doc_type = data.get("document_type", "")
        if doc_type not in DOCUMENT_TYPES:
            raise ValueError("Unsupported document type")
        required = ["full_name", "date_of_birth", "nationality", "address", "document_number"]
        missing = [field for field in required if not data.get(field)]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO profiles(session_id, tenant_id, full_name, date_of_birth, nationality, address, phone, email,
                document_type, document_number, issue_date, expiry_date, extra_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    tenant_id=excluded.tenant_id,
                    full_name=excluded.full_name,
                    date_of_birth=excluded.date_of_birth,
                    nationality=excluded.nationality,
                    address=excluded.address,
                    phone=excluded.phone,
                    email=excluded.email,
                    document_type=excluded.document_type,
                    document_number=excluded.document_number,
                    issue_date=excluded.issue_date,
                    expiry_date=excluded.expiry_date,
                    extra_json=excluded.extra_json
                """,
                (
                    session_id,
                    tenant_id,
                    data.get("full_name", "").strip(),
                    data.get("date_of_birth", "").strip(),
                    data.get("nationality", "").strip(),
                    data.get("address", "").strip(),
                    data.get("phone", "").strip(),
                    data.get("email", "").strip(),
                    doc_type,
                    data.get("document_number", "").strip(),
                    data.get("issue_date", "").strip(),
                    data.get("expiry_date", "").strip(),
                    json.dumps({}),
                ),
            )
        self.audit(session_id, "profile_saved", {"document_type": doc_type})

    def add_document(self, session_id, document_type, side, file_info, content_type, gateway_result=None):
        tenant_id = self.tenant_id_for_session(session_id)
        if document_type not in DOCUMENT_TYPES:
            raise ValueError("Unsupported document type")
        if side not in DOCUMENT_TYPES[document_type]["sides"]:
            raise ValueError("Unsupported document side")
        gateway_result = gateway_result or {}
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO documents(session_id, tenant_id, document_type, side, file_path, original_filename, content_type,
                sha256, size, normalized_json, risk_status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    tenant_id,
                    document_type,
                    side,
                    file_info["path"],
                    file_info["filename"],
                    content_type,
                    file_info["sha256"],
                    file_info["size"],
                    json.dumps({"gateway": gateway_result}),
                    "uploaded",
                    now_iso(),
                ),
            )
        self.audit(session_id, "document_uploaded", {"side": side, "document_type": document_type})

    def add_selfie(self, session_id, file_info, content_type):
        tenant_id = self.tenant_id_for_session(session_id)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO selfies(session_id, tenant_id, file_path, original_filename, content_type, sha256, size, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, tenant_id, file_info["path"], file_info["filename"], content_type, file_info["sha256"], file_info["size"], now_iso()),
            )
        self.audit(session_id, "selfie_uploaded", {"sha256": file_info["sha256"]})

    def add_liveness(self, session_id, file_info, result):
        tenant_id = self.tenant_id_for_session(session_id)
        status = result["risk_status"]
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO liveness_checks(session_id, tenant_id, file_path, sha256, result_json, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, tenant_id, file_info.get("path"), file_info.get("sha256"), json.dumps(result), status, now_iso()),
            )
        self.audit(session_id, "liveness_completed", {"status": status})

    def latest_liveness_with_file(self, session_id):
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM liveness_checks
                WHERE session_id = ? AND file_path IS NOT NULL AND file_path != ''
                ORDER BY id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            return row_to_dict(row)

    def update_liveness_result(self, check_id, result):
        with self.connect() as conn:
            conn.execute(
                "UPDATE liveness_checks SET result_json = ?, status = ? WHERE id = ?",
                (json.dumps(result), result["risk_status"], check_id),
            )

    def add_face_match(self, session_id, result):
        tenant_id = self.tenant_id_for_session(session_id)
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO face_matches(session_id, tenant_id, score, status, details_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, tenant_id, result.get("score"), result["status"], json.dumps(result), now_iso()),
            )
        self.audit(session_id, "face_match_scored", {"status": result["status"], "score": result.get("score")})

    def add_face_embedding(self, session_id, source_type, vector, provider, model_version, quality_score=None, face_box=None, source_id=None, status="active"):
        tenant_id = self.tenant_id_for_session(session_id)
        normalized = normalize_vector(vector)
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO face_embeddings(tenant_id, session_id, source_type, source_id, vector_json, provider,
                model_version, quality_score, face_box_json, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant_id,
                    session_id,
                    source_type,
                    str(source_id) if source_id is not None else None,
                    json.dumps(normalized),
                    provider,
                    model_version,
                    quality_score,
                    json.dumps(face_box or {}),
                    status,
                    now_iso(),
                ),
            )
            embedding_id = cursor.lastrowid
        self.audit(session_id, "face_embedding_stored", {"source_type": source_type, "embedding_id": embedding_id, "provider": provider})
        return embedding_id

    def latest_face_embedding(self, session_id, source_types):
        placeholders = ", ".join("?" for _ in source_types)
        with self.connect() as conn:
            row = conn.execute(
                f"""
                SELECT * FROM face_embeddings
                WHERE session_id = ? AND source_type IN ({placeholders}) AND status = 'active'
                ORDER BY id DESC
                LIMIT 1
                """,
                [session_id, *source_types],
            ).fetchone()
            return row_to_dict(row)

    def list_face_embeddings(self, tenant_id, exclude_session_id=None, source_types=None):
        params = [tenant_id]
        where = "tenant_id = ? AND status = 'active'"
        if exclude_session_id:
            where += " AND session_id != ?"
            params.append(exclude_session_id)
        if source_types:
            where += " AND source_type IN (" + ", ".join("?" for _ in source_types) + ")"
            params.extend(source_types)
        with self.connect() as conn:
            rows = conn.execute(f"SELECT * FROM face_embeddings WHERE {where} ORDER BY id DESC", params).fetchall()
            return [row_to_dict(row) for row in rows]

    def store_face_search_results(self, session_id, query_embedding_id, matches):
        tenant_id = self.tenant_id_for_session(session_id)
        with self.connect() as conn:
            conn.execute("DELETE FROM face_search_results WHERE session_id = ? AND query_embedding_id = ?", (session_id, query_embedding_id))
            for rank, match in enumerate(matches, start=1):
                conn.execute(
                    """
                    INSERT INTO face_search_results(tenant_id, session_id, query_embedding_id, candidate_embedding_id,
                    candidate_session_id, score, rank, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tenant_id,
                        session_id,
                        query_embedding_id,
                        match["embedding_id"],
                        match["session_id"],
                        match["score"],
                        rank,
                        now_iso(),
                    ),
                )
        if matches:
            self.audit(session_id, "face_search_completed", {"query_embedding_id": query_embedding_id, "matches": len(matches)})

    def get_case(self, session_id, tenant_id=None):
        with self.connect() as conn:
            session = row_to_dict(conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone())
            if not session:
                raise ValueError("Unknown KYC session")
            if tenant_id and session.get("tenant_id") != tenant_id:
                raise ValueError("Unknown KYC session")
            tenant = row_to_dict(conn.execute("SELECT * FROM tenants WHERE id = ?", (session.get("tenant_id"),)).fetchone())
            profile = row_to_dict(conn.execute("SELECT * FROM profiles WHERE session_id = ?", (session_id,)).fetchone())
            documents = [row_to_dict(row) for row in conn.execute("SELECT * FROM documents WHERE session_id = ? ORDER BY id", (session_id,))]
            selfies = [row_to_dict(row) for row in conn.execute("SELECT * FROM selfies WHERE session_id = ? ORDER BY id", (session_id,))]
            liveness = [row_to_dict(row) for row in conn.execute("SELECT * FROM liveness_checks WHERE session_id = ? ORDER BY id", (session_id,))]
            matches = [row_to_dict(row) for row in conn.execute("SELECT * FROM face_matches WHERE session_id = ? ORDER BY id", (session_id,))]
            embeddings = [row_to_dict(row) for row in conn.execute("SELECT * FROM face_embeddings WHERE session_id = ? ORDER BY id", (session_id,))]
            search_results = [row_to_dict(row) for row in conn.execute("SELECT * FROM face_search_results WHERE session_id = ? ORDER BY rank", (session_id,))]
            audit = [row_to_dict(row) for row in conn.execute("SELECT * FROM audit_events WHERE session_id = ? ORDER BY id", (session_id,))]
        return {
            "tenant": tenant,
            "session": session,
            "profile": profile,
            "documents": documents,
            "selfies": selfies,
            "liveness_checks": liveness,
            "face_matches": matches,
            "face_embeddings": embeddings,
            "face_search_results": search_results,
            "audit_events": audit,
        }

    def delete_evidence(self, session_id):
        with self.connect() as conn:
            conn.execute("DELETE FROM documents WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM selfies WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM liveness_checks WHERE session_id = ?", (session_id,))
        self.audit(session_id, "evidence_deleted", {})



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
        params = {"values_only": "false", "fields_only": "true"}
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


class FaceRecognitionProvider:
    provider_name = "base"
    model_version = "none"

    def detect_faces(self, image):
        raise NotImplementedError

    def extract_embedding(self, image, face_box):
        raise NotImplementedError

    def compare(self, embedding_a, embedding_b):
        return cosine_similarity(embedding_a, embedding_b)


class LocalONNXFaceRecognitionProvider(FaceRecognitionProvider):
    provider_name = "local_onnx"

    def __init__(self, model_path=None):
        self.model_path = Path(model_path or os.environ.get("FACE_RECOGNITION_MODEL", "models/arcface.onnx"))
        self.model_version = self.model_path.name
        self.session = None
        cascade_dir = Path(cv2.data.haarcascades)
        self.face_cascade = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_frontalface_default.xml"))
        if self.model_path.exists():
            try:
                import onnxruntime as ort
            except Exception as error:
                self.load_error = f"onnxruntime is required when FACE_RECOGNITION_MODEL is configured: {error}"
            else:
                self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
                self.load_error = None
        else:
            self.load_error = "Face recognition model is not configured."

    @property
    def available(self):
        return self.session is not None

    def detect_faces(self, image):
        if image is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        results = []
        h, w = image.shape[:2]
        for x, y, face_w, face_h in faces:
            area = float(face_w * face_h)
            quality = min(1.0, area / max(float(w * h) * 0.12, 1.0))
            results.append({"x": int(x), "y": int(y), "width": int(face_w), "height": int(face_h), "quality": round(quality, 4)})
        return sorted(results, key=lambda item: item["width"] * item["height"], reverse=True)

    def extract_embedding(self, image, face_box):
        if not self.available:
            return None
        crop = self.crop_face(image, face_box)
        blob = self.preprocess(crop)
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: blob})[0]
        return normalize_vector(output.reshape(-1))

    def crop_face(self, image, face_box):
        x = max(int(face_box["x"]), 0)
        y = max(int(face_box["y"]), 0)
        w = max(int(face_box["width"]), 1)
        h = max(int(face_box["height"]), 1)
        pad = int(max(w, h) * 0.18)
        y1 = max(y - pad, 0)
        y2 = min(y + h + pad, image.shape[0])
        x1 = max(x - pad, 0)
        x2 = min(x + w + pad, image.shape[1])
        return image[y1:y2, x1:x2]

    def preprocess(self, crop):
        resized = cv2.resize(crop, (112, 112))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        normalized = (rgb - 127.5) / 127.5
        return np.transpose(normalized, (2, 0, 1))[None, :, :, :]


class FaceMatchService:
    def __init__(self, repository=None, provider=None):
        self.repository = repository
        self.provider = provider or LocalONNXFaceRecognitionProvider()

    def score(self, document_path, selfie_path):
        doc_result = self.extract_embedding_from_file(document_path, "document")
        selfie_result = self.extract_embedding_from_file(selfie_path, "selfie")
        if not doc_result.get("embedding") or not selfie_result.get("embedding"):
            return {
                "score": None,
                "status": "needs_manual_review",
                "reason": doc_result.get("reason") or selfie_result.get("reason") or "Could not extract comparable face embeddings.",
                "provider": self.provider.provider_name,
                "model_version": self.provider.model_version,
            }

        return self.compare_embeddings(doc_result["embedding"], selfie_result["embedding"])

    def enroll_source(self, session_id, source_type, image_path, source_id=None):
        result = self.extract_embedding_from_file(image_path, source_type)
        if not self.repository or not result.get("embedding"):
            return result
        embedding_id = self.repository.add_face_embedding(
            session_id,
            source_type,
            result["embedding"],
            result["provider"],
            result["model_version"],
            quality_score=result.get("quality_score"),
            face_box=result.get("face_box"),
            source_id=source_id,
        )
        return {**result, "embedding_id": embedding_id}

    def compare_session(self, session_id):
        if not self.repository:
            return {"score": None, "status": "needs_manual_review", "reason": "Face repository is not configured."}
        document = self.repository.latest_face_embedding(session_id, ["document"])
        selfie = self.repository.latest_face_embedding(session_id, ["selfie", "liveness"])
        if not document or not selfie:
            return {
                "score": None,
                "status": "needs_manual_review",
                "reason": "Missing document or selfie face embedding.",
                "provider": self.provider.provider_name,
                "model_version": self.provider.model_version,
            }
        result = self.compare_embeddings(json.loads(document["vector_json"]), json.loads(selfie["vector_json"]))
        result["document_embedding_id"] = document["id"]
        result["selfie_embedding_id"] = selfie["id"]
        return result

    def search_tenant_gallery(self, session_id, limit=5):
        if not self.repository:
            return []
        session = self.repository.get_session(session_id)
        query = self.repository.latest_face_embedding(session_id, ["selfie", "liveness", "document"])
        if not session or not query:
            return []
        query_vector = json.loads(query["vector_json"])
        candidates = self.repository.list_face_embeddings(
            session["tenant_id"],
            exclude_session_id=session_id,
            source_types=["selfie", "liveness", "document"],
        )
        matches = []
        for candidate in candidates:
            score = cosine_similarity(query_vector, json.loads(candidate["vector_json"]))
            if score >= 0.45:
                matches.append(
                    {
                        "embedding_id": candidate["id"],
                        "session_id": candidate["session_id"],
                        "score": round(score, 4),
                        "source_type": candidate["source_type"],
                    }
                )
        matches = sorted(matches, key=lambda item: item["score"], reverse=True)[:limit]
        self.repository.store_face_search_results(session_id, query["id"], matches)
        return matches

    def compare_embeddings(self, embedding_a, embedding_b):
        score = round(self.provider.compare(embedding_a, embedding_b), 4)
        if score >= 0.55:
            status = "pass"
        elif score >= 0.45:
            status = "needs_manual_review"
        else:
            status = "warn"
        return {
            "score": score,
            "status": status,
            "provider": self.provider.provider_name,
            "model_version": self.provider.model_version,
            "thresholds": {"pass": 0.55, "review": 0.45},
        }

    def extract_embedding_from_file(self, image_path, source_type):
        image = cv2.imread(str(image_path))
        if image is None:
            return {"status": "needs_manual_review", "reason": "Image could not be read.", "source_type": source_type}
        faces = self.provider.detect_faces(image)
        if not faces:
            return {"status": "needs_manual_review", "reason": "No face detected.", "source_type": source_type}
        if not getattr(self.provider, "available", True):
            return {
                "status": "needs_manual_review",
                "reason": getattr(self.provider, "load_error", None) or "Face recognition model is not configured.",
                "source_type": source_type,
                "provider": self.provider.provider_name,
                "model_version": self.provider.model_version,
                "face_box": faces[0],
                "quality_score": faces[0].get("quality"),
            }
        embedding = self.provider.extract_embedding(image, faces[0])
        if embedding is None:
            return {"status": "needs_manual_review", "reason": "Embedding extraction failed.", "source_type": source_type}
        return {
            "status": "active",
            "source_type": source_type,
            "embedding": embedding,
            "provider": self.provider.provider_name,
            "model_version": self.provider.model_version,
            "face_box": faces[0],
            "quality_score": faces[0].get("quality"),
        }


class LivenessService:
    def __init__(self, detector=None):
        self.detector = detector or APILivenessDetector()
        self.session_states = {}

    def analyze_frame_bytes(self, data, session_id=None, challenge=None):
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid image frame")
        blink, value, face = self.detector.detect_blink(image)
        points = self.detector.detect_face_points(image) if face else None
        center_x = None
        head_position = "unknown"
        if points:
            xs = [point[0] for point in points]
            center_x = sum(xs) / len(xs)
            if center_x < 0.43:
                head_position = "left"
            elif center_x > 0.57:
                head_position = "right"
            else:
                head_position = "center"

        result = {
            "blink_detected": blink,
            "eye_value": value,
            "face_detected": face,
            "center_x": round(center_x, 4) if center_x is not None else None,
            "head_position": head_position,
            "backend": self.detector.backend,
        }
        if session_id:
            result["liveness_state"] = self.update_session_state(session_id, result, challenge or [])
        return result

    def update_session_state(self, session_id, frame_result, challenge):
        state = self.session_states.setdefault(
            session_id,
            {
                "frames": 0,
                "frames_with_face": 0,
                "blink_count": 0,
                "centers": [],
                "completed": {action: False for action in challenge},
            },
        )
        for action in challenge:
            state["completed"].setdefault(action, False)

        state["frames"] += 1
        if frame_result["face_detected"]:
            state["frames_with_face"] += 1
        if frame_result["blink_detected"]:
            state["blink_count"] += 1
            state["completed"]["blink"] = True
        if frame_result["center_x"] is not None:
            state["centers"].append(frame_result["center_x"])
        if frame_result["head_position"] == "left":
            state["completed"]["turn_left"] = True
        if frame_result["head_position"] == "right":
            state["completed"]["turn_right"] = True
        if frame_result["head_position"] == "center":
            state["completed"]["look_center"] = True

        face_detection_rate = state["frames_with_face"] / max(state["frames"], 1)
        center_range = max(state["centers"]) - min(state["centers"]) if len(state["centers"]) > 1 else 0
        if center_range > 0.12:
            state["completed"]["turn_left"] = True
            state["completed"]["turn_right"] = True

        completed = {action: state["completed"].get(action, False) for action in challenge}
        risk_status = self.score_state(completed, face_detection_rate)
        return {
            "risk_status": risk_status,
            "completed": completed,
            "blink_count": state["blink_count"],
            "face_detection_rate": round(face_detection_rate, 4),
            "movement_detected": center_range > 0.08,
            "frames_processed": state["frames"],
        }

    def score_state(self, completed, face_detection_rate):
        if completed and all(completed.values()) and face_detection_rate >= 0.5:
            return "pass"
        if face_detection_rate >= 0.35:
            return "warn"
        return "fail"

    def merge_results(self, video_result, live_result):
        challenge = live_result.get("challenge") or video_result.get("challenge") or []
        video_completed = video_result.get("completed") or {}
        live_completed = live_result.get("completed") or {}
        completed = {
            action: bool(video_completed.get(action) or live_completed.get(action))
            for action in challenge
        }
        face_detection_rate = max(
            float(video_result.get("face_detection_rate") or 0),
            float(live_result.get("face_detection_rate") or 0),
        )
        risk_status = self.score_state(completed, face_detection_rate)
        return {
            **video_result,
            "risk_status": risk_status,
            "challenge": challenge,
            "completed": completed,
            "blink_count": max(int(video_result.get("blink_count") or 0), int(live_result.get("blink_count") or 0)),
            "face_detection_rate": round(face_detection_rate, 4),
            "movement_detected": bool(video_result.get("movement_detected") or live_result.get("movement_detected")),
            "frames_processed": int(video_result.get("frames_processed") or 0) + int(live_result.get("frames_processed") or 0),
            "backend": live_result.get("backend") or video_result.get("backend"),
            "video_result": video_result,
            "live_frame_result": live_result,
        }

    def finalize_session(self, session_id, challenge):
        state = self.session_states.get(session_id)
        if not state:
            return {
                "risk_status": "needs_manual_review",
                "challenge": challenge,
                "completed": {action: False for action in challenge},
                "blink_count": 0,
                "face_detection_rate": 0,
                "movement_detected": False,
                "frames_processed": 0,
                "backend": self.detector.backend,
                "reason": "No live frame state was available; use recorded proof video for manual review.",
            }

        completed = {action: state["completed"].get(action, False) for action in challenge}
        face_detection_rate = state["frames_with_face"] / max(state["frames"], 1)
        center_range = max(state["centers"]) - min(state["centers"]) if len(state["centers"]) > 1 else 0
        result = {
            "risk_status": self.score_state(completed, face_detection_rate),
            "challenge": challenge,
            "completed": completed,
            "blink_count": state["blink_count"],
            "face_detection_rate": round(face_detection_rate, 4),
            "movement_detected": center_range > 0.08,
            "frames_processed": state["frames"],
            "backend": self.detector.backend,
        }
        self.session_states.pop(session_id, None)
        return result

    def analyze_video_file(self, path, challenge):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return {"risk_status": "fail", "error": "Cannot open liveness video", "challenge": challenge}

        self.detector.blink_count = 0
        self.detector.consecutive_low_ear = 0
        frame_count = 0
        frames_with_face = 0
        centers = []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        max_frames = int(min(fps * 12, 420))

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 3 != 0:
                continue
            blink, _, face = self.detector.detect_blink(frame)
            if face:
                frames_with_face += 1
                points = self.detector.detect_face_points(frame)
                if points:
                    xs = [point[0] for point in points]
                    centers.append(sum(xs) / len(xs))

        cap.release()
        face_detection_rate = frames_with_face / max(frame_count // 3, 1)
        center_range = max(centers) - min(centers) if len(centers) > 1 else 0
        movement_detected = center_range > 0.04
        checks = {
            "blink": self.detector.blink_count >= 1,
            "turn_left": movement_detected,
            "turn_right": movement_detected,
            "look_center": face_detection_rate >= 0.5,
        }
        completed = {action: checks.get(action, False) for action in challenge}
        passed = all(completed.values()) and face_detection_rate >= 0.4
        risk_status = "pass" if passed else "warn" if face_detection_rate >= 0.3 else "fail"
        return {
            "risk_status": risk_status,
            "challenge": challenge,
            "completed": completed,
            "blink_count": self.detector.blink_count,
            "face_detection_rate": round(face_detection_rate, 4),
            "movement_detected": movement_detected,
            "frames_processed": frame_count,
            "backend": self.detector.backend,
        }
