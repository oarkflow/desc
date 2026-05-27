import json
import os
import random
import sqlite3
import uuid

from kyc.core.constants import CHALLENGE_ACTIONS, DEMO_PROFILE
from kyc.core.registry import DOCUMENT_TYPES
from kyc.core.utils import hash_secret, normalize_vector, now_iso, row_to_dict

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



