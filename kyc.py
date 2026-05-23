import base64
import hashlib
import json
import os
import random
import re
import sqlite3
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from blink import APILivenessDetector


class DocumentConfigError(ValueError):
    pass


class DocumentRegistry:
    def __init__(self, config_path=None):
        self.config_path = Path(config_path or os.environ.get("DOCUMENT_TYPES_CONFIG", "config/document_types.yaml"))
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

APPLICANT_STATUSES = {"draft", "submitted", "resubmission_requested", "approved", "rejected"}
REVIEW_STATUSES = {
    "pending_review",
    "auto_checks_passed",
    "needs_review",
    "failed_checks",
    "manual_approved",
    "manual_rejected",
}
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
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    applicant_status TEXT NOT NULL,
                    review_status TEXT NOT NULL,
                    challenge_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    submitted_at TEXT
                );
                CREATE TABLE IF NOT EXISTS profiles (
                    session_id TEXT PRIMARY KEY,
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
                    file_path TEXT,
                    sha256 TEXT,
                    result_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS face_matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    score REAL,
                    status TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    note TEXT,
                    reviewer TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def audit(self, session_id, event_type, details=None):
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO audit_events(session_id, event_type, details_json, created_at) VALUES (?, ?, ?, ?)",
                (session_id, event_type, json.dumps(details or {}), now_iso()),
            )

    def create_session(self):
        session_id = uuid.uuid4().hex
        challenge = random.sample(CHALLENGE_ACTIONS, 3)
        ts = now_iso()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO sessions(id, applicant_status, review_status, challenge_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, "draft", "pending_review", json.dumps(challenge), ts, ts),
            )
        self.audit(session_id, "session_created", {"challenge": challenge})
        return self.get_session(session_id)

    def create_demo_session(self):
        session_id = uuid.uuid4().hex
        challenge = ["look_center", "blink", "turn_left", "turn_right"]
        ts = now_iso()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO sessions(id, applicant_status, review_status, challenge_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, "draft", "pending_review", json.dumps(challenge), ts, ts),
            )
        self.save_profile(session_id, DEMO_PROFILE)
        self.audit(session_id, "demo_session_created", {"challenge": challenge, "profile": DEMO_PROFILE})
        return self.get_session(session_id)

    def get_session(self, session_id):
        with self.connect() as conn:
            return row_to_dict(conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone())

    def update_status(self, session_id, applicant_status=None, review_status=None):
        session = self.get_session(session_id)
        if not session:
            raise ValueError("Unknown KYC session")
        applicant_status = applicant_status or session["applicant_status"]
        review_status = review_status or session["review_status"]
        if applicant_status not in APPLICANT_STATUSES or review_status not in REVIEW_STATUSES:
            raise ValueError("Invalid status")
        with self.connect() as conn:
            conn.execute(
                "UPDATE sessions SET applicant_status = ?, review_status = ?, updated_at = ? WHERE id = ?",
                (applicant_status, review_status, now_iso(), session_id),
            )
        self.audit(session_id, "status_updated", {"applicant_status": applicant_status, "review_status": review_status})

    def save_profile(self, session_id, data):
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
                INSERT INTO profiles(session_id, full_name, date_of_birth, nationality, address, phone, email,
                document_type, document_number, issue_date, expiry_date, extra_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
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
        if document_type not in DOCUMENT_TYPES:
            raise ValueError("Unsupported document type")
        if side not in DOCUMENT_TYPES[document_type]["sides"]:
            raise ValueError("Unsupported document side")
        gateway_result = gateway_result or {}
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO documents(session_id, document_type, side, file_path, original_filename, content_type,
                sha256, size, normalized_json, risk_status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
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
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO selfies(session_id, file_path, original_filename, content_type, sha256, size, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, file_info["path"], file_info["filename"], content_type, file_info["sha256"], file_info["size"], now_iso()),
            )
        self.audit(session_id, "selfie_uploaded", {"sha256": file_info["sha256"]})

    def add_liveness(self, session_id, file_info, result):
        status = result["risk_status"]
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO liveness_checks(session_id, file_path, sha256, result_json, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, file_info.get("path"), file_info.get("sha256"), json.dumps(result), status, now_iso()),
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
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO face_matches(session_id, score, status, details_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, result.get("score"), result["status"], json.dumps(result), now_iso()),
            )
        self.audit(session_id, "face_match_scored", {"status": result["status"], "score": result.get("score")})

    def submit(self, session_id):
        case = self.get_case(session_id)
        missing = []
        if not case["profile"]:
            missing.append("profile")
        if not case["liveness_checks"]:
            missing.append("liveness")
        if missing:
            raise ValueError(f"Cannot submit before completing: {', '.join(missing)}")
        review_status = self.compute_review_status(case)
        with self.connect() as conn:
            conn.execute(
                "UPDATE sessions SET applicant_status = ?, review_status = ?, submitted_at = ?, updated_at = ? WHERE id = ?",
                ("submitted", review_status, now_iso(), now_iso(), session_id),
            )
        self.audit(session_id, "session_submitted", {"review_status": review_status})
        return self.get_case(session_id)

    def compute_review_status(self, case):
        warnings = []
        usable_liveness = [check for check in case["liveness_checks"] if check["status"] in {"pass", "warn"}]
        if not usable_liveness:
            return "failed_checks"
        if not any(check["status"] == "pass" for check in usable_liveness):
            warnings.append("liveness")
        if case["documents"] and case["selfies"] and (not case["face_matches"] or case["face_matches"][-1]["status"] != "pass"):
            warnings.append("face_match")
        return "needs_review" if warnings else "auto_checks_passed"

    def decide(self, session_id, decision, note, reviewer="admin"):
        if decision not in {"approved", "rejected", "resubmission_requested"}:
            raise ValueError("Invalid decision")
        review_status = "manual_approved" if decision == "approved" else "manual_rejected"
        if decision == "resubmission_requested":
            review_status = "needs_review"
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO decisions(session_id, decision, note, reviewer, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, decision, note, reviewer, now_iso()),
            )
            conn.execute(
                "UPDATE sessions SET applicant_status = ?, review_status = ?, updated_at = ? WHERE id = ?",
                (decision, review_status, now_iso(), session_id),
            )
        self.audit(session_id, "admin_decision", {"decision": decision, "note": note})

    def list_cases(self):
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT s.*, p.full_name, p.document_type, p.document_number
                FROM sessions s
                LEFT JOIN profiles p ON p.session_id = s.id
                ORDER BY s.created_at DESC
                """
            ).fetchall()
            return [row_to_dict(row) for row in rows]

    def get_case(self, session_id):
        with self.connect() as conn:
            session = row_to_dict(conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone())
            if not session:
                raise ValueError("Unknown KYC session")
            profile = row_to_dict(conn.execute("SELECT * FROM profiles WHERE session_id = ?", (session_id,)).fetchone())
            documents = [row_to_dict(row) for row in conn.execute("SELECT * FROM documents WHERE session_id = ? ORDER BY id", (session_id,))]
            selfies = [row_to_dict(row) for row in conn.execute("SELECT * FROM selfies WHERE session_id = ? ORDER BY id", (session_id,))]
            liveness = [row_to_dict(row) for row in conn.execute("SELECT * FROM liveness_checks WHERE session_id = ? ORDER BY id", (session_id,))]
            matches = [row_to_dict(row) for row in conn.execute("SELECT * FROM face_matches WHERE session_id = ? ORDER BY id", (session_id,))]
            decisions = [row_to_dict(row) for row in conn.execute("SELECT * FROM decisions WHERE session_id = ? ORDER BY id", (session_id,))]
            audit = [row_to_dict(row) for row in conn.execute("SELECT * FROM audit_events WHERE session_id = ? ORDER BY id", (session_id,))]
        return {
            "session": session,
            "profile": profile,
            "documents": documents,
            "selfies": selfies,
            "liveness_checks": liveness,
            "face_matches": matches,
            "decisions": decisions,
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

    def extract(self, image_path, document_type=None):
        try:
            import requests
        except Exception as error:
            raise ValueError("requests is required to call the OCR HTTP gateway") from error

        path = Path(image_path)
        params = {"values_only": "false"}
        if document_type:
            params["document_type"] = document_type
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        try:
            with path.open("rb") as handle:
                response = requests.post(
                    f"{self.base_url}/ocr",
                    params=params,
                    files={"file": (path.name, handle, "application/octet-stream")},
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

class FaceMatchService:
    def __init__(self):
        cascade_dir = Path(cv2.data.haarcascades)
        self.face_cascade = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_frontalface_default.xml"))

    def score(self, document_path, selfie_path):
        doc_face = self.extract_face(document_path)
        selfie_face = self.extract_face(selfie_path)
        if doc_face is None or selfie_face is None:
            return {
                "score": None,
                "status": "needs_manual_review",
                "reason": "Could not reliably extract a face from the document or selfie.",
            }

        doc_hist = cv2.calcHist([doc_face], [0], None, [64], [0, 256])
        selfie_hist = cv2.calcHist([selfie_face], [0], None, [64], [0, 256])
        cv2.normalize(doc_hist, doc_hist)
        cv2.normalize(selfie_hist, selfie_hist)
        score = float(cv2.compareHist(doc_hist, selfie_hist, cv2.HISTCMP_CORREL))
        status = "pass" if score >= 0.55 else "warn"
        return {"score": round(score, 4), "status": status, "reason": "Local histogram face comparison; admin review required."}

    def extract_face(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        face = gray[y : y + h, x : x + w]
        return cv2.resize(face, (128, 128))


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
