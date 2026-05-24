import json
import io
import tempfile
import unittest
from pathlib import Path

import kyc.index as index


class FakeOCRGateway:
    def extract(self, *args, **kwargs):
        return {"engine": "fake", "response": {"full_name": "Test Applicant", "document_number": "N1"}}


class FakeFaceMatchService:
    def __init__(self, repo):
        self.repo = repo

    def enroll_source(self, session_id, source_type, image_path, source_id=None):
        embedding_id = self.repo.add_face_embedding(session_id, source_type, [1, 0, 0], "fake", "test", source_id=source_id)
        return {"status": "active", "embedding_id": embedding_id}

    def compare_session(self, session_id):
        return {"score": 1.0, "status": "pass", "provider": "fake", "model_version": "test"}

    def search_tenant_gallery(self, session_id):
        return []


class AdminEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        index.app.config.update(TESTING=True, SECRET_KEY="test")
        index.repo = index.KYCRepository(str(self.root / "kyc.db"))
        index.storage = index.LocalEvidenceStorage(str(self.root / "evidence"))
        index.ocr_gateway = FakeOCRGateway()
        index.face_match_service = FakeFaceMatchService(index.repo)
        self.client = index.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def test_liveness_evidence_uses_result_json_content_type(self):
        session_id = index.repo.create_demo_session()["id"]
        file_info = index.storage.save_bytes(session_id, "liveness", "proof.webm", b"webm-bytes")
        index.repo.add_liveness(
            session_id,
            file_info,
            {
                "risk_status": "pass",
                "content_type": "video/webm",
                "challenge": ["blink"],
                "completed": {"blink": True},
            },
        )
        case = index.repo.get_case(session_id)
        item_id = case["liveness_checks"][0]["id"]

        with self.client.session_transaction() as session:
            session["admin_authenticated"] = True
            session["admin_tenant_id"] = case["session"]["tenant_id"]
            session["admin_email"] = "admin@example.com"
            session["admin_role"] = "owner"

        response = self.client.get(f"/admin/evidence/liveness/{item_id}")
        try:
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, "video/webm")
            self.assertEqual(response.get_data(), b"webm-bytes")
        finally:
            response.close()

    def test_create_session_requires_valid_tenant_api_key(self):
        response = self.client.post("/api/kyc/sessions")
        self.assertEqual(response.status_code, 401)

        response = self.client.post("/api/kyc/sessions", headers={"X-Tenant-API-Key": "dev-tenant-key"})
        self.assertEqual(response.status_code, 201)
        data = response.get_json()
        self.assertIn("/kyc/", data["applicant_url"])
        self.assertEqual(data["case"]["session"]["id"], data["session_id"])

    def test_admin_cases_are_tenant_scoped(self):
        default_session = index.repo.create_session()
        tenant = index.repo.create_tenant("tenant-b", "Tenant B")
        other_session = index.repo.create_session(tenant["id"])

        with self.client.session_transaction() as session:
            session["admin_authenticated"] = True
            session["admin_tenant_id"] = default_session["tenant_id"]
            session["admin_email"] = "admin@example.com"
            session["admin_role"] = "owner"

        response = self.client.get("/admin/cases")
        body = response.get_data(as_text=True)

        self.assertIn(default_session["id"], body)
        self.assertNotIn(other_session["id"], body)

    def test_document_upload_stores_face_embedding(self):
        created = index.repo.create_session()
        response = self.client.post(
            f"/api/kyc/sessions/{created['id']}/documents",
            headers={"X-Session-Token": created["session_token"]},
            data={
                "document_type": "national_id",
                "side": "front",
                "file": (io.BytesIO(b"image-bytes"), "front.jpg"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        case = response.get_json()["case"]
        self.assertEqual(case["face_embeddings"][0]["source_type"], "document")


if __name__ == "__main__":
    unittest.main()
