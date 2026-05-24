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


class ApplicantFlowTests(unittest.TestCase):
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

    def test_root_starts_applicant_session_without_admin_login(self):
        response = self.client.get("/", follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertIn("/kyc/", response.headers["Location"])

        follow = self.client.get(response.headers["Location"])
        self.assertEqual(follow.status_code, 200)
        self.assertIn("KYC Verification", follow.get_data(as_text=True))

    def test_create_session_does_not_require_tenant_api_key(self):
        response = self.client.post("/api/kyc/sessions")
        self.assertEqual(response.status_code, 201)
        data = response.get_json()
        self.assertIn("/kyc/", data["applicant_url"])
        self.assertEqual(data["verification"]["session"]["id"], data["session_id"])

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
        verification = response.get_json()["verification"]
        self.assertEqual(verification["face_embeddings"][0]["source_type"], "document")


if __name__ == "__main__":
    unittest.main()
