import json
import io
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import kyc.index as index
import kyc.ocr.service as service


class FakeOCRGateway:
    def extract(self, *args, **kwargs):
        return {
            "engine": "fake",
            "response": {
                "full_name": "Test Applicant",
                "document_number": "N1",
                "objects": [
                    {
                        "label": "id_card",
                        "confidence": 0.91,
                        "box": {
                            "pixel": {"x": 1, "y": 2, "width": 100, "height": 60},
                            "normalized": {"x": 0.01, "y": 0.02, "width": 0.8, "height": 0.5},
                        },
                        "source": "opencv",
                    }
                ],
                "object_summary": {
                    "has_id_card": True,
                    "id_card_confidence": 0.91,
                    "face_count": 0,
                    "text_region_count": 2,
                },
            },
        }


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
        index.repo = index.KYCRepository(str(self.root / "kyc.db"))
        index.storage = index.LocalEvidenceStorage(str(self.root / "evidence"))
        index.ocr_gateway = FakeOCRGateway()
        index.face_match_service = FakeFaceMatchService(index.repo)
        self.client = TestClient(service.app)

    def tearDown(self):
        self.tmp.cleanup()

    def test_root_starts_applicant_session_without_admin_login(self):
        response = self.client.get("/", follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertIn("/kyc/", response.headers["Location"])

        follow = self.client.get(response.headers["Location"])
        self.assertEqual(follow.status_code, 200)
        self.assertIn("KYC Verification", follow.text)

    def test_create_session_does_not_require_tenant_api_key(self):
        response = self.client.post("/api/kyc/sessions")
        self.assertEqual(response.status_code, 201)
        data = response.json()
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
            },
            files={"file": ("front.jpg", io.BytesIO(b"image-bytes"), "image/jpeg")},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        verification = data["verification"]
        self.assertEqual(verification["face_embeddings"][0]["source_type"], "document")
        self.assertEqual(data["gateway"]["document_type_resolution"]["resolved_document_type"], "national_id")
        self.assertTrue(data["gateway"]["document_type_resolution"]["used_fallback"])
        self.assertEqual(data["objects"][0]["label"], "id_card")
        self.assertTrue(data["object_summary"]["has_id_card"])
        self.assertEqual(
            verification["documents"][0]["normalized"]["gateway"]["response"]["object_summary"]["text_region_count"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
