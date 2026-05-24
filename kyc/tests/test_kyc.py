import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from kyc.kyc import DocumentConfigError, DocumentRegistry, FaceMatchService, KYCRepository, LocalEvidenceStorage, OCRGatewayClient, OCRProfileMapper


class FakeFaceProvider:
    provider_name = "fake"
    model_version = "test"
    available = True

    def detect_faces(self, image):
        return [{"x": 0, "y": 0, "width": 10, "height": 10, "quality": 0.9}]

    def extract_embedding(self, image, face_box):
        return [1.0, 0.0, 0.0]

    def compare(self, embedding_a, embedding_b):
        import numpy as np

        a = np.asarray(embedding_a, dtype=np.float32)
        b = np.asarray(embedding_b, dtype=np.float32)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class KYCPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.repo = KYCRepository(str(self.root / "kyc.db"))
        self.storage = LocalEvidenceStorage(str(self.root / "evidence"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_session_profile_and_status_flow(self):
        session = self.repo.create_session()
        session_id = session["id"]

        self.assertTrue(session["tenant_id"])
        self.assertTrue(session["session_token"])

        self.repo.save_profile(
            session_id,
            {
                "full_name": "Test Applicant",
                "date_of_birth": "1990-01-01",
                "nationality": "Nepal",
                "address": "Kathmandu",
                "document_type": "passport",
                "document_number": "P1234567",
            },
        )

        case = self.repo.get_case(session_id)
        self.assertEqual(case["profile"]["full_name"], "Test Applicant")
        self.assertEqual(case["session"]["applicant_status"], "draft")

    def test_profile_rejects_unknown_document_type(self):
        session_id = self.repo.create_session()["id"]
        with self.assertRaises(ValueError):
            self.repo.save_profile(
                session_id,
                {
                    "full_name": "Test Applicant",
                    "date_of_birth": "1990-01-01",
                    "nationality": "Nepal",
                    "address": "Kathmandu",
                    "document_type": "library_card",
                    "document_number": "X1",
                },
            )

    def test_evidence_hash_and_document_metadata(self):
        session_id = self.repo.create_session()["id"]
        file_info = self.storage.save_bytes(session_id, "documents", "front.jpg", b"document-bytes")
        self.repo.add_document(
            session_id,
            "passport",
            "front",
            file_info,
            "image/jpeg",
            {"engine": "http_gateway", "response": {"values": {"passport_number": "P1234567"}}},
        )

        case = self.repo.get_case(session_id)
        document = case["documents"][0]
        self.assertEqual(document["sha256"], file_info["sha256"])
        self.assertEqual(document["risk_status"], "uploaded")
        self.assertEqual(json.loads(document["normalized_json"])["gateway"]["engine"], "http_gateway")
        self.assertTrue(Path(document["file_path"]).exists())

    def test_tenant_api_key_authenticates_active_tenant(self):
        tenant = self.repo.create_tenant("bank-a", "Bank A")
        self.repo.add_api_key(tenant["id"], "bank-a-secret", "primary")

        authenticated = self.repo.authenticate_api_key("bank-a-secret")
        session = self.repo.create_session(authenticated["id"])

        self.assertEqual(authenticated["slug"], "bank-a")
        self.assertEqual(session["tenant_id"], tenant["id"])

    def test_unknown_tenant_api_key_is_rejected(self):
        with self.assertRaises(ValueError):
            self.repo.authenticate_api_key("missing-secret")

    def test_demo_session_tracks_document_and_liveness_without_review_submit(self):
        session_id = self.repo.create_demo_session()["id"]
        file_info = self.storage.save_bytes(session_id, "documents", "front.jpg", b"document-bytes")
        self.repo.add_document(session_id, "national_id", "front", file_info, "image/jpeg", {"response": {"values": {"document_number": "N1"}}})
        self.repo.add_liveness(
            session_id,
            {},
            {
                "risk_status": "pass",
                "challenge": ["look_center", "blink"],
                "completed": {"look_center": True, "blink": True},
                "blink_count": 1,
                "face_detection_rate": 0.8,
                "movement_detected": False,
                "frames_processed": 10,
                "backend": "test",
            },
        )

        case = self.repo.get_case(session_id)
        self.assertEqual(case["session"]["applicant_status"], "draft")
        self.assertEqual(case["documents"][0]["document_type"], "national_id")
        self.assertEqual(case["liveness_checks"][0]["status"], "pass")

    def test_face_search_is_tenant_scoped(self):
        session_a = self.repo.create_session()
        session_b = self.repo.create_session()
        tenant_b = self.repo.create_tenant("tenant-b", "Tenant B")
        session_c = self.repo.create_session(tenant_b["id"])
        query_id = self.repo.add_face_embedding(session_a["id"], "selfie", [1, 0, 0], "fake", "test")
        self.repo.add_face_embedding(session_b["id"], "selfie", [0.99, 0.01, 0], "fake", "test")
        self.repo.add_face_embedding(session_c["id"], "selfie", [1, 0, 0], "fake", "test")

        matches = FaceMatchService(self.repo, FakeFaceProvider()).search_tenant_gallery(session_a["id"])
        case = self.repo.get_case(session_a["id"])

        self.assertEqual(query_id, case["face_search_results"][0]["query_embedding_id"])
        self.assertEqual([match["session_id"] for match in matches], [session_b["id"]])

    def test_delete_evidence_retains_face_embeddings(self):
        session_id = self.repo.create_session()["id"]
        file_info = self.storage.save_bytes(session_id, "documents", "front.jpg", b"document-bytes")
        self.repo.add_document(session_id, "passport", "front", file_info, "image/jpeg", {})
        embedding_id = self.repo.add_face_embedding(session_id, "document", [1, 0, 0], "fake", "test")

        self.repo.delete_evidence(session_id)
        case = self.repo.get_case(session_id)

        self.assertEqual(case["documents"], [])
        self.assertEqual(case["face_embeddings"][0]["id"], embedding_id)

    def test_missing_face_model_returns_manual_review(self):
        service = FaceMatchService(self.repo)
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "blank.jpg"
            image_path.write_bytes(b"not-an-image")
            result = service.enroll_source(self.repo.create_session()["id"], "document", image_path)

        self.assertEqual(result["status"], "needs_manual_review")

class OCRProfileMapperTests(unittest.TestCase):
    def test_maps_gateway_values_to_profile_fields(self):
        mapper = OCRProfileMapper()
        suggested = mapper.map(
            {
                "response": {
                    "values": {
                        "name": "Test Applicant",
                        "dob": "1990-01-01",
                        "passport_number": "P1234567",
                        "date_of_expiry": "2030-01-01",
                    }
                }
            },
            document_type="passport",
        )

        self.assertEqual(suggested["full_name"], "Test Applicant")
        self.assertEqual(suggested["date_of_birth"], "1990-01-01")
        self.assertEqual(suggested["document_number"], "P1234567")
        self.assertEqual(suggested["expiry_date"], "2030-01-01")
        self.assertEqual(suggested["document_type"], "passport")


class OCRGatewayClientTests(unittest.TestCase):
    def test_forwards_upload_content_type_to_gateway(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "passport.jpg"
            path.write_bytes(b"image-bytes")
            response = mock.Mock()
            response.ok = True
            response.status_code = 200
            response.headers = {"content-type": "application/json"}
            response.json.return_value = {"values": {"passport_number": "P1"}}

            with mock.patch("requests.post", return_value=response) as post:
                client = OCRGatewayClient(base_url="http://ocr.test")
                client.extract(path, document_type="passport", content_type="image/jpeg", filename="passport.jpg")

            _, kwargs = post.call_args
            uploaded = kwargs["files"]["file"]
            self.assertEqual(uploaded[0], "passport.jpg")
            self.assertEqual(uploaded[2], "image/jpeg")
            self.assertEqual(kwargs["params"]["values_only"], "true")
            self.assertNotIn("document_type", kwargs["params"])

    def test_can_opt_into_forwarding_document_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "passport.jpg"
            path.write_bytes(b"image-bytes")
            response = mock.Mock()
            response.ok = True
            response.status_code = 200
            response.headers = {"content-type": "application/json"}
            response.json.return_value = {"values": {"passport_number": "P1"}}

            with mock.patch.dict("os.environ", {"OCR_GATEWAY_SEND_DOCUMENT_TYPE": "true"}):
                with mock.patch("requests.post", return_value=response) as post:
                    client = OCRGatewayClient(base_url="http://ocr.test")
                    client.extract(path, document_type="passport", content_type="image/jpeg", filename="passport.jpg")

            _, kwargs = post.call_args
            self.assertEqual(kwargs["params"]["document_type"], "passport")


class DocumentRegistryTests(unittest.TestCase):
    def test_loads_json_document_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "document_types.json"
            path.write_text(
                json.dumps(
                    {
                        "document_types": [
                            {
                                "id": "test_id",
                                "label": "Test ID",
                                "sides": ["front"],
                                "profile_fields": ["document_number"],
                            }
                        ]
                    }
                )
            )
            registry = DocumentRegistry(path)
            self.assertEqual(registry.public_types()["test_id"]["label"], "Test ID")

    def test_rejects_invalid_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "document_types.json"
            path.write_text(json.dumps({"document_types": [{"id": "bad", "label": "Bad"}]}))
            with self.assertRaises(DocumentConfigError):
                DocumentRegistry(path)


if __name__ == "__main__":
    unittest.main()
