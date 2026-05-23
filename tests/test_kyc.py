import json
import tempfile
import unittest
from pathlib import Path

from kyc import DocumentConfigError, DocumentRegistry, KYCRepository, LocalEvidenceStorage


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

    def test_submit_requires_all_required_evidence(self):
        session_id = self.repo.create_session()["id"]
        with self.assertRaises(ValueError):
            self.repo.submit(session_id)

    def test_admin_decision_updates_state_and_audit(self):
        session_id = self.repo.create_session()["id"]
        self.repo.decide(session_id, "resubmission_requested", "Document is blurry")

        case = self.repo.get_case(session_id)
        self.assertEqual(case["session"]["applicant_status"], "resubmission_requested")
        self.assertEqual(case["decisions"][0]["note"], "Document is blurry")
        self.assertTrue(any(event["event_type"] == "admin_decision" for event in case["audit_events"]))

    def test_demo_session_can_submit_with_liveness_only(self):
        session_id = self.repo.create_demo_session()["id"]
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

        case = self.repo.submit(session_id)
        self.assertEqual(case["session"]["applicant_status"], "submitted")
        self.assertEqual(case["session"]["review_status"], "auto_checks_passed")


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
