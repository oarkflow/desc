import tempfile
import unittest
from pathlib import Path

from kyc import KYCRepository, LocalEvidenceStorage, OCRService


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
            {"raw_text": "PASSPORT P1234567", "normalized": {"candidate_ids": ["P1234567"]}, "risk_status": "pass"},
        )

        case = self.repo.get_case(session_id)
        self.assertEqual(case["documents"][0]["sha256"], file_info["sha256"])
        self.assertTrue(Path(case["documents"][0]["file_path"]).exists())

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


class OCRTests(unittest.TestCase):
    def test_normalize_extracts_candidate_ids_and_dates(self):
        result = OCRService().normalize("Passport No P1234567 expires 2030-12-31 नाम राम")
        self.assertIn("P1234567", result["candidate_ids"])
        self.assertIn("2030-12-31", result["candidate_dates"])
        self.assertIn("नाम", result["nepali_terms"])

    def test_text_and_confidence_ignores_empty_low_confidence_items(self):
        data = {"text": ["", "AARAV", "NP-DEMO-102938"], "conf": ["-1", "80", "70"]}
        text, confidence = OCRService().text_and_confidence(data)
        self.assertEqual(text, "AARAV NP-DEMO-102938")
        self.assertEqual(confidence, 75)

    def test_nepal_citizenship_structured_fields(self):
        text = (
            "नेपाली नागरिकताको प्रमाणपत्र ना.प्र.नं. १४-०१-७१-००३५० "
            "नामथर अजय कुमार चौधरी लिङ्ग पुरुष "
            "जिल्ला उदयपुर गा.वि. स. सुन्दरपुर वडा न. १ "
            "साल २०५३ महिना ०४ गते ३० थर भरत चौधरी ना.कि. बंशज"
        )
        fields = OCRService().extract_structured_fields(text)
        self.assertEqual(fields["document_type"], "nepal_citizenship")
        self.assertEqual(fields["citizenship_number"], "14-01-71-00350")
        self.assertEqual(fields["full_name_np"], "अजय कुमार चौधरी")
        self.assertEqual(fields["gender"], "male")
        self.assertEqual(fields["date_of_birth_bs"]["formatted"], "2053-04-30")
        self.assertEqual(fields["father_name_np"], "भरत चौधरी")
        self.assertEqual(fields["citizenship_type"], "descent")
        self.assertEqual(fields["district_np"], "उदयपुर")


if __name__ == "__main__":
    unittest.main()
