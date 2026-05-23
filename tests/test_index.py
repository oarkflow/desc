import json
import tempfile
import unittest
from pathlib import Path

import index


class AdminEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        index.app.config.update(TESTING=True, SECRET_KEY="test")
        index.repo = index.KYCRepository(str(self.root / "kyc.db"))
        index.storage = index.LocalEvidenceStorage(str(self.root / "evidence"))
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

        response = self.client.get(f"/admin/evidence/liveness/{item_id}")
        try:
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, "video/webm")
            self.assertEqual(response.get_data(), b"webm-bytes")
        finally:
            response.close()


if __name__ == "__main__":
    unittest.main()
