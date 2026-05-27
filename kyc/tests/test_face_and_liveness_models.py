import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from kyc.kyc import AntiSpoofingProvider, FaceMatchService, InsightFaceRecognitionProvider, KYCRepository, LivenessService


class EmbeddedFaceProvider:
    provider_name = "embedded"
    model_version = "test"
    available = True

    def detect_faces(self, image):
        return [
            {
                "x": 1,
                "y": 2,
                "width": 10,
                "height": 12,
                "quality": 0.91,
                "_embedding": [1.0, 0.0, 0.0],
            }
        ]

    def extract_embedding(self, image, face_box):
        return face_box.get("_embedding")

    def compare(self, embedding_a, embedding_b):
        return 1.0


class FakeBlinkDetector:
    backend = "fake"

    def __init__(self):
        self.blink_count = 0
        self.consecutive_low_ear = 0

    def detect_blink(self, image):
        self.blink_count += 1
        return True, 0.2, True

    def detect_face_points(self, image):
        return [(0.5, 0.4), (0.5, 0.6)]


class FakeAntiSpoofing:
    provider_name = "fake_anti_spoof"
    enabled = True
    available = True

    def analyze(self, image, face_box=None):
        return {
            "enabled": True,
            "available": True,
            "status": "spoof",
            "live_score": 0.1,
            "threshold": 0.65,
            "provider": self.provider_name,
            "model_version": "test",
        }


class ModelBackedKYCIntegrationTests(unittest.TestCase):
    def test_face_box_storage_strips_private_embedding_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = KYCRepository(str(root / "kyc.db"))
            image_path = root / "selfie.jpg"
            cv2.imwrite(str(image_path), np.zeros((24, 24, 3), dtype=np.uint8))

            service = FaceMatchService(repo, EmbeddedFaceProvider())
            result = service.enroll_source(repo.create_session()["id"], "selfie", image_path)

        self.assertEqual(result["status"], "active")
        self.assertEqual(result["provider"], "embedded")
        self.assertNotIn("_embedding", result["face_box"])
        self.assertEqual(result["face_box"]["quality"], 0.91)

    def test_liveness_marks_spoofed_frame_as_failed(self):
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)

        service = LivenessService(detector=FakeBlinkDetector(), anti_spoofing=FakeAntiSpoofing())
        result = service.analyze_frame_bytes(encoded.tobytes(), session_id="session-1", challenge=["blink"])

        self.assertEqual(result["anti_spoofing"]["status"], "spoof")
        self.assertEqual(result["liveness_state"]["risk_status"], "fail")

    def test_missing_anti_spoof_model_requests_manual_review_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict("os.environ", {"ANTI_SPOOF_ENABLED": "true"}):
                provider = AntiSpoofingProvider(model_path=Path(tmp) / "missing.onnx")
            result = provider.analyze(np.zeros((80, 80, 3), dtype=np.uint8))

        self.assertEqual(result["status"], "needs_manual_review")
        self.assertIn("missing", result["reason"].lower())

    def test_missing_required_insightface_artifact_returns_model_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "selfie.jpg"
            cv2.imwrite(str(image_path), np.zeros((24, 24, 3), dtype=np.uint8))
            provider = InsightFaceRecognitionProvider(model_root=Path(tmp) / "insightface")
            result = FaceMatchService(provider=provider).extract_embedding_from_file(image_path, "selfie")

        self.assertEqual(result["status"], "needs_manual_review")
        self.assertIn("InsightFace model artifact is missing", result["reason"])
