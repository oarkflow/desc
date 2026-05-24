import io
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import cv2
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ocr_service


def image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(buffer, format="PNG")
    return buffer.getvalue()


class OCREndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(ocr_service.app)
        self.item = {
            "text": "NATIONAL IDENTITY CARD",
            "confidence": 0.99,
            "box": [[0, 0], [2, 0], [2, 1], [0, 1]],
            "source_pass": "bootstrap",
        }
        self.field = ocr_service.OCRField(
            value="023-456-2930",
            confidence=0.98,
            source_text="ID No 023-456-2930",
        )

    def post_image(self, query=""):
        with mock.patch("ocr_service.run_profile_ocr") as run_profile_ocr:
            with mock.patch("ocr_service.extract_structured_fields") as extract_fields:
                with mock.patch("ocr_service.save_debug_image"):
                    run_profile_ocr.return_value = (
                        [self.item],
                        [self.item],
                        "nepali_national_id",
                        None,
                        0.75,
                        np.zeros((2, 2, 3), dtype=np.uint8),
                        {},
                    )
                    extract_fields.return_value = {"nid_number": self.field}
                    return self.client.post(
                        f"/ocr{query}",
                        files={
                            "file": (
                                "national-id.webp",
                                image_bytes(),
                                "application/octet-stream",
                            )
                        },
                    )

    def test_default_response_is_values_only_metadata_wrapper(self):
        response = self.post_image()

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["document_type"], "nepali_national_id")
        self.assertEqual(payload["values"], {"nid_number": "023-456-2930"})
        self.assertEqual(payload["meta"]["document_type"], "nepali_national_id")
        self.assertEqual(payload["meta"]["document_type_confidence"], 0.75)
        self.assertIn("objects", payload)
        self.assertEqual(payload["object_summary"]["text_region_count"], 1)
        self.assertNotIn("request_id", payload)
        self.assertNotIn("fields", payload)
        self.assertNotIn("items", payload)

    def test_full_response_can_be_requested_with_values_only_false(self):
        response = self.post_image("?values_only=false")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["document_type"], "nepali_national_id")
        self.assertEqual(payload["document_type_confidence"], 0.75)
        self.assertEqual(payload["meta"]["document_type"], "nepali_national_id")
        self.assertIn("request_id", payload)
        self.assertIn("fields", payload)
        self.assertIn("items", payload)
        self.assertIn("objects", payload)
        self.assertIn("object_summary", payload)

    def test_values_only_keeps_lightweight_metadata_wrapper(self):
        response = self.post_image("?values_only=true")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["document_type"], "nepali_national_id")
        self.assertEqual(payload["values"], {"nid_number": "023-456-2930"})
        self.assertEqual(payload["meta"]["document_type_confidence"], 0.75)
        self.assertIn("objects", payload)
        self.assertNotIn("items", payload)

    def test_detect_objects_false_returns_empty_detection_output(self):
        response = self.post_image("?values_only=true&detect_objects=false")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["objects"], [])
        self.assertEqual(
            payload["object_summary"],
            {
                "has_id_card": False,
                "id_card_confidence": 0.0,
                "face_count": 0,
                "text_region_count": 0,
            },
        )

    def test_fields_only_keeps_compact_structured_response(self):
        response = self.post_image("?values_only=false&fields_only=true")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["document_type"], "nepali_national_id")
        self.assertIn("fields", payload)
        self.assertNotIn("items", payload)

    def test_include_stats_adds_runtime_metadata_to_values_response(self):
        response = self.post_image("?values_only=true&include_stats=true")

        self.assertEqual(response.status_code, 200)
        meta = response.json()["meta"]
        self.assertEqual(meta["document_type"], "nepali_national_id")
        self.assertIn("processing_ms", meta)
        self.assertIn("resource_usage", meta)

    def test_rejects_unsupported_extension(self):
        response = self.client.post(
            "/ocr",
            files={"file": ("document.txt", b"hello", "application/octet-stream")},
        )

        self.assertEqual(response.status_code, 415)
        self.assertIn("Unsupported file type", response.json()["detail"])

    def test_rejects_invalid_image_bytes(self):
        response = self.client.post(
            "/ocr",
            files={"file": ("document.png", b"not-an-image", "image/png")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Invalid image file")

    def test_rejects_oversized_body(self):
        with mock.patch.object(ocr_service.settings, "MAX_FILE_MB", 0):
            response = self.client.post(
                "/ocr",
                files={"file": ("document.png", image_bytes(), "image/png")},
            )

        self.assertEqual(response.status_code, 413)


class OpenCVObjectDetectionTests(unittest.TestCase):
    def test_detects_synthetic_id_card(self):
        image = np.full((420, 640, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (90, 110), (550, 335), (20, 20, 20), 4)

        objects, summary = ocr_service.ObjectDetectionService("opencv").detect(image, [])

        cards = [item for item in objects if item["label"] == "id_card"]
        self.assertEqual(len(cards), 1)
        self.assertTrue(summary["has_id_card"])
        pixel = cards[0]["box"]["pixel"]
        self.assertGreater(pixel["width"], 400)
        self.assertGreater(pixel["height"], 180)
        self.assertGreaterEqual(cards[0]["box"]["normalized"]["x"], 0)
        self.assertLessEqual(cards[0]["box"]["normalized"]["width"], 1)

    def test_blank_image_has_no_id_card(self):
        image = np.full((420, 640, 3), 255, dtype=np.uint8)

        objects, summary = ocr_service.ObjectDetectionService("opencv").detect(image, [])

        self.assertEqual([item for item in objects if item["label"] == "id_card"], [])
        self.assertFalse(summary["has_id_card"])


if __name__ == "__main__":
    unittest.main()
