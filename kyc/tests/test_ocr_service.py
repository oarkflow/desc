import io
import json
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


def pdf_bytes() -> bytes:
    return b"%PDF-1.4\n% test\n"


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
        self.assertNotIn("objects", payload)
        self.assertNotIn("object_summary", payload)
        self.assertNotIn("tamper_score", payload)
        self.assertNotIn("status", payload)
        self.assertNotIn("flags", payload)
        self.assertNotIn("manual_review_required", payload)
        self.assertNotIn("tamper", payload)
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
        self.assertIn("tamper", payload)

    def test_values_only_keeps_lightweight_metadata_wrapper(self):
        response = self.post_image("?values_only=true")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["document_type"], "nepali_national_id")
        self.assertEqual(payload["values"], {"nid_number": "023-456-2930"})
        self.assertEqual(payload["meta"]["document_type_confidence"], 0.75)
        self.assertNotIn("objects", payload)
        self.assertNotIn("tamper", payload)
        self.assertNotIn("items", payload)

    def test_detect_objects_false_still_returns_values_only_payload(self):
        response = self.post_image("?values_only=true&detect_objects=false")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["values"], {"nid_number": "023-456-2930"})
        self.assertNotIn("objects", payload)
        self.assertNotIn("object_summary", payload)

    def test_fields_only_keeps_compact_structured_response(self):
        response = self.post_image("?values_only=false&fields_only=true")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["document_type"], "nepali_national_id")
        self.assertIn("fields", payload)
        self.assertIn("tamper", payload)
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

    def test_pdf_upload_processes_pages_and_aggregates_tamper(self):
        pages = [
            {
                "page_number": 1,
                "image": np.zeros((2, 2, 3), dtype=np.uint8),
                "width": 2,
                "height": 2,
                "mime_type": "application/pdf",
            },
            {
                "page_number": 2,
                "image": np.zeros((2, 2, 3), dtype=np.uint8),
                "width": 2,
                "height": 2,
                "mime_type": "application/pdf",
            },
        ]
        with mock.patch("ocr_service.load_document_pages", return_value=pages):
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
                        response = self.client.post(
                            "/ocr?values_only=false&fields_only=true",
                            files={"file": ("document.pdf", pdf_bytes(), "application/pdf")},
                        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["meta"]["page_count"], 2)
        self.assertEqual(run_profile_ocr.call_count, 2)
        self.assertEqual(payload["tamper"]["checks"]["aggregate_strategy"], "max_page_score")


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

    def test_citizenship_sample_filters_small_false_face_candidates(self):
        image_path = Path(__file__).resolve().parents[1] / "testdata" / "citizenship.jpg"
        image = np.array(Image.open(image_path).convert("RGB"))

        objects, summary = ocr_service.ObjectDetectionService("opencv").detect(image, [])

        faces = [item for item in objects if item["label"] == "face"]
        self.assertLessEqual(summary["face_count"], 1)
        self.assertEqual(summary["face_count"], len(faces))


class TamperDetectionTests(unittest.TestCase):
    def setUp(self):
        self.service = ocr_service.TamperDetectionService()
        self.image = np.full((120, 220, 3), 255, dtype=np.uint8)
        self.field = ocr_service.OCRField(
            value="bad-number",
            confidence=0.9,
            source_text="bad-number",
            evidence=[
                {
                    "type": "value_ocr_line",
                    "bounds": [180, 90, 210, 110],
                }
            ],
        )

    def test_validator_failure_sets_tamper_flag(self):
        profile = ocr_service.DocumentProfile(
            fields={
                "citizenship_number": ocr_service.ProfileFieldConfig(
                    validators=[
                        ocr_service.ValidatorConfig(
                            type="regex",
                            pattern=r"\d{5}-\d{5}",
                        )
                    ]
                )
            }
        )

        result = self.service.analyze(
            self.image,
            self.image,
            "nepali_citizenship",
            profile,
            {"citizenship_number": self.field},
            [],
            [],
            ocr_service.empty_object_summary(),
        )

        codes = {flag.code for flag in result.flags}
        self.assertIn("field_validator_failed", codes)
        self.assertGreaterEqual(result.tamper_score, 0.16)

    def test_profile_regex_failure_sets_tamper_flag(self):
        profile = ocr_service.DocumentProfile(
            fields={
                "nid_number": ocr_service.ProfileFieldConfig(
                    regex=r"[0-9]{3}-[0-9]{3}-[0-9]{4}",
                )
            }
        )

        result = self.service.analyze(
            self.image,
            self.image,
            "nepali_national_id",
            profile,
            {"nid_number": self.field},
            [],
            [],
            ocr_service.empty_object_summary(),
        )

        self.assertIn("field_regex_failed", {flag.code for flag in result.flags})

    def test_date_consistency_detects_dob_after_issue_date(self):
        fields = {
            "date_of_birth": ocr_service.OCRField(
                value="2025-01-01",
                confidence=0.9,
                source_text="2025-01-01",
            ),
            "issue_date": ocr_service.OCRField(
                value="2024-01-01",
                confidence=0.9,
                source_text="2024-01-01",
            ),
        }

        result = self.service.analyze(
            self.image,
            self.image,
            "nepali_citizenship",
            ocr_service.DocumentProfile(),
            fields,
            [],
            [],
            ocr_service.empty_object_summary(),
        )

        self.assertIn("date_consistency_failed", {flag.code for flag in result.flags})
        self.assertTrue(result.manual_review_required)

    def test_layout_mismatch_uses_configured_regions(self):
        profile = ocr_service.DocumentProfile(
            fields={"full_name": ocr_service.ProfileFieldConfig()},
            tamper=ocr_service.TamperConfig(
                field_regions={"full_name": [0.0, 0.0, 0.25, 0.25]}
            ),
        )

        result = self.service.analyze(
            self.image,
            self.image,
            "nepali_citizenship",
            profile,
            {"full_name": self.field},
            [],
            [],
            ocr_service.empty_object_summary(),
        )

        self.assertIn("layout_region_mismatch", {flag.code for flag in result.flags})

    def test_multi_anchor_regions_choose_highest_confidence_value(self):
        image = np.full((100, 200, 3), 255, dtype=np.uint8)
        lines = [
            {
                "text": "LOW VALUE",
                "confidence": 0.45,
                "box": [[10, 10], [50, 10], [50, 20], [10, 20]],
                "source_pass": "default",
                "source_kind": "printed",
            },
            {
                "text": "HIGH VALUE",
                "confidence": 0.92,
                "box": [[130, 60], [190, 60], [190, 75], [130, 75]],
                "source_pass": "default",
                "source_kind": "printed",
            },
        ]
        profile = ocr_service.DocumentProfile(
            fields={
                "full_name": ocr_service.ProfileFieldConfig(
                    strategies=["anchor_region"],
                    anchor_regions=[
                        [0.0, 0.0, 0.35, 0.35],
                        [0.55, 0.45, 1.0, 0.9],
                    ],
                )
            }
        )

        fields = ocr_service.extract_structured_fields(lines, image, profile)

        self.assertEqual(fields["full_name"].value, "HIGH VALUE")
        self.assertEqual(fields["full_name"].details["region_index"], 1)

    def test_multi_retry_regions_create_multiple_retry_crops(self):
        field_config = ocr_service.ProfileFieldConfig(
            retry_regions=[
                [0.0, 0.0, 0.25, 0.25],
                [0.5, 0.5, 0.75, 0.75],
            ]
        )

        crops = ocr_service.retry_crops_for_field(self.image, None, field_config)

        self.assertEqual(len(crops), 2)
        self.assertEqual(crops[0][2]["region_index"], 0)
        self.assertEqual(crops[1][2]["region_index"], 1)

    def test_layout_accepts_field_inside_any_configured_region(self):
        profile = ocr_service.DocumentProfile(
            fields={"full_name": ocr_service.ProfileFieldConfig()},
            tamper=ocr_service.TamperConfig(
                field_regions={
                    "full_name": [
                        [0.0, 0.0, 0.25, 0.25],
                        [0.75, 0.7, 1.0, 1.0],
                    ]
                }
            ),
        )

        result = self.service.analyze(
            self.image,
            self.image,
            "nepali_citizenship",
            profile,
            {"full_name": self.field},
            [],
            [],
            ocr_service.empty_object_summary(),
        )

        self.assertNotIn("layout_region_mismatch", {flag.code for flag in result.flags})

    def test_expected_object_accepts_any_configured_region(self):
        face = {
            "label": "face",
            "confidence": 0.9,
            "box": ocr_service.detection_box(175, 82, 30, 25, 220, 120),
        }
        profile = ocr_service.DocumentProfile(
            tamper=ocr_service.TamperConfig(
                expected_objects=[
                    ocr_service.ExpectedObjectConfig(
                        label="face",
                        min_confidence=0.5,
                        regions=[
                            [0.0, 0.0, 0.25, 0.25],
                            [0.75, 0.65, 1.0, 1.0],
                        ],
                    )
                ]
            )
        )

        result = self.service.analyze(
            self.image,
            self.image,
            "nepali_national_id",
            profile,
            {},
            [],
            [face],
            {"has_id_card": False, "id_card_confidence": 0.0, "face_count": 1, "text_region_count": 0},
        )

        self.assertNotIn("object_region_mismatch", {flag.code for flag in result.flags})

    def test_face_quality_flags_multiple_faces(self):
        face_a = {
            "label": "face",
            "confidence": 0.75,
            "box": ocr_service.detection_box(10, 10, 40, 40, 220, 120),
        }
        face_b = {
            "label": "face",
            "confidence": 0.75,
            "box": ocr_service.detection_box(100, 10, 40, 40, 220, 120),
        }

        result = self.service.analyze(
            self.image,
            self.image,
            "nepali_national_id",
            ocr_service.DocumentProfile(),
            {},
            [],
            [face_a, face_b],
            {"has_id_card": False, "id_card_confidence": 0.0, "face_count": 2, "text_region_count": 0},
        )

        self.assertIn("multiple_faces_detected", {flag.code for flag in result.flags})

    def test_protected_hologram_region_flags_flat_blue_overlay(self):
        image = np.full((120, 220, 3), 180, dtype=np.uint8)
        cv2.rectangle(image, (120, 10), (190, 60), (50, 80, 245), -1)
        profile = ocr_service.DocumentProfile(
            tamper=ocr_service.TamperConfig(
                protected_regions={"hologram": [0.54, 0.08, 0.88, 0.52]}
            )
        )

        result = self.service.analyze(
            image,
            image,
            "nepali_citizenship_old_front",
            profile,
            {},
            [],
            [],
            ocr_service.empty_object_summary(),
        )

        self.assertIn("protected_region_color_anomaly", {flag.code for flag in result.flags})

    def test_protected_photo_region_flags_replacement_background(self):
        image = np.full((120, 220, 3), 150, dtype=np.uint8)
        cv2.rectangle(image, (4, 45), (70, 110), (245, 245, 245), -1)
        cv2.circle(image, (36, 75), 16, (40, 40, 40), -1)
        profile = ocr_service.DocumentProfile(
            tamper=ocr_service.TamperConfig(
                protected_regions={"photo": [0.0, 0.35, 0.34, 0.95]}
            )
        )

        result = self.service.analyze(
            image,
            image,
            "nepali_citizenship_old_front",
            profile,
            {},
            [],
            [],
            ocr_service.empty_object_summary(),
        )

        self.assertIn("photo_region_background_anomaly", {flag.code for flag in result.flags})

    def test_protected_regions_scan_multiple_boxes(self):
        image = np.full((120, 220, 3), 180, dtype=np.uint8)
        cv2.rectangle(image, (150, 60), (205, 105), (50, 80, 245), -1)
        profile = ocr_service.DocumentProfile(
            tamper=ocr_service.TamperConfig(
                protected_regions={
                    "hologram": [
                        [0.0, 0.0, 0.25, 0.25],
                        [0.68, 0.48, 0.94, 0.9],
                    ]
                }
            )
        )

        result = self.service.analyze(
            image,
            image,
            "nepali_citizenship_old_front",
            profile,
            {},
            [],
            [],
            ocr_service.empty_object_summary(),
        )

        flags = [flag for flag in result.flags if flag.code == "protected_region_color_anomaly"]
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0].evidence["region_index"], 1)

    def test_ffmpeg_tampered_citizenship_fixture_flags_photo_and_hologram(self):
        testdata = Path(__file__).resolve().parents[1] / "testdata"
        tampered_path = testdata / "citizenship-full-tampered-ffmpeg.jpg"
        if not tampered_path.exists():
            self.skipTest("ffmpeg tampered fixture is not present")
        config_dir = Path(__file__).resolve().parents[1] / "config" / "document_types"
        profile = ocr_service.load_document_type_file(
            config_dir / "nepali_citizenship_old_front.yaml"
        )["nepali_citizenship_old_front"]
        original = np.array(Image.open(testdata / "citizenship.jpg").convert("RGB"))
        tampered = np.array(Image.open(tampered_path).convert("RGB"))

        original_codes = {
            flag.code
            for flag in self.service.check_protected_regions(profile, original)
        }
        tampered_codes = {
            flag.code
            for flag in self.service.check_protected_regions(profile, tampered)
        }

        self.assertNotIn("protected_region_color_anomaly", original_codes)
        self.assertNotIn("photo_region_background_anomaly", original_codes)
        self.assertIn("protected_region_color_anomaly", tampered_codes)
        self.assertIn("photo_region_background_anomaly", tampered_codes)

    def test_production_mode_requires_model_configuration(self):
        with mock.patch.object(ocr_service.settings, "TAMPER_MODE", "production"):
            with mock.patch.object(ocr_service.settings, "YOLO_MODEL_PATH", ""):
                with mock.patch.object(ocr_service.settings, "FACE_MODEL_PATH", ""):
                    with self.assertRaises(RuntimeError):
                        ocr_service.TamperDetectionService().startup_validate()


class DocumentProfileConfigTests(unittest.TestCase):
    def test_bundled_document_profiles_include_tamper_config(self):
        config_dir = Path(__file__).resolve().parents[1] / "config" / "document_types"
        for path in config_dir.glob("*.yaml"):
            profiles = ocr_service.load_document_type_file(path)
            self.assertTrue(profiles, path.name)
            for profile in profiles.values():
                self.assertIsInstance(profile.tamper, ocr_service.TamperConfig)

        national_id = ocr_service.load_document_type_file(config_dir / "nepali_national_id.yaml")["nepali_national_id"]
        self.assertIn("nid_number", national_id.tamper.required_fields)
        self.assertTrue(
            any(item.label == "face" and item.required for item in national_id.tamper.expected_objects)
        )


class OCRCLITests(unittest.TestCase):
    def test_values_only_cli_outputs_extracted_key_values_only(self):
        test_image = Path(__file__).resolve().parents[1] / "testdata" / "citizenship.jpg"
        field = ocr_service.OCRField(
            value="14-09-71-00350",
            confidence=0.99,
            source_text="14-09-71-00350",
        )
        with mock.patch("ocr_service.load_image", return_value=np.zeros((2, 2, 3), dtype=np.uint8)):
            with mock.patch("ocr_service.run_profile_ocr") as run_profile_ocr:
                with mock.patch("ocr_service.extract_structured_fields", return_value={"citizenship_number": field}):
                    with mock.patch("ocr_service.ObjectDetectionService") as detector:
                        with mock.patch("ocr_service.TamperDetectionService") as tamper_detector:
                            tamper_detector.side_effect = AssertionError("values-only should not run tamper detection")
                            detector.side_effect = AssertionError("values-only should not run object detection")
                            run_profile_ocr.return_value = (
                                [],
                                [],
                                "nepali_citizenship_old_front",
                                ocr_service.DocumentProfile(),
                                1.0,
                                np.zeros((2, 2, 3), dtype=np.uint8),
                                {},
                            )
                            buffer = io.StringIO()
                            with mock.patch("sys.stdout", buffer):
                                ocr_service.run_cli(str(test_image), values_only=True)

        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload, {"citizenship_number": "14-09-71-00350"})

    def test_json_cli_includes_tamper_response(self):
        test_image = Path(__file__).resolve().parents[1] / "testdata" / "citizenship.jpg"
        field = ocr_service.OCRField(
            value="14-09-71-00350",
            confidence=0.99,
            source_text="14-09-71-00350",
        )
        with mock.patch("ocr_service.load_image", return_value=np.zeros((2, 2, 3), dtype=np.uint8)):
            with mock.patch("ocr_service.run_profile_ocr") as run_profile_ocr:
                with mock.patch("ocr_service.extract_structured_fields", return_value={"citizenship_number": field}):
                    with mock.patch("ocr_service.ObjectDetectionService") as detector:
                        run_profile_ocr.return_value = (
                            [],
                            [],
                            "nepali_citizenship_old_front",
                            ocr_service.DocumentProfile(),
                            1.0,
                            np.zeros((2, 2, 3), dtype=np.uint8),
                            {},
                        )
                        detector.return_value.detect.return_value = ([], ocr_service.empty_object_summary())
                        buffer = io.StringIO()
                        with mock.patch("sys.stdout", buffer):
                            ocr_service.run_cli(str(test_image), json_output=True)

        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["values"], {"citizenship_number": "14-09-71-00350"})
        self.assertIn("tamper_score", payload)
        self.assertIn("tamper", payload)


if __name__ == "__main__":
    unittest.main()
