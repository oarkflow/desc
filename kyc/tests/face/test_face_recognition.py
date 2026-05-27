import cv2
import numpy as np
from pathlib import Path

from kyc.face.recognizer import FusionRecognizer, SFaceSearcher
from kyc.face.engine import FacePlatform


def make_face(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((160, 160, 3), 180, dtype=np.uint8)
    cv2.circle(image, (80, 78), 58, (130, 130, 130), -1)
    cv2.circle(image, (58, 65), 9, (20, 20, 20), -1)
    cv2.circle(image, (102, 65), 9, (20, 20, 20), -1)
    cv2.ellipse(image, (80, 100), (28, 12), 0, 0, 180, (30, 30, 30), 3)
    noise = rng.normal(0, 3, image.shape).astype(np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def test_cv2_contrib_face_recognizer_is_available():
    assert hasattr(cv2, "face")
    assert hasattr(cv2.face, "LBPHFaceRecognizer_create")


def test_fusion_recognizer_identifies_enrolled_face():
    recognizer = FusionRecognizer(min_confidence=0.5)
    enrolled = make_face(1)

    recognizer.enroll([enrolled], "sample_person")
    result = recognizer.predict(enrolled)

    assert result.label == "sample_person"
    assert result.confidence >= 0.5
    assert not result.is_unknown


def test_sface_search_finds_matching_obama_images_when_fixtures_exist():
    query = Path("kyc/face/tests/test-1.webp")
    folder = Path("kyc/face/tests")
    yunet = Path("kyc/models/face_detection_yunet_2023mar.onnx")
    sface = Path("kyc/models/face_recognition_sface_2021dec.onnx")
    required = [
        query,
        folder / "obama1.jpg",
        folder / "obama_and_biden.jpg",
        yunet,
        sface,
    ]
    if not all(path.exists() for path in required):
        return

    image_paths = [
        str(path)
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        and path != query
    ]
    searcher = SFaceSearcher(str(yunet), str(sface))
    matches = searcher.search(str(query), image_paths)
    matched_paths = {Path(match.image_path).name for match in matches if match.is_match}

    assert "obama1.jpg" in matched_paths
    assert "obama_and_biden.jpg" in matched_paths


def test_yunet_detects_single_obama_face_when_fixture_exists():
    image = Path("kyc/face/tests/obama1.jpg")
    yunet = Path("kyc/models/face_detection_yunet_2023mar.onnx")
    if not image.exists() or not yunet.exists():
        return

    platform = FacePlatform(
        detection_mode="yunet",
        yunet_model_path=str(yunet),
        landmark_mode="region",
        recognition_enabled=False,
    )
    try:
        result = platform.analyze(str(image), return_annotated=False)
    finally:
        platform.close()

    assert result.num_faces == 1
