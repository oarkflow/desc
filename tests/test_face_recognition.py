import cv2
import numpy as np

from face.recognizer import FusionRecognizer


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
