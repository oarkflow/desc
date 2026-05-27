import numpy as np
import pytest

from kyc.face.engine import FacePlatform
from kyc.face.landmarks import LandmarkResult, MEDIAPIPE_LANDMARK_COUNT
from kyc.face.visualizer import draw_faces
from kyc.face.detector import FaceDetection


def test_478_point_landmark_result_preserves_prebuilt_groups():
    points = np.zeros((MEDIAPIPE_LANDMARK_COUNT, 2), dtype=float)
    groups = {
        "left_iris": points[[473, 474, 475, 476, 477]],
        "right_iris": points[[468, 469, 470, 471, 472]],
    }

    result = LandmarkResult(points=points, groups=groups, mode="mediapipe_478")

    assert result.groups.keys() == groups.keys()
    assert result.to_dict()["num_points"] == MEDIAPIPE_LANDMARK_COUNT
    assert result.to_dict()["groups"]["left_iris"] == groups["left_iris"].tolist()


def test_visualizer_draws_478_point_landmarks_without_error():
    image = np.zeros((240, 240, 3), dtype=np.uint8)
    points = np.column_stack([
        np.linspace(40, 200, MEDIAPIPE_LANDMARK_COUNT),
        np.linspace(60, 180, MEDIAPIPE_LANDMARK_COUNT),
    ])
    groups = {
        "left_iris": points[[473, 474, 475, 476, 477]],
        "right_iris": points[[468, 469, 470, 471, 472]],
        "face_oval": points[[10, 338, 297, 332]],
    }
    landmarks = [LandmarkResult(points=points, groups=groups, mode="mediapipe_478", face_index=0)]
    detections = [FaceDetection(bbox=(40, 40, 160, 160), confidence=0.9, index=0)]

    annotated = draw_faces(image, detections, landmarks)

    assert annotated.shape == image.shape
    assert np.any(annotated != image)


def test_mediapipe_mode_requires_model_path():
    with pytest.raises(FileNotFoundError, match="478-point"):
        FacePlatform(landmark_mode="mediapipe", recognition_enabled=False)


def test_mediapipe_mode_rejects_non_478_output(monkeypatch):
    class FakeMediaPipeDetector:
        def __init__(self, model_path):
            pass

        def detect(self, image):
            return [LandmarkResult(points=np.zeros((468, 2)), mode="mediapipe_478")]

        def close(self):
            pass

    monkeypatch.setattr("kyc.face.engine.MediaPipeLandmarkDetector", FakeMediaPipeDetector)
    platform = FacePlatform(
        mediapipe_model_path="fake.task",
        landmark_mode="mediapipe",
        recognition_enabled=False,
    )
    platform.detector.detect = lambda image: [FaceDetection(bbox=(10, 10, 80, 80), confidence=0.9, index=0)]

    with pytest.raises(RuntimeError, match="478"):
        platform.analyze(np.zeros((120, 120, 3), dtype=np.uint8), return_annotated=False)
