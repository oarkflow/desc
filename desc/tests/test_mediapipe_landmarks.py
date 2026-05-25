import numpy as np

from face.landmarks import LandmarkResult
from face.visualizer import draw_faces
from face.detector import FaceDetection


def test_478_point_landmark_result_preserves_prebuilt_groups():
    points = np.zeros((478, 2), dtype=float)
    groups = {
        "left_iris": points[[473, 474, 475, 476, 477]],
        "right_iris": points[[468, 469, 470, 471, 472]],
    }

    result = LandmarkResult(points=points, groups=groups, mode="mediapipe_478")

    assert result.groups.keys() == groups.keys()
    assert result.to_dict()["num_points"] == 478
    assert result.to_dict()["groups"]["left_iris"] == groups["left_iris"].tolist()


def test_visualizer_draws_478_point_landmarks_without_error():
    image = np.zeros((240, 240, 3), dtype=np.uint8)
    points = np.column_stack([
        np.linspace(40, 200, 478),
        np.linspace(60, 180, 478),
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
