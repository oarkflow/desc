"""
Facial Landmark Module
Provides three backends:
  - MediaPipe (478-point): FaceLandmarker Tasks API with iris landmarks
  - LBF (68-point): cv2.face.FacemarkLBF, requires lbfmodel.yaml
  - Region-based (6-point): eyes, nose, mouth using bundled Haar cascades
"""

from __future__ import annotations

import cv2
import numpy as np
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

MEDIAPIPE_LANDMARK_COUNT = 478

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    mp = None
    mp_python = None
    mp_vision = None


# ─── Named landmark indices (68-point dlib/LBF convention) ───────────────────
LANDMARK_GROUPS = {
    "jaw":           list(range(0, 17)),
    "right_eyebrow": list(range(17, 22)),
    "left_eyebrow":  list(range(22, 27)),
    "nose_bridge":   list(range(27, 31)),
    "nose_tip":      list(range(31, 36)),
    "right_eye":     list(range(36, 42)),
    "left_eye":      list(range(42, 48)),
    "outer_lips":    list(range(48, 60)),
    "inner_lips":    list(range(60, 68)),
}

# Connectivity for drawing
LANDMARK_CONNECTIONS = [
    list(range(0, 17)),        # jaw
    list(range(17, 22)),       # right brow
    list(range(22, 27)),       # left brow
    list(range(27, 31)),       # nose bridge
    list(range(31, 36)),       # nose tip
    list(range(36, 42)) + [36],  # right eye (closed)
    list(range(42, 48)) + [42],  # left eye (closed)
    list(range(48, 60)) + [48],  # outer lips (closed)
    list(range(60, 68)) + [60],  # inner lips (closed)
]


MP_LANDMARK_GROUPS = {
    "silhouette": [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
        365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
        132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
    ],
    "left_eye": [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398],
    "right_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "left_eyebrow": [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
    "right_eyebrow": [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
    "left_iris": [473, 474, 475, 476, 477],
    "right_iris": [468, 469, 470, 471, 472],
    "lips_outer": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
    "lips_inner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191],
    "nose": [1, 2, 98, 327, 4, 5, 6, 122, 351, 196, 419, 3, 51, 281, 248, 456],
    "face_oval": [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
        365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
        132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    ],
}


@dataclass
class LandmarkResult:
    """Full facial landmark result for one face."""
    points: np.ndarray           # (N, 2) float array
    groups: Dict[str, np.ndarray] = field(default_factory=dict)
    mode: str = "lbf"            # 'lbf' | 'region'
    face_index: int = 0
    blendshapes: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.points is not None:
            if len(self.points) == 68:
                self.groups = {
                    name: self.points[idx]
                    for name, idx in LANDMARK_GROUPS.items()
                }
            # MediaPipe groups are pre-built because they use a different
            # convention and include iris points.

    # ── Derived measurements ─────────────────────────────────────────────────
    def eye_distance(self) -> Optional[float]:
        if "right_eye" in self.groups and "left_eye" in self.groups:
            rc = self.groups["right_eye"].mean(axis=0)
            lc = self.groups["left_eye"].mean(axis=0)
            return float(np.linalg.norm(rc - lc))
        return None

    def face_width(self) -> Optional[float]:
        if "jaw" in self.groups and len(self.groups["jaw"]) >= 2:
            return float(self.groups["jaw"][-1][0] - self.groups["jaw"][0][0])
        if "face_oval" in self.groups and len(self.groups["face_oval"]) >= 2:
            xs = self.groups["face_oval"][:, 0]
            return float(xs.max() - xs.min())
        return None

    def mouth_open_ratio(self) -> Optional[float]:
        """Approximation: vertical mouth opening / face height."""
        if (
            "outer_lips" in self.groups
            and "jaw" in self.groups
            and len(self.groups["outer_lips"]) >= 10
            and len(self.groups["jaw"]) >= 9
        ):
            top = self.groups["outer_lips"][3][1]   # top lip center
            bot = self.groups["outer_lips"][9][1]   # bottom lip center
            jaw_h = self.groups["jaw"][8][1] - self.groups["jaw"][0][1]
            return float((bot - top) / jaw_h) if jaw_h > 0 else None
        if (
            "lips_inner" in self.groups
            and "face_oval" in self.groups
            and len(self.groups["lips_inner"]) >= 2
        ):
            lips = self.groups["lips_inner"]
            face = self.groups["face_oval"]
            mouth_h = float(lips[:, 1].max() - lips[:, 1].min())
            face_h = float(face[:, 1].max() - face[:, 1].min())
            return float(mouth_h / face_h) if face_h > 0 else None
        return None

    def yaw_estimate(self) -> Optional[float]:
        """Rough left-right head turn (degrees) from nose & jaw symmetry."""
        if (
            "jaw" in self.groups
            and "nose_bridge" in self.groups
            and len(self.groups["jaw"]) >= 2
        ):
            jaw = self.groups["jaw"]
            nose = self.groups["nose_bridge"].mean(axis=0)
            jaw_mid_x = (jaw[0][0] + jaw[-1][0]) / 2
            dx = float(nose[0] - jaw_mid_x)
            face_w = float(jaw[-1][0] - jaw[0][0])
            return float(np.degrees(np.arctan2(dx, face_w))) if face_w > 0 else None
        if (
            "left_eye" in self.groups
            and "right_eye" in self.groups
            and "nose" in self.groups
        ):
            left_eye = self.groups["left_eye"].mean(axis=0)
            right_eye = self.groups["right_eye"].mean(axis=0)
            nose = self.groups["nose"].mean(axis=0)
            eye_mid_x = float((left_eye[0] + right_eye[0]) / 2)
            eye_dist = float(np.linalg.norm(left_eye - right_eye))
            if eye_dist > 0:
                return float(np.degrees(np.arctan2(float(nose[0] - eye_mid_x), eye_dist)))
        return None

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "face_index": self.face_index,
            "num_points": len(self.points),
            "points": self.points.tolist(),
            "groups": {k: v.tolist() for k, v in self.groups.items()},
            "metrics": {
                "eye_distance": self.eye_distance(),
                "face_width": self.face_width(),
                "mouth_open_ratio": self.mouth_open_ratio(),
                "yaw_estimate_deg": self.yaw_estimate(),
            },
            "blendshapes": {k: round(v, 4) for k, v in self.blendshapes.items()},
        }


class LBFLandmarkDetector:
    """
    68-point landmark detector using OpenCV's FacemarkLBF.
    Requires: lbfmodel.yaml (≈ 54 MB)
    Download: https://github.com/kurnianggoro/GSOC2017/blob/master/data/lbfmodel.yaml
    """

    def __init__(self, model_path: str):
        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel(str(model_path))

    def detect(self, image: np.ndarray,
               face_bboxes: List[Tuple[int, int, int, int]]) -> List[LandmarkResult]:
        if not face_bboxes or image is None:
            return []

        rects = np.array([[x, y, w, h] for (x, y, w, h) in face_bboxes])
        ok, landmarks = self.facemark.fit(image, rects)
        if not ok or landmarks is None:
            return []

        results = []
        for i, lm in enumerate(landmarks):
            pts = lm[0].reshape(-1, 2).astype(float)
            results.append(LandmarkResult(points=pts, mode="lbf", face_index=i))
        return results


class MediaPipeLandmarkDetector:
    """
    478-point landmark detector using MediaPipe FaceLandmarker Tasks API.
    Adds iris landmarks (points 468-477) on top of the face mesh.
    """

    def __init__(self, model_path: str, num_faces: int = 10):
        if mp is None or mp_python is None or mp_vision is None:
            raise ImportError("mediapipe is not installed")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"MediaPipe model not found: {model_path}")

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=num_faces,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._num_faces = num_faces

    def detect(self, image: np.ndarray, face_bboxes=None) -> List["LandmarkResult"]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = self.landmarker.detect(mp_image)

        results = []
        h, w = image.shape[:2]
        for i, face_landmarks in enumerate(detection_result.face_landmarks):
            pts = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks], dtype=float)
            if len(pts) != MEDIAPIPE_LANDMARK_COUNT:
                raise RuntimeError(
                    f"MediaPipe FaceLandmarker returned {len(pts)} points; "
                    f"expected {MEDIAPIPE_LANDMARK_COUNT}."
                )
            groups: Dict[str, np.ndarray] = {}
            for name, indices in MP_LANDMARK_GROUPS.items():
                valid = [idx for idx in indices if idx < len(pts)]
                if valid:
                    groups[name] = pts[valid]
            blendshapes = {}
            if i < len(detection_result.face_blendshapes):
                blendshapes = {
                    item.category_name: float(item.score)
                    for item in detection_result.face_blendshapes[i]
                }

            results.append(LandmarkResult(
                points=pts,
                groups=groups,
                mode="mediapipe_478",
                face_index=i,
                blendshapes=blendshapes,
            ))

        return results

    def close(self) -> None:
        if getattr(self, "landmarker", None) is not None:
            self.landmarker.close()
            self.landmarker = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class RegionLandmarkDetector:
    """
    Lightweight landmark detector using OpenCV Haar cascades.
    Returns 6 key points: left eye center, right eye center,
    nose tip estimate, mouth center, jaw left, jaw right.
    Works with bundled OpenCV data — no extra downloads needed.
    """

    def __init__(self):
        self._eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        self._eye_glasses_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        )
        self._smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )

    def detect(self, image: np.ndarray,
               face_bboxes: List[Tuple[int, int, int, int]]) -> List[LandmarkResult]:
        results = []
        gray = _gray(image)

        for i, (fx, fy, fw, fh) in enumerate(face_bboxes):
            face_gray = gray[fy:fy + fh, fx:fx + fw]
            pts = self._extract_points(face_gray, fx, fy, fw, fh)
            groups: Dict[str, np.ndarray] = {}

            if len(pts) >= 2:
                groups["right_eye"] = np.array([pts[0]])
                groups["left_eye"]  = np.array([pts[1]])
            if len(pts) >= 3:
                groups["nose_tip"] = np.array([pts[2]])
            if len(pts) >= 4:
                groups["outer_lips"] = np.array([pts[3]])
            if len(pts) >= 6:
                groups["jaw"] = np.array([pts[4], pts[5]])

            r = LandmarkResult(
                points=np.array(pts, dtype=float),
                groups=groups,
                mode="region",
                face_index=i,
            )
            results.append(r)

        return results

    def _extract_points(self, face_gray, fx, fy, fw, fh) -> List[List[float]]:
        pts: List[List[float]] = []

        # Detect eyes in the upper half of the face
        upper = face_gray[: fh // 2, :]
        eyes = self._eye_cascade.detectMultiScale(upper, 1.1, 5, minSize=(15, 15))
        if len(eyes) == 0:
            eyes = self._eye_glasses_cascade.detectMultiScale(
                upper, 1.1, 5, minSize=(15, 15)
            )

        # Sort eyes by x: right eye first (smaller x in image)
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])[:2]
            for ex, ey, ew, eh in eyes:
                pts.append([fx + ex + ew / 2, fy + ey + eh / 2])
        elif len(eyes) == 1:
            ex, ey, ew, eh = eyes[0]
            pts.append([fx + ex + ew / 2, fy + ey + eh / 2])
            pts.append([fx + fw / 2, fy + fh * 0.3])  # estimate opposite
        else:
            pts.append([fx + fw * 0.33, fy + fh * 0.3])
            pts.append([fx + fw * 0.67, fy + fh * 0.3])

        # Nose tip estimate
        pts.append([fx + fw / 2, fy + fh * 0.55])

        # Mouth/smile detection in lower half
        lower = face_gray[fh // 2 :, :]
        smiles = self._smile_cascade.detectMultiScale(lower, 1.7, 20, minSize=(25, 15))
        if len(smiles) > 0:
            sx, sy, sw, sh = smiles[0]
            pts.append([fx + sx + sw / 2, fy + fh // 2 + sy + sh / 2])
        else:
            pts.append([fx + fw / 2, fy + fh * 0.75])

        # Jaw left / right estimates
        pts.append([float(fx + fw * 0.05), float(fy + fh * 0.85)])
        pts.append([float(fx + fw * 0.95), float(fy + fh * 0.85)])

        return pts


def _gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
