"""
Face Detection Module
Supports multiple backends:
  - Haar Cascade (always available, bundled with OpenCV)
  - YuNet DNN (requires model file, highest accuracy)
  - SSD DNN (requires Caffe model files)
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class FaceDetection:
    """Represents a single detected face."""
    bbox: Tuple[int, int, int, int]   # x, y, w, h
    confidence: float
    landmarks: Optional[np.ndarray] = None  # 5-point for YuNet
    index: int = 0

    @property
    def x(self): return self.bbox[0]
    @property
    def y(self): return self.bbox[1]
    @property
    def w(self): return self.bbox[2]
    @property
    def h(self): return self.bbox[3]

    def to_rect(self) -> Tuple[int, int, int, int]:
        return self.bbox

    def face_roi(self, image: np.ndarray, padding: float = 0.15) -> np.ndarray:
        """Extract face region with optional padding."""
        x, y, w, h = self.bbox
        ih, iw = image.shape[:2]
        px = int(w * padding)
        py = int(h * padding)
        x1 = max(0, x - px)
        y1 = max(0, y - py)
        x2 = min(iw, x + w + px)
        y2 = min(ih, y + h + py)
        return image[y1:y2, x1:x2]

    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


class HaarCascadeDetector:
    """OpenCV Haar Cascade face detector — always available, no model download needed."""

    def __init__(self, scale_factor: float = 1.05, min_neighbors: int = 4,
                 min_size: Tuple[int, int] = (30, 30)):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        if image is None or image.size == 0:
            return []

        gray = _to_gray(image)
        gray = cv2.equalizeHist(gray)

        detections = []

        # Frontal face detection
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=self.min_size,
        )
        if len(faces) > 0:
            for i, (x, y, w, h) in enumerate(faces):
                detections.append(FaceDetection(
                    bbox=(int(x), int(y), int(w), int(h)),
                    confidence=0.85,
                    index=i,
                ))

        # Profile detection for additional coverage
        profiles = self.profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=self.min_size,
        )
        existing_centers = [d.center() for d in detections]
        for x, y, w, h in profiles if len(profiles) > 0 else []:
            cx, cy = x + w // 2, y + h // 2
            # Avoid duplicate detections
            if not any(abs(cx - ec[0]) < w * 0.5 and abs(cy - ec[1]) < h * 0.5
                       for ec in existing_centers):
                detections.append(FaceDetection(
                    bbox=(int(x), int(y), int(w), int(h)),
                    confidence=0.70,
                    index=len(detections),
                ))

        # Re-index
        for i, d in enumerate(detections):
            d.index = i

        return detections


class YuNetDetector:
    """
    OpenCV YuNet face detector — high accuracy, requires model file.
    Download: https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.6,
                 nms_threshold: float = 0.3, top_k: int = 5000):
        self.detector = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320),
            score_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
        )
        self.conf_threshold = conf_threshold

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, raw = self.detector.detect(image)

        detections = []
        if raw is None:
            return detections

        for i, face in enumerate(raw):
            x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            confidence = float(face[-1])
            # 5 landmark points: right eye, left eye, nose, right mouth corner, left mouth corner
            lms = face[4:14].reshape(5, 2).astype(int) if len(face) >= 14 else None
            detections.append(FaceDetection(
                bbox=(x, y, fw, fh),
                confidence=confidence,
                landmarks=lms,
                index=i,
            ))

        return detections


class MultiScaleDetector:
    """
    Enhanced multi-scale detector combining Haar cascade with image augmentation
    for improved detection rate while staying model-free.
    """

    def __init__(self, min_size: Tuple[int, int] = (30, 30)):
        self._haar = HaarCascadeDetector(scale_factor=1.03, min_neighbors=3, min_size=min_size)

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        all_detections = []

        # Original image
        all_detections.extend(self._haar.detect(image))

        # Try slightly brightened image
        bright = np.clip(image.astype(np.int32) + 30, 0, 255).astype(np.uint8)
        all_detections.extend(self._haar.detect(bright))

        # Try with CLAHE enhancement
        gray = _to_gray(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        all_detections.extend(self._haar.detect(enhanced))

        # NMS to merge overlapping detections
        return _nms(all_detections, iou_threshold=0.4)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _iou(a: FaceDetection, b: FaceDetection) -> float:
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    ix = max(ax, bx)
    iy = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    if ix2 <= ix or iy2 <= iy:
        return 0.0
    inter = (ix2 - ix) * (iy2 - iy)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _nms(detections: List[FaceDetection], iou_threshold: float = 0.4) -> List[FaceDetection]:
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept = []
    for det in sorted_dets:
        if all(_iou(det, k) < iou_threshold for k in kept):
            kept.append(det)

    for i, d in enumerate(kept):
        d.index = i

    return kept
