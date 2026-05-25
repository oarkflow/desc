"""
Visualization Module
Draw face bounding boxes, landmark points, recognition labels, and metrics.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple

from .detector import FaceDetection
from .landmarks import LandmarkResult, LANDMARK_CONNECTIONS
from .recognizer import RecognitionResult


# Color palette (BGR)
COLORS = {
    "bbox_known":   (0, 200, 0),
    "bbox_unknown": (0, 50, 230),
    "bbox_neutral": (220, 180, 0),
    "landmark":     (0, 200, 255),
    "landmark_eye": (50, 255, 100),
    "landmark_lip": (80, 100, 255),
    "landmark_jaw": (180, 180, 180),
    "landmark_nose":(0, 220, 220),
    "connection":   (0, 160, 200),
    "text_bg":      (20, 20, 20),
    "text_fg":      (255, 255, 255),
    "confidence_bar": (0, 220, 120),
}

GROUP_COLORS = {
    "jaw":           (180, 180, 180),
    "right_eyebrow": (255, 200, 0),
    "left_eyebrow":  (255, 200, 0),
    "nose_bridge":   (0, 220, 220),
    "nose_tip":      (0, 220, 220),
    "right_eye":     (50, 255, 100),
    "left_eye":      (50, 255, 100),
    "outer_lips":    (80, 100, 255),
    "inner_lips":    (120, 60, 255),
}


def draw_faces(
    image: np.ndarray,
    detections: List[FaceDetection],
    landmarks_list: Optional[List[LandmarkResult]] = None,
    recognition_list: Optional[List[RecognitionResult]] = None,
    draw_landmarks: bool = True,
    draw_connections: bool = True,
    draw_metrics: bool = False,
) -> np.ndarray:
    """
    Draw all face analysis results onto a copy of the image.
    Returns annotated BGR image.
    """
    canvas = image.copy()

    lm_map = {}
    if landmarks_list:
        for lm in landmarks_list:
            lm_map[lm.face_index] = lm

    rec_map = {}
    if recognition_list:
        for rec in recognition_list:
            pass  # indexed by position
        for i, rec in enumerate(recognition_list):
            rec_map[i] = rec

    for det in detections:
        rec = rec_map.get(det.index)
        lm  = lm_map.get(det.index)

        # Choose box color
        if rec is None:
            color = COLORS["bbox_neutral"]
        elif rec.is_unknown:
            color = COLORS["bbox_unknown"]
        else:
            color = COLORS["bbox_known"]

        _draw_bbox(canvas, det, color, rec)

        if draw_landmarks and lm is not None:
            if draw_connections:
                _draw_connections(canvas, lm)
            _draw_points(canvas, lm)

        if draw_metrics and lm is not None:
            _draw_metrics(canvas, det, lm)

    return canvas


def _draw_bbox(canvas, det: FaceDetection, color: Tuple, rec: Optional[RecognitionResult]):
    x, y, w, h = det.bbox
    thickness = 2

    # Corner-accent style box
    corner = min(w, h) // 5
    pts = [
        ((x, y + corner), (x, y), (x + corner, y)),
        ((x + w - corner, y), (x + w, y), (x + w, y + corner)),
        ((x + w, y + h - corner), (x + w, y + h), (x + w - corner, y + h)),
        ((x + corner, y + h), (x, y + h), (x, y + h - corner)),
    ]
    for p1, mid, p2 in pts:
        cv2.line(canvas, p1, mid, color, thickness + 1, cv2.LINE_AA)
        cv2.line(canvas, mid, p2, color, thickness + 1, cv2.LINE_AA)

    # Label
    if rec is not None:
        label = rec.label if not rec.is_unknown else "Unknown"
        conf_pct = int(rec.confidence * 100)
        text = f"{label}  {conf_pct}%"
    else:
        text = f"Face #{det.index + 1}  {int(det.confidence * 100)}%"

    _draw_label(canvas, text, x, y, color)

    if rec is not None:
        _draw_confidence_bar(canvas, rec.confidence, x, y + h + 4, w)


def _draw_label(canvas, text: str, x: int, y: int, color):
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 5
    ty = max(y - 4, th + pad * 2)
    cv2.rectangle(canvas,
                  (x, ty - th - pad * 2),
                  (x + tw + pad * 2, ty),
                  COLORS["text_bg"], -1)
    cv2.putText(canvas, text,
                (x + pad, ty - pad),
                font, scale, color, thickness, cv2.LINE_AA)


def _draw_confidence_bar(canvas, confidence: float, x: int, y: int, width: int):
    bar_h = 5
    cv2.rectangle(canvas, (x, y), (x + width, y + bar_h), (60, 60, 60), -1)
    filled = int(width * confidence)
    bar_color = (
        (0, 220, 120) if confidence > 0.7 else
        (0, 190, 255) if confidence > 0.4 else
        (0, 80, 240)
    )
    cv2.rectangle(canvas, (x, y), (x + filled, y + bar_h), bar_color, -1)


def _draw_connections(canvas, lm: LandmarkResult):
    if lm.mode != "lbf" or len(lm.points) < 68:
        return

    pts = lm.points.astype(int)
    for chain in LANDMARK_CONNECTIONS:
        for j in range(len(chain) - 1):
            a, b = chain[j], chain[j + 1]
            if a < len(pts) and b < len(pts):
                cv2.line(canvas, tuple(pts[a]), tuple(pts[b]),
                         COLORS["connection"], 1, cv2.LINE_AA)


def _draw_points(canvas, lm: LandmarkResult):
    pts = lm.points.astype(int)

    if lm.mode == "lbf" and lm.groups:
        for group_name, group_pts in lm.groups.items():
            color = GROUP_COLORS.get(group_name, COLORS["landmark"])
            for pt in group_pts.astype(int):
                cv2.circle(canvas, tuple(pt), 2, color, -1, cv2.LINE_AA)
    else:
        for pt in pts:
            cv2.circle(canvas, tuple(pt), 4, COLORS["landmark"], -1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(pt), 4, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_metrics(canvas, det: FaceDetection, lm: LandmarkResult):
    x, y, w, h = det.bbox
    metrics = []
    ed = lm.eye_distance()
    if ed is not None:
        metrics.append(f"Eye dist: {ed:.1f}px")
    fw = lm.face_width()
    if fw is not None:
        metrics.append(f"Face W: {fw:.1f}px")
    yaw = lm.yaw_estimate()
    if yaw is not None:
        metrics.append(f"Yaw: {yaw:+.1f}°")
    mor = lm.mouth_open_ratio()
    if mor is not None:
        metrics.append(f"Mouth: {mor:.2f}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, txt in enumerate(metrics):
        ty = y + h + 22 + i * 18
        cv2.putText(canvas, txt, (x, ty), font, 0.42, (200, 255, 200), 1, cv2.LINE_AA)


def save_image(image: np.ndarray, path: str) -> None:
    cv2.imwrite(path, image)


def display_image(image: np.ndarray, window_name: str = "Face Analysis",
                  wait: bool = True) -> None:
    cv2.imshow(window_name, image)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
