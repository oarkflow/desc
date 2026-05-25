"""
FacePlatform — Main Engine
Ties together: detection, landmark extraction, face recognition.
"""

from __future__ import annotations

import json
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .detector import (
    FaceDetection, HaarCascadeDetector, YuNetDetector,
    MultiScaleDetector, _nms,
)
from .landmarks import LBFLandmarkDetector, RegionLandmarkDetector, LandmarkResult
from .recognizer import FusionRecognizer, RecognitionResult
from .image_loader import load_image, image_info
from .visualizer import draw_faces, save_image


@dataclass
class AnalysisResult:
    """Complete analysis result for a single image."""
    image_path: str
    image_info: dict
    faces: List[FaceDetection]
    landmarks: List[LandmarkResult]
    recognitions: List[RecognitionResult]
    annotated_image: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "image_info": self.image_info,
            "num_faces": self.num_faces,
            "faces": [
                {
                    "index": f.index,
                    "bbox": list(f.bbox),
                    "confidence": round(f.confidence, 4),
                    "landmarks": self.landmarks[i].to_dict() if i < len(self.landmarks) else None,
                    "recognition": self.recognitions[i].to_dict() if i < len(self.recognitions) else None,
                }
                for i, f in enumerate(self.faces)
            ],
        }

    def print_summary(self) -> None:
        print(f"\n{'═'*60}")
        print(f"  Image : {self.image_path}")
        print(f"  Size  : {self.image_info['width']}×{self.image_info['height']}  "
              f"({self.image_info['size_mp']} MP)")
        print(f"  Faces : {self.num_faces} detected")
        print(f"{'─'*60}")

        for i, face in enumerate(self.faces):
            print(f"\n  Face #{i+1}  bbox={face.bbox}  conf={face.confidence:.2f}")

            if i < len(self.landmarks):
                lm = self.landmarks[i]
                print(f"  Landmarks : {len(lm.points)} points ({lm.mode})")
                ed = lm.eye_distance()
                yaw = lm.yaw_estimate()
                if ed: print(f"             eye-dist={ed:.1f}px", end="")
                if yaw: print(f"  yaw={yaw:+.1f}°", end="")
                if ed or yaw: print()

            if i < len(self.recognitions):
                rec = self.recognitions[i]
                status = "✓ KNOWN" if not rec.is_unknown else "? UNKNOWN"
                print(f"  Identity  : {status} → {rec.label}  "
                      f"confidence={rec.confidence:.2%}  [{rec.method}]")

        print(f"{'═'*60}\n")


class FacePlatform:
    """
    All-in-one face analysis platform.

    Usage:
        platform = FacePlatform(lbf_model_path="lbfmodel.yaml")
        platform.enroll_from_folder("alice", "photos/alice/")
        result = platform.analyze("photo.jpg", save_annotated="output.jpg")
        result.print_summary()
    """

    def __init__(
        self,
        lbf_model_path: Optional[str] = None,
        yunet_model_path: Optional[str] = None,
        recognizer_db_path: Optional[str] = None,
        detection_mode: str = "multiscale",   # 'haar' | 'multiscale' | 'yunet'
        landmark_mode: str = "auto",          # 'auto' | 'lbf' | 'region'
        min_face_size: int = 30,
        recognition_enabled: bool = True,
        lbph_threshold: float = 80.0,
        hog_threshold: float = 0.50,
    ):
        # ── Detector ─────────────────────────────────────────────────────────
        if detection_mode == "yunet" and yunet_model_path:
            self.detector = YuNetDetector(yunet_model_path)
        elif detection_mode == "multiscale":
            self.detector = MultiScaleDetector(min_size=(min_face_size, min_face_size))
        else:
            self.detector = HaarCascadeDetector(min_size=(min_face_size, min_face_size))

        # ── Landmark detector ─────────────────────────────────────────────────
        self._lbf_path = lbf_model_path
        self._landmark_mode = landmark_mode
        self._lbf_detector: Optional[LBFLandmarkDetector] = None
        self._region_detector = RegionLandmarkDetector()

        if landmark_mode in ("auto", "lbf") and lbf_model_path:
            try:
                self._lbf_detector = LBFLandmarkDetector(lbf_model_path)
                print(f"[FacePlatform] LBF 68-point landmark model loaded ✓")
            except Exception as e:
                print(f"[FacePlatform] LBF model failed ({e}), using region fallback")

        # ── Recognizer ────────────────────────────────────────────────────────
        self.recognizer = FusionRecognizer(lbph_threshold, hog_threshold) if recognition_enabled else None
        self._recognition_enabled = recognition_enabled

        if recognizer_db_path and Path(recognizer_db_path + "_lbph.yml").exists():
            self.recognizer.load(recognizer_db_path)
            print(f"[FacePlatform] Recognizer DB loaded — "
                  f"known faces: {self.recognizer.known_labels}")

    # ── Enrollment ────────────────────────────────────────────────────────────
    def enroll(self, label: str, face_images: List[np.ndarray]) -> None:
        """Add a person's face images to the recognition database."""
        if not self._recognition_enabled or not self.recognizer:
            raise RuntimeError("Recognition is disabled.")
        if not face_images:
            raise ValueError("No face images provided.")
        self.recognizer.enroll(face_images, label)
        print(f"[FacePlatform] Enrolled '{label}' with {len(face_images)} image(s)")

    def enroll_from_image(self, label: str, image_path: str) -> None:
        """Detect faces in an image and enroll them under the given label."""
        img = load_image(image_path)
        dets = self.detector.detect(img)
        if not dets:
            raise ValueError(f"No face found in {image_path}")
        face = max(dets, key=lambda d: d.w * d.h)
        self.enroll(label, [face.face_roi(img)])

    def enroll_from_folder(self, label: str, folder_path: str,
                           extensions=(".jpg", ".jpeg", ".png", ".bmp", ".webp")) -> int:
        """Enroll all face images from a folder."""
        folder = Path(folder_path)
        images = [f for f in folder.iterdir()
                  if f.suffix.lower() in extensions]
        if not images:
            raise FileNotFoundError(f"No images found in {folder_path}")

        rois = []
        for img_path in images:
            try:
                img = load_image(str(img_path))
                dets = self.detector.detect(img)
                if dets:
                    face = max(dets, key=lambda d: d.w * d.h)
                    rois.append(face.face_roi(img))
            except Exception as e:
                print(f"  [warn] {img_path.name}: {e}")

        if not rois:
            raise ValueError(f"No faces found in any image in {folder_path}")

        self.enroll(label, rois)
        return len(rois)

    def save_database(self, path: str) -> None:
        """Persist the recognizer database to disk."""
        if self.recognizer:
            self.recognizer.save(path)
            print(f"[FacePlatform] Database saved to {path}*")

    def load_database(self, path: str) -> None:
        """Load a previously saved recognizer database."""
        if self.recognizer:
            self.recognizer.load(path)

    # ── Analysis ──────────────────────────────────────────────────────────────
    def analyze(
        self,
        image_or_path,
        save_annotated: Optional[str] = None,
        draw_metrics: bool = False,
        return_annotated: bool = True,
    ) -> AnalysisResult:
        """
        Run full face analysis on an image.

        Args:
            image_or_path: file path (str/Path) OR numpy BGR array
            save_annotated: if given, saves annotated image to this path
            draw_metrics: overlay landmark metrics on the image
            return_annotated: include annotated image in result

        Returns:
            AnalysisResult with detections, landmarks, and (if enrolled) recognitions
        """
        if isinstance(image_or_path, (str, Path)):
            image_path = str(image_or_path)
            image = load_image(image_path)
        else:
            image = image_or_path
            image_path = "<array>"

        info = image_info(image)

        # 1. Detection
        detections = self.detector.detect(image)

        # 2. Landmark extraction
        landmarks = []
        if detections:
            bboxes = [d.bbox for d in detections]
            if self._lbf_detector:
                try:
                    landmarks = self._lbf_detector.detect(image, bboxes)
                except Exception:
                    landmarks = self._region_detector.detect(image, bboxes)
            else:
                landmarks = self._region_detector.detect(image, bboxes)

        # 3. Recognition
        recognitions = []
        if self._recognition_enabled and self.recognizer and self.recognizer.known_labels:
            for det in detections:
                roi = det.face_roi(image)
                rec = self.recognizer.predict(roi)
                recognitions.append(rec)

        # 4. Visualization
        annotated = None
        if return_annotated or save_annotated:
            annotated = draw_faces(
                image, detections, landmarks,
                recognitions if recognitions else None,
                draw_landmarks=True,
                draw_connections=True,
                draw_metrics=draw_metrics,
            )
            if save_annotated:
                save_image(annotated, save_annotated)

        return AnalysisResult(
            image_path=image_path,
            image_info=info,
            faces=detections,
            landmarks=landmarks,
            recognitions=recognitions,
            annotated_image=annotated if return_annotated else None,
        )

    def analyze_batch(
        self,
        image_paths: List[str],
        output_folder: Optional[str] = None,
        draw_metrics: bool = False,
    ) -> List[AnalysisResult]:
        """Analyze multiple images in sequence."""
        results = []
        out = Path(output_folder) if output_folder else None
        if out:
            out.mkdir(parents=True, exist_ok=True)

        for i, path in enumerate(image_paths, 1):
            print(f"  [{i}/{len(image_paths)}] {path}")
            try:
                save_path = str(out / Path(path).name) if out else None
                result = self.analyze(path, save_annotated=save_path,
                                      draw_metrics=draw_metrics)
                results.append(result)
            except Exception as e:
                print(f"  [error] {path}: {e}")

        return results

    @property
    def known_faces(self) -> List[str]:
        if self.recognizer:
            return self.recognizer.known_labels
        return []
