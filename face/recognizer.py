"""
Face Recognition Module
Backends:
  - LBPH  : Local Binary Pattern Histogram (fast, no extra models)
  - Eigen : Eigenfaces / PCA
  - Fisher: Fisherfaces / LDA
  - HOG   : Histogram of Oriented Gradients cosine-similarity (used as embedding)
All recognizers support enrollment (add known faces) and predict (identify unknown face).
"""

from __future__ import annotations

import cv2
import numpy as np
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any


FACE_SIZE = (128, 128)   # normalize all faces to this size


@dataclass
class RecognitionResult:
    label: str
    confidence: float        # higher = more confident (0-1)
    distance: float          # raw distance/score (lower=better for LBPH/Eigen)
    method: str

    @property
    def is_unknown(self) -> bool:
        return self.label == "unknown"

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "distance": round(self.distance, 4),
            "method": self.method,
            "is_unknown": self.is_unknown,
        }


def _preprocess(face_img: np.ndarray) -> np.ndarray:
    """Normalize a face ROI for recognition."""
    if face_img is None or face_img.size == 0:
        return np.zeros((*FACE_SIZE,), dtype=np.uint8)

    gray = face_img if len(face_img.shape) == 2 else cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, FACE_SIZE, interpolation=cv2.INTER_LANCZOS4)

    # CLAHE equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(resized)

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(eq, (3, 3), 0)
    return blurred


def _hog_embedding(face_img: np.ndarray) -> np.ndarray:
    """Compute HOG-based face embedding vector."""
    face = _preprocess(face_img)
    hog = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    descriptor = hog.compute(face).flatten()
    norm = np.linalg.norm(descriptor)
    return descriptor / norm if norm > 0 else descriptor


class LBPHRecognizer:
    """
    Local Binary Pattern Histogram face recognizer.
    Robust to lighting changes. Best for small (<100) face sets.
    threshold: predictions above this are marked 'unknown'
    """

    def __init__(self, threshold: float = 80.0):
        self._model = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=threshold
        )
        self._labels: Dict[int, str] = {}
        self._trained = False
        self.threshold = threshold

    def enroll(self, face_images: List[np.ndarray], label: str) -> None:
        """Add faces for a person. Call train() after all enrollments."""
        label_id = self._label_to_id(label)
        imgs = [_preprocess(f) for f in face_images]
        ids = [label_id] * len(imgs)
        if self._trained:
            self._model.update(imgs, np.array(ids))
        else:
            self._model.train(imgs, np.array(ids))
            self._trained = True

    def predict(self, face_img: np.ndarray) -> RecognitionResult:
        if not self._trained:
            return RecognitionResult("unknown", 0.0, 9999.0, "lbph")

        face = _preprocess(face_img)
        label_id, distance = self._model.predict(face)
        label = self._labels.get(label_id, "unknown")

        if distance > self.threshold:
            label = "unknown"

        # Confidence: map distance [0, threshold*2] → [1, 0]
        confidence = max(0.0, 1.0 - distance / (self.threshold * 2))
        return RecognitionResult(label, confidence, float(distance), "lbph")

    def _label_to_id(self, label: str) -> int:
        for k, v in self._labels.items():
            if v == label:
                return k
        new_id = max(self._labels.keys(), default=-1) + 1
        self._labels[new_id] = label
        return new_id

    def save(self, path: str) -> None:
        self._model.save(str(path) + "_lbph.yml")
        with open(str(path) + "_meta.pkl", "wb") as f:
            pickle.dump({"labels": self._labels, "trained": self._trained,
                         "threshold": self.threshold}, f)

    def load(self, path: str) -> None:
        self._model.read(str(path) + "_lbph.yml")
        with open(str(path) + "_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self._labels = meta["labels"]
        self._trained = meta["trained"]
        self.threshold = meta["threshold"]

    @property
    def known_labels(self) -> List[str]:
        return list(self._labels.values())


class HOGRecognizer:
    """
    HOG embedding + cosine similarity face recognizer.
    Averages multiple embeddings per person for robust matching.
    Good when LBPH confidence is ambiguous.
    threshold: minimum cosine similarity to consider a match (0-1)
    """

    def __init__(self, threshold: float = 0.50):
        self._embeddings: Dict[str, List[np.ndarray]] = {}
        self._mean_embeddings: Dict[str, np.ndarray] = {}
        self.threshold = threshold

    def enroll(self, face_images: List[np.ndarray], label: str) -> None:
        embs = [_hog_embedding(f) for f in face_images]
        self._embeddings.setdefault(label, []).extend(embs)
        all_embs = np.array(self._embeddings[label])
        self._mean_embeddings[label] = all_embs.mean(axis=0)

    def predict(self, face_img: np.ndarray) -> RecognitionResult:
        if not self._mean_embeddings:
            return RecognitionResult("unknown", 0.0, 0.0, "hog")

        query = _hog_embedding(face_img)
        best_label = "unknown"
        best_sim = -1.0

        for label, mean_emb in self._mean_embeddings.items():
            sim = float(np.dot(query, mean_emb) /
                        (np.linalg.norm(query) * np.linalg.norm(mean_emb) + 1e-9))
            if sim > best_sim:
                best_sim = sim
                best_label = label

        if best_sim < self.threshold:
            best_label = "unknown"
            confidence = 0.0
        else:
            confidence = (best_sim - self.threshold) / (1.0 - self.threshold)

        return RecognitionResult(best_label, confidence, best_sim, "hog")

    def save(self, path: str) -> None:
        with open(str(path) + "_hog.pkl", "wb") as f:
            pickle.dump({
                "embeddings": self._embeddings,
                "mean_embeddings": self._mean_embeddings,
                "threshold": self.threshold,
            }, f)

    def load(self, path: str) -> None:
        with open(str(path) + "_hog.pkl", "rb") as f:
            data = pickle.load(f)
        self._embeddings = data["embeddings"]
        self._mean_embeddings = data["mean_embeddings"]
        self.threshold = data["threshold"]

    @property
    def known_labels(self) -> List[str]:
        return list(self._mean_embeddings.keys())


class FusionRecognizer:
    """
    Ensemble recognizer: combines LBPH + HOG predictions using weighted voting.
    Better accuracy than either alone.
    """

    def __init__(
        self,
        lbph_threshold: float = 80.0,
        hog_threshold: float = 0.50,
        min_confidence: float = 0.50,
    ):
        self.lbph = LBPHRecognizer(threshold=lbph_threshold)
        self.hog = HOGRecognizer(threshold=hog_threshold)
        self.min_confidence = min_confidence

    def enroll(self, face_images: List[np.ndarray], label: str) -> None:
        self.lbph.enroll(face_images, label)
        self.hog.enroll(face_images, label)

    def predict(self, face_img: np.ndarray) -> RecognitionResult:
        r_lbph = self.lbph.predict(face_img)
        r_hog  = self.hog.predict(face_img)

        # Agreement boost
        if not r_lbph.is_unknown and not r_hog.is_unknown:
            if r_lbph.label == r_hog.label:
                combined_conf = (r_lbph.confidence * 0.5 + r_hog.confidence * 0.5) * 1.15
                return self._accept_or_unknown(RecognitionResult(
                    r_lbph.label, min(1.0, combined_conf),
                    r_lbph.distance, "fusion[agree]"
                ))
            # Disagreement: pick higher confidence
            if r_lbph.confidence >= r_hog.confidence:
                return self._accept_or_unknown(RecognitionResult(
                    r_lbph.label, r_lbph.confidence * 0.80,
                    r_lbph.distance, "fusion[lbph-wins]"
                ))
            return self._accept_or_unknown(RecognitionResult(
                r_hog.label, r_hog.confidence * 0.80,
                r_hog.distance, "fusion[hog-wins]"
            ))

        if not r_lbph.is_unknown:
            return self._accept_or_unknown(RecognitionResult(
                r_lbph.label, r_lbph.confidence * 0.75,
                r_lbph.distance, "fusion[lbph-only]"
            ))
        if not r_hog.is_unknown:
            return self._accept_or_unknown(RecognitionResult(
                r_hog.label, r_hog.confidence * 0.75,
                r_hog.distance, "fusion[hog-only]"
            ))

        return RecognitionResult("unknown", 0.0, r_lbph.distance, "fusion[unknown]")

    def _accept_or_unknown(self, result: RecognitionResult) -> RecognitionResult:
        if result.confidence >= self.min_confidence:
            return result
        return RecognitionResult("unknown", result.confidence, result.distance, result.method)

    def save(self, path: str) -> None:
        self.lbph.save(path)
        self.hog.save(path)

    def load(self, path: str) -> None:
        self.lbph.load(path)
        self.hog.load(path)

    @property
    def known_labels(self) -> List[str]:
        return list(set(self.lbph.known_labels + self.hog.known_labels))
