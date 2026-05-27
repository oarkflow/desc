from io import BytesIO

import cv2
import numpy as np
from PIL import Image


class TamperAnalyzer:
    def analyze(self, image, image_bytes: bytes):
        signals = []
        score_parts = []

        metadata_signal = self._metadata_signal(image_bytes)
        signals.append(metadata_signal)
        score_parts.append(metadata_signal["score"])

        ela_signal = self._ela_signal(image)
        signals.append(ela_signal)
        score_parts.append(ela_signal["score"])

        noise_signal = self._noise_signal(image)
        signals.append(noise_signal)
        score_parts.append(noise_signal["score"])

        edge_signal = self._edge_signal(image)
        signals.append(edge_signal)
        score_parts.append(edge_signal["score"])

        score = round(min(1.0, sum(score_parts) / len(score_parts)), 3)
        if score >= 0.65:
            verdict = "likely_tampered"
        elif score >= 0.4:
            verdict = "suspicious"
        else:
            verdict = "no_obvious_tampering"

        return {
            "verdict": verdict,
            "score": score,
            "signals": signals,
            "note": "Heuristic local analysis only; use original files and forensic tools for high-stakes verification.",
        }

    def _metadata_signal(self, image_bytes):
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                exif = image.getexif()
                has_exif = bool(exif)
                software = str(exif.get(0x0131, "")).lower()
        except Exception:
            return self._signal("metadata", 0.25, "metadata could not be read")

        if any(name in software for name in ("photoshop", "gimp", "snapseed", "canva")):
            return self._signal("metadata", 0.7, f"editing software metadata found: {software}")
        if has_exif:
            return self._signal("metadata", 0.05, "metadata is present")
        return self._signal("metadata", 0.25, "metadata is missing or stripped")

    def _ela_signal(self, image):
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return self._signal("error_level", 0.0, "jpeg recompression unavailable")

        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        diff = cv2.absdiff(image, recompressed)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        p95 = float(np.percentile(gray, 95))
        mean = float(np.mean(gray))
        score = min(1.0, max(0.0, (p95 - 12.0) / 32.0))
        return self._signal("error_level", score, f"jpeg error mean={mean:.2f}, p95={p95:.2f}")

    def _noise_signal(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.absdiff(gray, blur)
        block_size = max(24, min(gray.shape[:2]) // 8)

        values = []
        for y in range(0, gray.shape[0] - block_size + 1, block_size):
            for x in range(0, gray.shape[1] - block_size + 1, block_size):
                values.append(float(np.std(noise[y : y + block_size, x : x + block_size])))

        if len(values) < 4:
            return self._signal("noise_consistency", 0.0, "image too small for noise-grid analysis")

        values = np.array(values)
        ratio = float(values.max() / max(values.mean(), 1e-6))
        score = min(1.0, max(0.0, (ratio - 2.0) / 3.0))
        return self._signal("noise_consistency", score, f"local noise variance ratio={ratio:.2f}")

    def _edge_signal(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 180)
        density = float(np.mean(edges > 0))
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(np.percentile(np.abs(lap), 95))
        score = min(1.0, max(0.0, (density * sharpness - 2.5) / 8.0))
        return self._signal("edge_consistency", score, f"edge density={density:.3f}, sharpness95={sharpness:.2f}")

    def _signal(self, name, score, detail):
        return {"name": name, "score": round(float(score), 3), "detail": detail}
