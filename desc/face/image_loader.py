"""
Image Loading Utilities
Supports: JPEG, PNG, BMP, TIFF, WEBP, GIF (first frame), HEIC*, RAW*
(*requires optional packages: pillow-heif, rawpy)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import struct


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
    ".webp", ".gif", ".pgm", ".ppm", ".pbm",
    ".heic", ".heif", ".avif",
    ".nef", ".cr2", ".cr3", ".arw", ".orf", ".raf", ".dng",
}


def load_image(path: str, max_dim: int = 2048) -> Optional[np.ndarray]:
    """
    Load an image from any common format.
    Returns a BGR NumPy array, or None on failure.
    Auto-resizes if larger than max_dim to speed up processing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    ext = path.suffix.lower()

    # ── Standard formats via OpenCV ──────────────────────────────────────────
    img = _try_opencv(path)

    # ── HEIC / HEIF ──────────────────────────────────────────────────────────
    if img is None and ext in (".heic", ".heif", ".avif"):
        img = _try_heif(path)

    # ── Camera RAW ───────────────────────────────────────────────────────────
    if img is None and ext in (".nef", ".cr2", ".cr3", ".arw", ".orf", ".raf", ".dng"):
        img = _try_rawpy(path)

    # ── Pillow fallback ───────────────────────────────────────────────────────
    if img is None:
        img = _try_pillow(path)

    if img is None:
        raise ValueError(f"Could not load image: {path} (unsupported or corrupt)")

    # ── Ensure 3-channel BGR ─────────────────────────────────────────────────
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # ── Resize if too large ───────────────────────────────────────────────────
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)

    return img


def _try_opencv(path: Path) -> Optional[np.ndarray]:
    try:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is not None and img.size > 0:
            return img
    except Exception:
        pass

    # Try with imdecode (handles some edge cases)
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if img is not None and img.size > 0:
            return img
    except Exception:
        pass

    return None


def _try_pillow(path: Path) -> Optional[np.ndarray]:
    try:
        from PIL import Image
        pil_img = Image.open(str(path))

        # Handle GIF: take first frame
        if hasattr(pil_img, "n_frames") and pil_img.n_frames > 1:
            pil_img.seek(0)

        # Convert to RGB first (handles palettes, CMYK, etc.)
        pil_img = pil_img.convert("RGB")
        arr = np.array(pil_img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _try_heif(path: Path) -> Optional[np.ndarray]:
    try:
        from PIL import Image
        import pillow_heif
        pillow_heif.register_heif_opener()
        return _try_pillow(path)
    except ImportError:
        pass

    try:
        import pyheif
        heif_file = pyheif.read(str(path))
        from PIL import Image as PILImage
        pil_img = PILImage.frombytes(
            heif_file.mode, heif_file.size, heif_file.data,
            "raw", heif_file.mode, heif_file.stride,
        )
        arr = np.array(pil_img.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _try_rawpy(path: Path) -> Optional[np.ndarray]:
    try:
        import rawpy
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=False,
                no_auto_bright=False,
                output_bps=8,
            )
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def image_info(image: np.ndarray) -> dict:
    """Return basic metadata about a loaded image."""
    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    return {
        "width": w,
        "height": h,
        "channels": channels,
        "dtype": str(image.dtype),
        "size_mp": round(w * h / 1_000_000, 2),
    }
