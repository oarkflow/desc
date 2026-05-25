"""
face — Face Recognition & Landmark Extraction Platform.
"""
from .engine import FacePlatform, AnalysisResult
from .detector import FaceDetection, HaarCascadeDetector, MultiScaleDetector
from .landmarks import LandmarkResult, LANDMARK_GROUPS
from .recognizer import FusionRecognizer, RecognitionResult
from .image_loader import load_image, image_info
from .visualizer import draw_faces, save_image

__version__ = "1.0.0"
__all__ = [
    "FacePlatform", "AnalysisResult",
    "FaceDetection", "LandmarkResult", "RecognitionResult",
    "load_image", "draw_faces",
]
