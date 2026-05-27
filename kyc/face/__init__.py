"""
face — Face Recognition & Landmark Extraction Platform.
"""
from .engine import FacePlatform, AnalysisResult
from .detector import FaceDetection, HaarCascadeDetector, MultiScaleDetector
from .landmarks import LandmarkResult, LANDMARK_GROUPS, MEDIAPIPE_LANDMARK_COUNT
from .recognizer import FusionRecognizer, RecognitionResult
from .attributes import DemographicEstimator, FaceAttributes
from .image_loader import load_image, image_info
from .visualizer import draw_faces, save_image

__version__ = "1.0.0"
__all__ = [
    "FacePlatform", "AnalysisResult",
    "FaceDetection", "LandmarkResult", "RecognitionResult", "FaceAttributes", "DemographicEstimator",
    "MEDIAPIPE_LANDMARK_COUNT",
    "load_image", "draw_faces",
]
