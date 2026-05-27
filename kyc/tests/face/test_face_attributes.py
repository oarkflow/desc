import numpy as np

from kyc.face.attributes import _age_result, _gender_result, estimate_face_attributes
from kyc.face.detector import FaceDetection


def test_face_attributes_include_demographic_model_results():
    image = np.full((120, 120, 3), 128, dtype=np.uint8)
    detection = FaceDetection(bbox=(20, 20, 60, 70), confidence=0.9, index=0)
    demographics = {
        "estimated_age": {"band": "25-32", "confidence": 0.8},
        "gender": {"label": "female_presenting", "confidence": 0.9},
    }

    attributes = estimate_face_attributes(image, detection, demographics=demographics).to_dict()

    assert attributes["estimated_age"]["band"] == "25-32"
    assert attributes["gender"]["label"] == "female_presenting"
    assert attributes["quality"]["label"] in {"poor", "fair", "good"}


def test_age_calibration_threshold_returns_unknown_when_confidence_is_low():
    result = _age_result(
        np.array([0.05, 0.1, 0.1, 0.2, 0.34, 0.1, 0.06, 0.05]),
        {"min_confidence": 0.35, "thresholds": {"25-32": 0.8}},
    )

    assert result["band"] == "unknown"
    assert result["raw_bucket"] == "25-32"


def test_gender_calibration_weights_can_change_prediction():
    result = _gender_result(
        np.array([0.55, 0.45]),
        {"min_confidence": 0.5, "weights": {"female_presenting": 2.0}},
    )

    assert result["label"] == "female_presenting"
