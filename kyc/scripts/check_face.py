import argparse
import tempfile
from pathlib import Path

from check_common import KYC_ROOT, default_test_image, existing_path, print_json, require


MODEL_DIR = KYC_ROOT / "models"


def main():
    parser = argparse.ArgumentParser(description="Check face detection, landmarks, demographics, and recognition.")
    parser.add_argument("--image", type=existing_path, default=default_test_image("citizenship.jpg"))
    parser.add_argument("--detection-mode", default="auto", choices=["auto", "yunet", "multiscale", "haar"])
    parser.add_argument("--landmark-mode", default="auto", choices=["auto", "mediapipe", "lbf", "region"])
    parser.add_argument("--min-faces", type=int, default=1)
    parser.add_argument("--require-landmarks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-recognition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-demographics", action="store_true")
    args = parser.parse_args()

    from kyc.face import FacePlatform

    yunet = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
    lbf = MODEL_DIR / "lbfmodel.yaml"
    mediapipe = MODEL_DIR / "face_landmarker.task"
    age_model = MODEL_DIR / "age_net.caffemodel"
    age_proto = MODEL_DIR / "age_deploy.prototxt"
    gender_model = MODEL_DIR / "gender_net.caffemodel"
    gender_proto = MODEL_DIR / "gender_deploy.prototxt"
    calibration = MODEL_DIR / "demographic_calibration.json"

    detection_mode = args.detection_mode
    if detection_mode == "auto":
        detection_mode = "yunet" if yunet.exists() else "multiscale"

    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "face_db")
        platform = FacePlatform(
            lbf_model_path=str(lbf) if lbf.exists() else None,
            mediapipe_model_path=str(mediapipe) if mediapipe.exists() else None,
            yunet_model_path=str(yunet) if yunet.exists() else None,
            recognizer_db_path=db,
            detection_mode=detection_mode,
            landmark_mode=args.landmark_mode,
            recognition_enabled=True,
            demographic_enabled=True,
            age_model_path=str(age_model) if age_model.exists() else None,
            age_proto_path=str(age_proto) if age_proto.exists() else None,
            gender_model_path=str(gender_model) if gender_model.exists() else None,
            gender_proto_path=str(gender_proto) if gender_proto.exists() else None,
            demographic_calibration_path=str(calibration) if calibration.exists() else None,
        )
        if args.require_recognition:
            platform.enroll_from_image("fixture", str(args.image))
        result = platform.analyze(str(args.image), return_annotated=False)

    payload = result.to_dict()
    require(result.num_faces >= args.min_faces, f"Face analysis found {result.num_faces} faces, expected at least {args.min_faces}.")
    if args.require_landmarks:
        require(any(face.get("landmarks") for face in payload["faces"]), "No landmarks were returned.")
    if args.require_recognition:
        require(any(face.get("recognition") and not face["recognition"].get("is_unknown") for face in payload["faces"]), "Recognition did not identify the enrolled fixture face.")
    if args.require_demographics:
        require(any(face.get("attributes", {}).get("demographics") for face in payload["faces"]), "Demographic model output was not returned.")

    print_json(
        {
            "check": "face",
            "status": "pass",
            "image": str(args.image),
            "detection_mode": detection_mode,
            "landmark_mode": args.landmark_mode,
            "num_faces": result.num_faces,
            "faces": payload["faces"],
        }
    )


if __name__ == "__main__":
    main()
