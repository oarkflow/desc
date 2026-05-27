import os
from pathlib import Path
from urllib.request import urlretrieve

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_NAME = "yolov8n.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_NAME = "face_landmarker.task"
LBF_URL = "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"
LBF_NAME = "lbfmodel.yaml"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
YUNET_NAME = "face_detection_yunet_2023mar.onnx"
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx"
)
SFACE_NAME = "face_recognition_sface_2021dec.onnx"
AGE_PROTO_URL = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt"
AGE_PROTO_NAME = "age_deploy.prototxt"
AGE_MODEL_URL = "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel"
AGE_MODEL_NAME = "age_net.caffemodel"
GENDER_PROTO_URL = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt"
GENDER_PROTO_NAME = "gender_deploy.prototxt"
GENDER_MODEL_URL = "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel"
GENDER_MODEL_NAME = "gender_net.caffemodel"
INSIGHTFACE_ROOT = MODEL_DIR / "insightface"
INSIGHTFACE_MODEL_NAME = os.environ.get("INSIGHTFACE_MODEL_NAME", "buffalo_l")
ANTI_SPOOF_MODEL_URL = os.environ.get("ANTI_SPOOF_MODEL_URL")
ANTI_SPOOF_MODEL_NAME = os.environ.get("ANTI_SPOOF_MODEL_NAME", "anti_spoof.onnx")


def download_if_missing(url, target):
    if not target.exists():
        urlretrieve(url, target)
    return target


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target = MODEL_DIR / MODEL_NAME

    download_if_missing(MODEL_URL, target)

    print(f"YOLOv8n ready at {target}")

    face_target = MODEL_DIR / FACE_LANDMARKER_NAME
    download_if_missing(FACE_LANDMARKER_URL, face_target)
    print(f"MediaPipe face landmarker ready at {face_target}")

    for url, name in (
        (LBF_URL, LBF_NAME),
        (YUNET_URL, YUNET_NAME),
        (SFACE_URL, SFACE_NAME),
        (AGE_PROTO_URL, AGE_PROTO_NAME),
        (AGE_MODEL_URL, AGE_MODEL_NAME),
        (GENDER_PROTO_URL, GENDER_PROTO_NAME),
        (GENDER_MODEL_URL, GENDER_MODEL_NAME),
    ):
        target = MODEL_DIR / name
        download_if_missing(url, target)
        print(f"OpenCV model ready at {target}")

    insightface_dir = INSIGHTFACE_ROOT / "models" / INSIGHTFACE_MODEL_NAME
    if insightface_dir.exists():
        print(f"InsightFace model ready at {insightface_dir}")
    elif os.environ.get("INSIGHTFACE_ALLOW_DOWNLOAD", "").strip().lower() in {"1", "true", "yes", "on"}:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name=INSIGHTFACE_MODEL_NAME, root=str(INSIGHTFACE_ROOT), providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print(f"InsightFace model ready at {insightface_dir}")
    else:
        print(
            "InsightFace model not downloaded. Set INSIGHTFACE_ALLOW_DOWNLOAD=true "
            f"to fetch {INSIGHTFACE_MODEL_NAME} into {INSIGHTFACE_ROOT}."
        )

    if ANTI_SPOOF_MODEL_URL:
        anti_spoof_target = MODEL_DIR / ANTI_SPOOF_MODEL_NAME
        download_if_missing(ANTI_SPOOF_MODEL_URL, anti_spoof_target)
        print(f"Anti-spoofing ONNX model ready at {anti_spoof_target}")
    else:
        print("Anti-spoofing model URL not configured. Set ANTI_SPOOF_MODEL_URL to download a vetted ONNX artifact.")


if __name__ == "__main__":
    main()
