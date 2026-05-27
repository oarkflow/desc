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


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target = MODEL_DIR / MODEL_NAME

    if not target.exists():
        urlretrieve(MODEL_URL, target)

    print(f"YOLOv8n ready at {target}")

    face_target = MODEL_DIR / FACE_LANDMARKER_NAME
    if not face_target.exists():
        urlretrieve(FACE_LANDMARKER_URL, face_target)
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
        if not target.exists():
            urlretrieve(url, target)
        print(f"OpenCV model ready at {target}")


if __name__ == "__main__":
    main()
