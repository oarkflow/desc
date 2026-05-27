import os
import shutil
import tempfile
import zipfile
from hashlib import sha256
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
INSIGHTFACE_MODEL_URLS = {
    "buffalo_l": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
}
INSIGHTFACE_MODEL_FILES = {
    "buffalo_l": {
        "1k3d68.onnx",
        "2d106det.onnx",
        "det_10g.onnx",
        "genderage.onnx",
        "w600k_r50.onnx",
    },
}
ANTI_SPOOF_MODEL_URL = os.environ.get(
    "ANTI_SPOOF_MODEL_URL",
    "https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx/resolve/main/minifasnet_v2.onnx",
)
ANTI_SPOOF_MODEL_NAME = os.environ.get("ANTI_SPOOF_MODEL_NAME", "anti_spoof.onnx")
ANTI_SPOOF_MODEL_SHA256 = os.environ.get(
    "ANTI_SPOOF_MODEL_SHA256",
    "d7b3cd9ba8a7ceb13baa8c4720902e27ca3112eff52f926c08804af6b6eecc7b",
)


def file_sha256(path):
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path, expected):
    if expected and file_sha256(path) != expected:
        raise RuntimeError(f"Checksum mismatch for {path}")


def download_if_missing(url, target, expected_sha256=None):
    if target.exists():
        verify_sha256(target, expected_sha256)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(url, target)
        verify_sha256(target, expected_sha256)
    return target


def insightface_model_ready(model_dir, model_name):
    expected_files = INSIGHTFACE_MODEL_FILES.get(model_name)
    if not expected_files:
        return model_dir.exists()
    return all((model_dir / name).exists() for name in expected_files)


def download_insightface_model(model_name):
    insightface_dir = INSIGHTFACE_ROOT / "models" / model_name
    if insightface_model_ready(insightface_dir, model_name):
        return insightface_dir

    url = os.environ.get("INSIGHTFACE_MODEL_URL") or INSIGHTFACE_MODEL_URLS.get(model_name)
    if not url:
        raise RuntimeError(
            f"No InsightFace download URL configured for {model_name!r}. "
            "Set INSIGHTFACE_MODEL_URL to a zip file containing the model ONNX files."
        )

    insightface_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="insightface-", dir="/tmp") as tmp:
        archive = Path(tmp) / f"{model_name}.zip"
        urlretrieve(url, archive)
        extract_dir = Path(tmp) / "extract"
        extract_dir.mkdir()
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(extract_dir)

        source_dir = extract_dir / model_name if (extract_dir / model_name).is_dir() else extract_dir
        for item in source_dir.iterdir():
            target = insightface_dir / item.name
            if item.is_file():
                shutil.copy2(item, target)
            elif item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)

    if not insightface_model_ready(insightface_dir, model_name):
        raise RuntimeError(f"Downloaded InsightFace model is incomplete: {insightface_dir}")
    return insightface_dir


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for label, url, name, checksum in (
        ("YOLOv8n", MODEL_URL, MODEL_NAME, None),
        ("MediaPipe face landmarker", FACE_LANDMARKER_URL, FACE_LANDMARKER_NAME, None),
        ("OpenCV model", LBF_URL, LBF_NAME, None),
        ("OpenCV model", YUNET_URL, YUNET_NAME, None),
        ("OpenCV model", SFACE_URL, SFACE_NAME, None),
        ("OpenCV model", AGE_PROTO_URL, AGE_PROTO_NAME, None),
        ("OpenCV model", AGE_MODEL_URL, AGE_MODEL_NAME, None),
        ("OpenCV model", GENDER_PROTO_URL, GENDER_PROTO_NAME, None),
        ("OpenCV model", GENDER_MODEL_URL, GENDER_MODEL_NAME, None),
        ("Anti-spoofing ONNX model", ANTI_SPOOF_MODEL_URL, ANTI_SPOOF_MODEL_NAME, ANTI_SPOOF_MODEL_SHA256),
    ):
        target = MODEL_DIR / name
        download_if_missing(url, target, checksum)
        print(f"{label} ready at {target}")

    insightface_dir = download_insightface_model(INSIGHTFACE_MODEL_NAME)
    print(f"InsightFace model ready at {insightface_dir}")


if __name__ == "__main__":
    main()
