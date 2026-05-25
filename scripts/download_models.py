import os
from pathlib import Path
from shutil import move

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

from ultralytics import YOLO


MODEL_DIR = Path("models")
MODEL_NAME = "yolov8n.pt"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target = MODEL_DIR / MODEL_NAME

    model = YOLO(MODEL_NAME)
    source = Path(model.ckpt_path)
    if source.resolve() != target.resolve():
        move(str(source), target)

    print(f"YOLOv8n ready at {target}")


if __name__ == "__main__":
    main()
