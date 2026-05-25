import os
from ultralytics import YOLO
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

class ObjectDetector:
    def __init__(self, model_path: str = "models/yolov8n.pt", confidence: float = 0.25):
        self.model_path = Path(model_path)
        self.confidence = confidence
        # Lightweight CPU-friendly model. Ultralytics will download it once if
        # the setup script has not already placed it under models/.
        self.model = YOLO(str(self.model_path if self.model_path.exists() else "yolov8n.pt"))

    def detect(self, image):
        results = self.model(image, imgsz=640, conf=self.confidence, verbose=False)[0]

        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = self.model.names[cls_id]
            xyxy = box.xyxy[0].tolist()

            detections.append({
                "label": label,
                "confidence": round(conf, 3),
                "box": [round(float(value), 1) for value in xyxy],
            })

        # sort by confidence (important for captioning)
        detections.sort(key=lambda x: x["confidence"], reverse=True)

        return detections
