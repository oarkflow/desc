from fastapi import HTTPException, UploadFile
try:
    from kyc.describe.detector import ObjectDetector
    from kyc.describe.captioner import CaptionGenerator
    from kyc.describe.ocr import OCRReader
    from kyc.describe.tamper import TamperAnalyzer
except ModuleNotFoundError:
    from describe.detector import ObjectDetector
    from describe.captioner import CaptionGenerator
    from describe.ocr import OCRReader
    from describe.tamper import TamperAnalyzer
import numpy as np
import cv2

detector = ObjectDetector()
captioner = CaptionGenerator()
ocr = OCRReader()
tamper_analyzer = TamperAnalyzer()


def read_image(file_bytes: bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Upload must be a valid image file.")
    return img


async def health():
    return {
        "status": "ok",
        "detector_loaded": detector.model is not None,
        "ocr_available": ocr.available,
        "ocr_languages": ocr.languages,
        "model_path": str(detector.model_path),
    }


async def describe_image(file: UploadFile):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must use an image content type.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Upload cannot be empty.")

    image = read_image(image_bytes)
    height, width = image.shape[:2]

    detections = detector.detect(image)
    text = ocr.read(image)
    tamper = tamper_analyzer.analyze(image, image_bytes)
    caption = captioner.generate(detections, text=text)
    tags = captioner.tags(detections, text=text)

    return {
        "caption": caption,
        "objects": detections,
        "text": text,
        "text_languages": ocr.languages,
        "tags": tags,
        "tamper": tamper,
        "object_count": len(detections),
        "width": width,
        "height": height,
    }
