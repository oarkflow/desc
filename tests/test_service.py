import asyncio
from io import BytesIO

import pytest
from fastapi import HTTPException
from PIL import Image, ImageDraw

from app.captioner import CaptionGenerator
from app.main import describe_image
from app.ocr import OCRReader


def make_image(fmt):
    image = Image.new("RGB", (320, 220), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((36, 42, 132, 166), fill="red")
    draw.ellipse((176, 58, 272, 154), fill="blue")
    draw.text((42, 178), "SALE 50", fill="black")
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    buffer.seek(0)
    return buffer


class UploadStub:
    def __init__(self, content, content_type):
        self._content = content
        self.content_type = content_type

    async def read(self):
        return self._content


def test_caption_rules_generate_action_and_tags():
    captioner = CaptionGenerator()
    detections = [
        {"label": "dog", "confidence": 0.91, "box": [0, 0, 10, 10]},
        {"label": "sports ball", "confidence": 0.88, "box": [20, 20, 30, 30]},
    ]

    assert captioner.generate(detections) == "A dog playing with a ball in an outdoor scene."
    assert captioner.tags(detections) == ["dog", "sports ball", "outdoor"]


def test_ocr_filter_accepts_english_and_nepali_text():
    reader = OCRReader()

    assert reader._keep_word("SALE", "88")
    assert reader._keep_word("नेपाली", "88")
    assert not reader._keep_word("j", "88")
    assert not reader._keep_word("SALE", "12")


def test_rejects_non_image_upload():
    upload = UploadStub(b"not an image", "text/plain")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(describe_image(upload))

    assert exc.value.status_code == 400


def test_describe_accepts_common_image_extensions(monkeypatch):
    def fake_detect(image):
        return [{"label": "person", "confidence": 0.9, "box": [10.0, 20.0, 80.0, 180.0]}]

    monkeypatch.setattr("app.main.detector.detect", fake_detect)
    monkeypatch.setattr("app.main.ocr.read", lambda image: "")
    monkeypatch.setattr(
        "app.main.tamper_analyzer.analyze",
        lambda image, image_bytes: {"verdict": "no_obvious_tampering", "score": 0.1, "signals": [], "note": ""},
    )

    cases = [
        ("sample.jpg", "JPEG", "image/jpeg"),
        ("sample.png", "PNG", "image/png"),
        ("sample.webp", "WEBP", "image/webp"),
        ("sample.bmp", "BMP", "image/bmp"),
    ]

    for _, fmt, content_type in cases:
        upload = UploadStub(make_image(fmt).read(), content_type)
        body = asyncio.run(describe_image(upload))

        assert body["caption"] == "An image showing a person."
        assert body["objects"][0]["label"] == "person"
        assert body["tags"] == ["person"]
        assert body["text_languages"]
        assert body["tamper"]["verdict"] == "no_obvious_tampering"
        assert body["width"] == 320
        assert body["height"] == 220
