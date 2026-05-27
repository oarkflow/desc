from pydantic import BaseModel
from typing import List, Optional

class ObjectItem(BaseModel):
    label: str
    confidence: float
    box: List[float]

class ImageResponse(BaseModel):
    caption: str
    objects: List[ObjectItem]
    text: str = ""
    text_languages: List[str]
    tags: List[str]
    tamper: dict
    object_count: int
    width: int
    height: int


class HealthResponse(BaseModel):
    status: str
    detector_loaded: bool
    ocr_available: bool
    ocr_languages: List[str]
    model_path: Optional[str] = None
