import json
import os
from pathlib import Path

import cv2
import numpy as np

from kyc.core.utils import cosine_similarity, normalize_vector
from kyc.quiet import configure_quiet_ml_runtime, suppress_native_output

configure_quiet_ml_runtime()


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_det_size(value):
    if not value:
        return (640, 640)
    parts = [part.strip() for part in value.replace("x", ",").split(",") if part.strip()]
    if len(parts) != 2:
        return (640, 640)
    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError:
        return (640, 640)


def public_face_box(face_box):
    safe = {}
    for key, value in (face_box or {}).items():
        if key.startswith("_"):
            continue
        if isinstance(value, np.ndarray):
            safe[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            safe[key] = value.item()
        else:
            safe[key] = value
    return safe


class FaceRecognitionProvider:
    provider_name = "base"
    model_version = "none"

    def detect_faces(self, image):
        raise NotImplementedError

    def extract_embedding(self, image, face_box):
        raise NotImplementedError

    def compare(self, embedding_a, embedding_b):
        return cosine_similarity(embedding_a, embedding_b)


class LocalONNXFaceRecognitionProvider(FaceRecognitionProvider):
    provider_name = "local_onnx"

    def __init__(self, model_path=None):
        self.model_path = Path(model_path or os.environ.get("FACE_RECOGNITION_MODEL", MODEL_DIR / "arcface.onnx"))
        self.model_version = self.model_path.name
        self.session = None
        cascade_dir = Path(cv2.data.haarcascades)
        self.face_cascade = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_frontalface_default.xml"))
        if self.model_path.exists():
            try:
                import onnxruntime as ort
            except Exception as error:
                self.load_error = f"onnxruntime is required when FACE_RECOGNITION_MODEL is configured: {error}"
            else:
                self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
                self.load_error = None
        else:
            self.load_error = "Face recognition model is not configured."

    @property
    def available(self):
        return self.session is not None

    def detect_faces(self, image):
        if image is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        results = []
        h, w = image.shape[:2]
        for x, y, face_w, face_h in faces:
            area = float(face_w * face_h)
            quality = min(1.0, area / max(float(w * h) * 0.12, 1.0))
            results.append({"x": int(x), "y": int(y), "width": int(face_w), "height": int(face_h), "quality": round(quality, 4)})
        return sorted(results, key=lambda item: item["width"] * item["height"], reverse=True)

    def extract_embedding(self, image, face_box):
        if not self.available:
            return None
        crop = self.crop_face(image, face_box)
        blob = self.preprocess(crop)
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: blob})[0]
        return normalize_vector(output.reshape(-1))

    def crop_face(self, image, face_box):
        x = max(int(face_box["x"]), 0)
        y = max(int(face_box["y"]), 0)
        w = max(int(face_box["width"]), 1)
        h = max(int(face_box["height"]), 1)
        pad = int(max(w, h) * 0.18)
        y1 = max(y - pad, 0)
        y2 = min(y + h + pad, image.shape[0])
        x1 = max(x - pad, 0)
        x2 = min(x + w + pad, image.shape[1])
        return image[y1:y2, x1:x2]

    def preprocess(self, crop):
        resized = cv2.resize(crop, (112, 112))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        normalized = (rgb - 127.5) / 127.5
        return np.transpose(normalized, (2, 0, 1))[None, :, :, :]


class InsightFaceRecognitionProvider(FaceRecognitionProvider):
    provider_name = "insightface"

    def __init__(self, model_name=None, model_root=None, det_size=None):
        self.model_name = model_name or os.environ.get("INSIGHTFACE_MODEL_NAME", "buffalo_l")
        self.model_root = Path(model_root or os.environ.get("INSIGHTFACE_MODEL_ROOT", MODEL_DIR / "insightface"))
        self.det_size = det_size or _parse_det_size(os.environ.get("INSIGHTFACE_DET_SIZE"))
        self.model_version = self.model_name
        self.app = None
        self.load_error = None

        expected_model_dir = self.model_root / "models" / self.model_name
        allow_download = _env_flag("INSIGHTFACE_ALLOW_DOWNLOAD", False)
        if not expected_model_dir.exists() and not allow_download:
            self.load_error = (
                f"InsightFace model artifact is missing at {expected_model_dir}. "
                "Place the model there or set INSIGHTFACE_ALLOW_DOWNLOAD=true during setup."
            )
            return

        try:
            with suppress_native_output():
                from insightface.app import FaceAnalysis

                self.app = FaceAnalysis(
                    name=self.model_name,
                    root=str(self.model_root),
                    providers=["CPUExecutionProvider"],
                )
                self.app.prepare(ctx_id=-1, det_size=self.det_size)
        except ImportError as error:
            self.load_error = f"insightface is required for InsightFace recognition: {error}"
        except Exception as error:
            self.app = None
            self.load_error = f"InsightFace model could not be loaded: {error}"

    @property
    def available(self):
        return self.app is not None

    def detect_faces(self, image):
        if image is None or not self.available:
            return []
        faces = self.app.get(image)
        results = []
        image_h, image_w = image.shape[:2]
        for face in faces:
            x1, y1, x2, y2 = [int(round(value)) for value in face.bbox.tolist()]
            x1 = max(0, min(x1, image_w - 1))
            y1 = max(0, min(y1, image_h - 1))
            x2 = max(x1 + 1, min(x2, image_w))
            y2 = max(y1 + 1, min(y2, image_h))
            width = x2 - x1
            height = y2 - y1
            det_score = float(getattr(face, "det_score", 0.0) or 0.0)
            area_ratio = (width * height) / max(float(image_w * image_h), 1.0)
            face_box = {
                "x": x1,
                "y": y1,
                "width": width,
                "height": height,
                "quality": round(min(1.0, max(det_score, area_ratio * 8)), 4),
                "det_score": round(det_score, 4),
            }
            if getattr(face, "kps", None) is not None:
                face_box["landmarks"] = face.kps.astype(float).round(4).tolist()
            if getattr(face, "embedding", None) is not None:
                face_box["_embedding"] = normalize_vector(face.embedding.reshape(-1))
            results.append(face_box)
        return sorted(results, key=lambda item: item["width"] * item["height"], reverse=True)

    def extract_embedding(self, image, face_box):
        if not self.available:
            return None
        embedded = (face_box or {}).get("_embedding")
        if embedded is not None:
            return normalize_vector(embedded)
        faces = self.detect_faces(image)
        if not faces:
            return None
        return faces[0].get("_embedding")


def default_face_provider():
    configured = os.environ.get("FACE_RECOGNITION_PROVIDER", "auto").strip().lower()
    if configured in {"insightface", "auto"}:
        insightface_provider = InsightFaceRecognitionProvider()
        if configured == "insightface" or insightface_provider.available:
            return insightface_provider
    return LocalONNXFaceRecognitionProvider()


class FaceMatchService:
    def __init__(self, repository=None, provider=None):
        self.repository = repository
        self.provider = provider or default_face_provider()

    def score(self, document_path, selfie_path):
        doc_result = self.extract_embedding_from_file(document_path, "document")
        selfie_result = self.extract_embedding_from_file(selfie_path, "selfie")
        if not doc_result.get("embedding") or not selfie_result.get("embedding"):
            return {
                "score": None,
                "status": "needs_manual_review",
                "reason": doc_result.get("reason") or selfie_result.get("reason") or "Could not extract comparable face embeddings.",
                "provider": self.provider.provider_name,
                "model_version": self.provider.model_version,
            }

        return self.compare_embeddings(doc_result["embedding"], selfie_result["embedding"])

    def enroll_source(self, session_id, source_type, image_path, source_id=None):
        result = self.extract_embedding_from_file(image_path, source_type)
        if not self.repository or not result.get("embedding"):
            return result
        embedding_id = self.repository.add_face_embedding(
            session_id,
            source_type,
            result["embedding"],
            result["provider"],
            result["model_version"],
            quality_score=result.get("quality_score"),
            face_box=result.get("face_box"),
            source_id=source_id,
        )
        return {**result, "embedding_id": embedding_id}

    def compare_session(self, session_id):
        if not self.repository:
            return {"score": None, "status": "needs_manual_review", "reason": "Face repository is not configured."}
        document = self.repository.latest_face_embedding(session_id, ["document"])
        selfie = self.repository.latest_face_embedding(session_id, ["selfie", "liveness"])
        if not document or not selfie:
            return {
                "score": None,
                "status": "needs_manual_review",
                "reason": "Missing document or selfie face embedding.",
                "provider": self.provider.provider_name,
                "model_version": self.provider.model_version,
            }
        result = self.compare_embeddings(json.loads(document["vector_json"]), json.loads(selfie["vector_json"]))
        result["document_embedding_id"] = document["id"]
        result["selfie_embedding_id"] = selfie["id"]
        return result

    def search_tenant_gallery(self, session_id, limit=5):
        if not self.repository:
            return []
        session = self.repository.get_session(session_id)
        query = self.repository.latest_face_embedding(session_id, ["selfie", "liveness", "document"])
        if not session or not query:
            return []
        query_vector = json.loads(query["vector_json"])
        candidates = self.repository.list_face_embeddings(
            session["tenant_id"],
            exclude_session_id=session_id,
            source_types=["selfie", "liveness", "document"],
        )
        matches = []
        for candidate in candidates:
            score = cosine_similarity(query_vector, json.loads(candidate["vector_json"]))
            if score >= 0.45:
                matches.append(
                    {
                        "embedding_id": candidate["id"],
                        "session_id": candidate["session_id"],
                        "score": round(score, 4),
                        "source_type": candidate["source_type"],
                    }
                )
        matches = sorted(matches, key=lambda item: item["score"], reverse=True)[:limit]
        self.repository.store_face_search_results(session_id, query["id"], matches)
        return matches

    def compare_embeddings(self, embedding_a, embedding_b):
        score = round(self.provider.compare(embedding_a, embedding_b), 4)
        if score >= 0.55:
            status = "pass"
        elif score >= 0.45:
            status = "needs_manual_review"
        else:
            status = "warn"
        return {
            "score": score,
            "status": status,
            "provider": self.provider.provider_name,
            "model_version": self.provider.model_version,
            "thresholds": {"pass": 0.55, "review": 0.45},
        }

    def extract_embedding_from_file(self, image_path, source_type):
        image = cv2.imread(str(image_path))
        if image is None:
            return {"status": "needs_manual_review", "reason": "Image could not be read.", "source_type": source_type}
        faces = self.provider.detect_faces(image)
        if not faces:
            reason = "No face detected."
            if not getattr(self.provider, "available", True):
                reason = getattr(self.provider, "load_error", None) or reason
            return {"status": "needs_manual_review", "reason": reason, "source_type": source_type}
        if not getattr(self.provider, "available", True):
            return {
                "status": "needs_manual_review",
                "reason": getattr(self.provider, "load_error", None) or "Face recognition model is not configured.",
                "source_type": source_type,
                "provider": self.provider.provider_name,
                "model_version": self.provider.model_version,
                "face_box": public_face_box(faces[0]),
                "quality_score": faces[0].get("quality"),
            }
        embedding = self.provider.extract_embedding(image, faces[0])
        if embedding is None:
            return {"status": "needs_manual_review", "reason": "Embedding extraction failed.", "source_type": source_type}
        return {
            "status": "active",
            "source_type": source_type,
            "embedding": embedding,
            "provider": self.provider.provider_name,
            "model_version": self.provider.model_version,
            "face_box": public_face_box(faces[0]),
            "quality_score": faces[0].get("quality"),
        }
