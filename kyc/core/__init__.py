try:  # pragma: no cover - supports repository-root imports
    from kyc.core.registry import DocumentConfigError, DocumentRegistry, DOCUMENT_REGISTRY, DOCUMENT_TYPES
    from kyc.core.repository import KYCRepository
    from kyc.core.storage import LocalEvidenceStorage
    from kyc.core.ocr import OCRGatewayClient, OCRProfileMapper
    from kyc.core.face import FaceRecognitionProvider, InsightFaceRecognitionProvider, LocalONNXFaceRecognitionProvider, FaceMatchService
    from kyc.core.liveness import AntiSpoofingProvider, LivenessService
    from kyc.core.utils import now_iso, row_to_dict, hash_secret, normalize_vector, cosine_similarity, decode_data_url
    from kyc.core.constants import CHALLENGE_ACTIONS, DEMO_PROFILE
except ModuleNotFoundError:  # pragma: no cover - supports `cd kyc && python ocr_service.py`
    from core.registry import DocumentConfigError, DocumentRegistry, DOCUMENT_REGISTRY, DOCUMENT_TYPES
    from core.repository import KYCRepository
    from core.storage import LocalEvidenceStorage
    from core.ocr import OCRGatewayClient, OCRProfileMapper
    from core.face import FaceRecognitionProvider, InsightFaceRecognitionProvider, LocalONNXFaceRecognitionProvider, FaceMatchService
    from core.liveness import AntiSpoofingProvider, LivenessService
    from core.utils import now_iso, row_to_dict, hash_secret, normalize_vector, cosine_similarity, decode_data_url
    from core.constants import CHALLENGE_ACTIONS, DEMO_PROFILE

__all__ = [
    "DocumentConfigError", "DocumentRegistry", "DOCUMENT_REGISTRY", "DOCUMENT_TYPES",
    "KYCRepository", "LocalEvidenceStorage", "OCRGatewayClient", "OCRProfileMapper",
    "FaceRecognitionProvider", "InsightFaceRecognitionProvider", "LocalONNXFaceRecognitionProvider",
    "FaceMatchService", "AntiSpoofingProvider", "LivenessService",
    "now_iso", "row_to_dict", "hash_secret", "normalize_vector", "cosine_similarity", "decode_data_url",
    "CHALLENGE_ACTIONS", "DEMO_PROFILE",
]
