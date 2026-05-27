import argparse
import io
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from check_common import default_test_image, existing_path, image_mime, print_json, require


class LocalFakeOCRGateway:
    def extract(self, *args, **kwargs):
        return {
            "engine": "script_fake",
            "response": {
                "document_type": "nepali_national_id",
                "values": {"nid_number": "123-456-7890"},
                "meta": {"document_type": "nepali_national_id", "document_type_confidence": 1.0},
                "objects": [
                    {
                        "label": "id_card",
                        "confidence": 0.91,
                        "box": {"pixel": {"x": 1, "y": 2, "width": 100, "height": 60}},
                        "source": "script_fake",
                    }
                ],
                "object_summary": {"has_id_card": True, "id_card_confidence": 0.91, "face_count": 1},
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Check KYC session, document upload, selfie upload, and liveness API flow.")
    parser.add_argument("--document", type=existing_path, default=default_test_image("national-id.webp"))
    parser.add_argument("--selfie", type=existing_path, default=default_test_image("citizenship.jpg"))
    parser.add_argument("--use-real-ocr-gateway", action="store_true")
    args = parser.parse_args()

    import kyc.index as index
    import kyc.ocr.service as service
    from kyc.core.repository import KYCRepository
    from kyc.core.storage import LocalEvidenceStorage

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        index.repo = KYCRepository(str(root / "kyc.db"))
        index.storage = LocalEvidenceStorage(str(root / "evidence"))
        if not args.use_real_ocr_gateway:
            index.ocr_gateway = LocalFakeOCRGateway()

        client = TestClient(service.app)
        created = client.post("/api/kyc/sessions")
        require(created.status_code == 201, f"Session create failed: {created.text[:500]}")
        session = created.json()
        session_id = session["session_id"]
        token = session["session_token"]
        headers = {"X-Session-Token": token}

        profile = client.post(
            f"/api/kyc/sessions/{session_id}/profile",
            headers=headers,
            json={
                "full_name": "Script Check",
                "date_of_birth": "1990-01-01",
                "nationality": "Nepal",
                "address": "Kathmandu",
                "document_type": "national_id",
                "document_number": "123-456-7890",
            },
        )
        require(profile.status_code == 200, f"Profile save failed: {profile.text[:500]}")

        document_bytes = Path(args.document).read_bytes()
        document = client.post(
            f"/api/kyc/sessions/{session_id}/documents",
            headers=headers,
            data={"document_type": "national_id", "side": "front"},
            files={"file": (Path(args.document).name, io.BytesIO(document_bytes), image_mime(args.document))},
        )
        require(document.status_code == 200, f"Document upload failed: {document.text[:500]}")

        selfie_bytes = Path(args.selfie).read_bytes()
        selfie = client.post(
            f"/api/kyc/sessions/{session_id}/selfie",
            headers=headers,
            files={"file": (Path(args.selfie).name, io.BytesIO(selfie_bytes), image_mime(args.selfie))},
        )
        require(selfie.status_code == 200, f"Selfie upload failed: {selfie.text[:500]}")

        liveness = client.post(
            f"/api/kyc/sessions/{session_id}/liveness/frame",
            headers=headers,
            files={"file": (Path(args.selfie).name, io.BytesIO(selfie_bytes), image_mime(args.selfie))},
        )
        require(liveness.status_code == 200, f"Liveness frame failed: {liveness.text[:500]}")

        complete = client.post(f"/api/kyc/sessions/{session_id}/liveness/complete", headers=headers)
        require(complete.status_code == 200, f"Liveness completion failed: {complete.text[:500]}")

        case = client.get(f"/api/kyc/sessions/{session_id}", headers=headers)
        require(case.status_code == 200, f"Case fetch failed: {case.text[:500]}")
        case_payload = case.json()
        require(case_payload.get("documents"), "KYC case did not retain the uploaded document.")
        require(case_payload.get("selfies"), "KYC case did not retain the uploaded selfie.")
        require(case_payload.get("liveness_checks"), "KYC case did not retain the liveness check.")

    print_json(
        {
            "check": "kyc_flow",
            "status": "pass",
            "session_id": session_id,
            "document_status": document.json()["verification"]["documents"][0]["risk_status"],
            "liveness": complete.json()["liveness"],
            "summary": {
                "documents": len(case_payload.get("documents", [])),
                "selfies": len(case_payload.get("selfies", [])),
                "liveness_checks": len(case_payload.get("liveness_checks", [])),
                "face_embeddings": len(case_payload.get("face_embeddings", [])),
            },
        }
    )


if __name__ == "__main__":
    main()
