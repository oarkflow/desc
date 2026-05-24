import json
import os

from flask import Flask, abort, jsonify, redirect, render_template, request, session, url_for

from kyc.kyc import (
    DOCUMENT_TYPES,
    FaceMatchService,
    KYCRepository,
    LivenessService,
    LocalEvidenceStorage,
    OCRGatewayClient,
    OCRProfileMapper,
    decode_data_url,
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-kyc-secret-change-me")

repo = KYCRepository(os.environ.get("KYC_DB", "kyc.db"))
storage = LocalEvidenceStorage(os.environ.get("EVIDENCE_DIR", "evidence"))
ocr_gateway = OCRGatewayClient()
ocr_mapper = OCRProfileMapper()
face_match_service = FaceMatchService(repo)
liveness_service = LivenessService()


def json_error(message, status=400):
    return jsonify({"error": message}), status


def applicant_session(session_id):
    token = request.headers.get("X-Session-Token") or request.args.get("token") or session.get("applicant_session_token")
    return repo.verify_session_token(session_id, token)


def uploaded_bytes(name="file"):
    if name in request.files:
        upload = request.files[name]
        return upload.read(), upload.filename, upload.content_type
    payload = request.get_json(silent=True) or {}
    data_url = payload.get("data_url") or payload.get("frame")
    if data_url:
        return decode_data_url(data_url), payload.get("filename", "capture.jpg"), payload.get("content_type", "image/jpeg")
    raw = request.get_data()
    if raw:
        return raw, request.headers.get("X-Filename", "upload.bin"), request.content_type
    raise ValueError("No upload provided")


def verification_summary(verification):
    safe = dict(verification)
    safe["document_types"] = DOCUMENT_TYPES
    for collection in ("documents", "selfies", "liveness_checks", "face_matches", "face_embeddings", "face_search_results", "audit_events"):
        safe[collection] = [dict(item) for item in safe[collection]]
    for document in safe["documents"]:
        try:
            document["normalized"] = json.loads(document.get("normalized_json") or "{}")
        except json.JSONDecodeError:
            document["normalized"] = {}
    if safe["session"]:
        safe["session"]["challenge"] = json.loads(safe["session"]["challenge_json"])
    return safe


@app.route("/")
def index():
    session_token = session.get("applicant_session_token")
    if session_token and repo.get_session_by_token(session_token):
        return redirect(url_for("applicant_kyc", session_token=session_token))
    created = repo.create_demo_session()
    session["applicant_session_token"] = created["session_token"]
    return redirect(url_for("applicant_kyc", session_token=created["session_token"]))


@app.route("/kyc/<session_token>")
def applicant_kyc(session_token):
    try:
        kyc_session = repo.get_session_by_token(session_token)
        if not kyc_session:
            abort(404)
        session["applicant_session_token"] = session_token
        return render_template("index.html", document_types=DOCUMENT_TYPES, session_id=kyc_session["id"], session_token=session_token)
    except ValueError:
        abort(404)


@app.route("/api/document-types")
def document_types():
    return jsonify(DOCUMENT_TYPES)


@app.route("/api/kyc/sessions", methods=["POST"])
def create_kyc_session():
    created = repo.create_demo_session()
    session["applicant_session_token"] = created["session_token"]
    applicant_url = url_for("applicant_kyc", session_token=created["session_token"], _external=True)
    return jsonify({"session_id": created["id"], "session_token": created["session_token"], "applicant_url": applicant_url, "verification": verification_summary(repo.get_case(created["id"]))}), 201


@app.route("/api/kyc/demo-session", methods=["POST"])
def create_demo_kyc_session():
    created = repo.create_demo_session()
    return jsonify({"session_id": created["id"], "session_token": created["session_token"], "applicant_url": url_for("applicant_kyc", session_token=created["session_token"], _external=True), "verification": verification_summary(repo.get_case(created["id"]))})


@app.route("/api/kyc/sessions/<session_id>")
def get_kyc_session(session_id):
    try:
        applicant_session(session_id)
        return jsonify(verification_summary(repo.get_case(session_id)))
    except ValueError as error:
        return json_error(str(error), 404)


@app.route("/api/kyc/sessions/<session_id>/profile", methods=["POST"])
def save_profile(session_id):
    try:
        applicant_session(session_id)
        data = request.get_json(silent=True) or request.form.to_dict()
        repo.save_profile(session_id, data)
        return jsonify(verification_summary(repo.get_case(session_id)))
    except ValueError as error:
        return json_error(str(error))


@app.route("/api/kyc/sessions/<session_id>/documents", methods=["POST"])
def upload_document(session_id):
    try:
        applicant_session(session_id)
        data, filename, content_type = uploaded_bytes()
        document_type = request.form.get("document_type") or request.args.get("document_type")
        side = request.form.get("side") or request.args.get("side", "front")
        file_info = storage.save_bytes(session_id, "documents", filename, data)
        try:
            gateway_result = ocr_gateway.extract(
                file_info["path"],
                document_type=document_type,
                content_type=content_type,
                filename=file_info["filename"],
            )
        except ValueError as error:
            gateway_result = {"engine": "http_gateway", "error": str(error), "response": {}}
        suggested_profile = ocr_mapper.map(gateway_result, document_type=document_type)
        repo.add_document(session_id, document_type, side, file_info, content_type, gateway_result)
        face_match_service.enroll_source(session_id, "document", file_info["path"], source_id=side)
        return jsonify({"document": file_info, "gateway": gateway_result, "suggested_profile": suggested_profile, "verification": verification_summary(repo.get_case(session_id))})
    except ValueError as error:
        return json_error(str(error))


@app.route("/api/kyc/sessions/<session_id>/selfie", methods=["POST"])
def upload_selfie(session_id):
    try:
        applicant_session(session_id)
        data, filename, content_type = uploaded_bytes()
        file_info = storage.save_bytes(session_id, "selfies", filename, data)
        repo.add_selfie(session_id, file_info, content_type)
        face_match_service.enroll_source(session_id, "selfie", file_info["path"])
        return jsonify({"selfie": file_info, "verification": verification_summary(repo.get_case(session_id))})
    except ValueError as error:
        return json_error(str(error))


@app.route("/api/kyc/sessions/<session_id>/challenge")
def get_challenge(session_id):
    try:
        applicant_session(session_id)
        verification = repo.get_case(session_id)
        return jsonify({"challenge": json.loads(verification["session"]["challenge_json"])})
    except ValueError as error:
        return json_error(str(error), 404)


@app.route("/api/kyc/sessions/<session_id>/liveness/frame", methods=["POST"])
def process_liveness_frame(session_id):
    try:
        applicant_session(session_id)
        data, _, _ = uploaded_bytes()
        verification = repo.get_case(session_id)
        challenge = json.loads(verification["session"]["challenge_json"])
        result = liveness_service.analyze_frame_bytes(data, session_id=session_id, challenge=challenge)
        repo.audit(session_id, "liveness_frame_processed", result)
        return jsonify(result)
    except ValueError as error:
        return json_error(str(error))


@app.route("/api/kyc/sessions/<session_id>/liveness/complete", methods=["POST"])
def complete_liveness(session_id):
    try:
        applicant_session(session_id)
        verification = repo.get_case(session_id)
        challenge = json.loads(verification["session"]["challenge_json"])
        result = liveness_service.finalize_session(session_id, challenge)
        video_check = repo.latest_liveness_with_file(session_id)
        if video_check:
            try:
                video_result = json.loads(video_check["result_json"])
            except json.JSONDecodeError:
                video_result = {}
            merged = liveness_service.merge_results(video_result, result)
            repo.update_liveness_result(video_check["id"], merged)
            repo.audit(session_id, "liveness_merged", {"liveness_check_id": video_check["id"], "status": merged["risk_status"]})
            result = merged
        else:
            repo.add_liveness(session_id, {}, result)
        return jsonify({"liveness": result, "verification": verification_summary(repo.get_case(session_id))})
    except ValueError as error:
        return json_error(str(error))


@app.route("/api/kyc/sessions/<session_id>/liveness/video", methods=["POST"])
def upload_liveness_video(session_id):
    try:
        applicant_session(session_id)
        data, filename, content_type = uploaded_bytes()
        file_info = storage.save_bytes(session_id, "liveness", filename or "liveness.webm", data)
        verification = repo.get_case(session_id)
        challenge = json.loads(verification["session"]["challenge_json"])
        result = liveness_service.analyze_video_file(file_info["path"], challenge)
        result["content_type"] = content_type
        repo.add_liveness(session_id, file_info, result)
        return jsonify({"liveness": result, "verification": verification_summary(repo.get_case(session_id))})
    except ValueError as error:
        return json_error(str(error))


if __name__ == "__main__":
    print("Starting KYC verification server...")
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5555)))
