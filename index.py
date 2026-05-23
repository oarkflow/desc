import json
import os
from functools import wraps
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, session, url_for

from kyc import (
    DOCUMENT_TYPES,
    FaceMatchService,
    KYCRepository,
    LivenessService,
    LocalEvidenceStorage,
    OCRService,
    decode_data_url,
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-kyc-secret-change-me")

repo = KYCRepository(os.environ.get("KYC_DB", "kyc.db"))
storage = LocalEvidenceStorage(os.environ.get("EVIDENCE_DIR", "evidence"))
ocr_service = OCRService()
face_match_service = FaceMatchService()
liveness_service = LivenessService()


def json_error(message, status=400):
    return jsonify({"error": message}), status


def require_admin(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("admin_authenticated"):
            return redirect(url_for("admin_login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


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


def case_summary(case):
    safe = dict(case)
    safe["document_types"] = DOCUMENT_TYPES
    for collection in ("documents", "selfies", "liveness_checks", "face_matches", "audit_events", "decisions"):
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
    return render_template("index.html", document_types=DOCUMENT_TYPES)


@app.route("/api/document-types")
def document_types():
    return jsonify(DOCUMENT_TYPES)


@app.route("/api/kyc/sessions", methods=["POST"])
def create_kyc_session():
    return jsonify(case_summary(repo.get_case(repo.create_session()["id"])))


@app.route("/api/kyc/demo-session", methods=["POST"])
def create_demo_kyc_session():
    return jsonify(case_summary(repo.get_case(repo.create_demo_session()["id"])))


@app.route("/api/kyc/sessions/<session_id>")
def get_kyc_session(session_id):
    try:
        return jsonify(case_summary(repo.get_case(session_id)))
    except ValueError as error:
        return json_error(str(error), 404)


@app.route("/api/kyc/sessions/<session_id>/profile", methods=["POST"])
def save_profile(session_id):
    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        repo.save_profile(session_id, data)
        return jsonify(case_summary(repo.get_case(session_id)))
    except ValueError as error:
        return json_error(str(error))


@app.route("/api/kyc/sessions/<session_id>/documents", methods=["POST"])
def upload_document(session_id):
    try:
        data, filename, content_type = uploaded_bytes()
        document_type = request.form.get("document_type") or request.args.get("document_type")
        side = request.form.get("side") or request.args.get("side", "front")
        file_info = storage.save_bytes(session_id, "documents", filename, data)
        ocr_result = ocr_service.extract(file_info["path"])
        repo.add_document(session_id, document_type, side, file_info, content_type, ocr_result)
        return jsonify({"document": file_info, "ocr": ocr_result, "case": case_summary(repo.get_case(session_id))})
    except ValueError as error:
        return json_error(str(error))


@app.route("/api/kyc/sessions/<session_id>/selfie", methods=["POST"])
def upload_selfie(session_id):
    try:
        data, filename, content_type = uploaded_bytes()
        file_info = storage.save_bytes(session_id, "selfies", filename, data)
        repo.add_selfie(session_id, file_info, content_type)
        return jsonify({"selfie": file_info, "case": case_summary(repo.get_case(session_id))})
    except ValueError as error:
        return json_error(str(error))


@app.route("/api/kyc/sessions/<session_id>/challenge")
def get_challenge(session_id):
    try:
        case = repo.get_case(session_id)
        return jsonify({"challenge": json.loads(case["session"]["challenge_json"])})
    except ValueError as error:
        return json_error(str(error), 404)


@app.route("/api/kyc/sessions/<session_id>/liveness/frame", methods=["POST"])
def process_liveness_frame(session_id):
    try:
        data, _, _ = uploaded_bytes()
        case = repo.get_case(session_id)
        challenge = json.loads(case["session"]["challenge_json"])
        result = liveness_service.analyze_frame_bytes(data, session_id=session_id, challenge=challenge)
        repo.audit(session_id, "liveness_frame_processed", result)
        return jsonify(result)
    except ValueError as error:
        return json_error(str(error))


@app.route("/api/kyc/sessions/<session_id>/liveness/complete", methods=["POST"])
def complete_liveness(session_id):
    try:
        case = repo.get_case(session_id)
        challenge = json.loads(case["session"]["challenge_json"])
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
        return jsonify({"liveness": result, "case": case_summary(repo.get_case(session_id))})
    except ValueError as error:
        return json_error(str(error))


@app.route("/api/kyc/sessions/<session_id>/liveness/video", methods=["POST"])
def upload_liveness_video(session_id):
    try:
        data, filename, content_type = uploaded_bytes()
        file_info = storage.save_bytes(session_id, "liveness", filename or "liveness.webm", data)
        case = repo.get_case(session_id)
        challenge = json.loads(case["session"]["challenge_json"])
        result = liveness_service.analyze_video_file(file_info["path"], challenge)
        result["content_type"] = content_type
        repo.add_liveness(session_id, file_info, result)
        return jsonify({"liveness": result, "case": case_summary(repo.get_case(session_id))})
    except ValueError as error:
        return json_error(str(error))


@app.route("/api/kyc/sessions/<session_id>/submit", methods=["POST"])
def submit_kyc_session(session_id):
    try:
        case = repo.get_case(session_id)
        front_docs = [doc for doc in case["documents"] if doc["side"] == "front"] or case["documents"]
        if front_docs and case["selfies"]:
            match = face_match_service.score(front_docs[-1]["file_path"], case["selfies"][-1]["file_path"])
            repo.add_face_match(session_id, match)
        submitted = repo.submit(session_id)
        return jsonify(case_summary(submitted))
    except ValueError as error:
        return json_error(str(error))


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    error = None
    if request.method == "POST":
        password = request.form.get("password", "")
        expected = os.environ.get("ADMIN_PASSWORD", "admin")
        if password == expected:
            session["admin_authenticated"] = True
            return redirect(request.args.get("next") or url_for("admin_cases"))
        error = "Invalid admin password"
    return render_template("admin_login.html", error=error)


@app.route("/admin/logout", methods=["POST"])
@require_admin
def admin_logout():
    session.clear()
    return redirect(url_for("admin_login"))


@app.route("/admin/cases")
@require_admin
def admin_cases():
    return render_template("admin_cases.html", cases=repo.list_cases(), document_types=DOCUMENT_TYPES)


@app.route("/admin/cases/<session_id>")
@require_admin
def admin_case(session_id):
    try:
        return render_template("admin_case.html", case=case_summary(repo.get_case(session_id)), document_types=DOCUMENT_TYPES)
    except ValueError:
        abort(404)


@app.route("/admin/cases/<session_id>/decision", methods=["POST"])
@require_admin
def admin_decision(session_id):
    try:
        repo.decide(session_id, request.form.get("decision", ""), request.form.get("note", ""))
        return redirect(url_for("admin_case", session_id=session_id))
    except ValueError as error:
        return json_error(str(error))


@app.route("/admin/cases/<session_id>/delete-evidence", methods=["POST"])
@require_admin
def admin_delete_evidence(session_id):
    storage.delete_session(session_id)
    repo.delete_evidence(session_id)
    return redirect(url_for("admin_case", session_id=session_id))


@app.route("/admin/evidence/<kind>/<int:item_id>")
@require_admin
def admin_evidence(kind, item_id):
    table_map = {"documents": "documents", "selfies": "selfies", "liveness": "liveness_checks"}
    table = table_map.get(kind)
    if not table:
        abort(404)
    with repo.connect() as conn:
        if kind == "liveness":
            row = conn.execute("SELECT file_path, result_json FROM liveness_checks WHERE id = ?", (item_id,)).fetchone()
            content_type = "video/webm"
            if row:
                try:
                    content_type = json.loads(row["result_json"]).get("content_type") or content_type
                except json.JSONDecodeError:
                    pass
        else:
            row = conn.execute(f"SELECT file_path, content_type FROM {table} WHERE id = ?", (item_id,)).fetchone()
            content_type = row["content_type"] if row else None
    if not row or not row["file_path"] or not Path(row["file_path"]).exists():
        abort(404)
    return send_file(row["file_path"], mimetype=content_type or "application/octet-stream")


if __name__ == "__main__":
    print("Starting KYC verification server...")
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5555)))
