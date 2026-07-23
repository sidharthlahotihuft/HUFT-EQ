"""
HUFT CS Call Quality Portal
----------------------------
Upload a call recording -> Whisper transcribes it -> AI (or heuristic fallback)
scores it against the 100-point Voice rubric -> per-call report card + an
overview dashboard across all scored calls.

Deploy target: Vercel (Python/Flask runtime) + Supabase (Postgres + Storage).

Architecture notes (why it's shaped this way):
  - Vercel Functions cap request bodies at 4.5MB, so audio never passes through
    our backend on upload. The browser uploads directly to Supabase Storage
    using a short-lived signed upload URL that /api/upload-url hands out.
  - Vercel Functions are stateless (no background threads survive past the
    response), so transcription and scoring are two separate, short,
    idempotent API calls that the browser calls in sequence
    (see the script in templates/report_card.html). If a call is refreshed
    mid-pipeline, it just resumes from whatever `status` says.

Local dev:
    pip install -r requirements.txt
    cp .env.example .env   # fill in Supabase + API keys
    python app.py
Then open http://localhost:5001
"""
import os
import tempfile

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, url_for

import audio_storage
import scoring
import storage
import transcribe

load_dotenv()

app = Flask(__name__, static_folder="public/static", static_url_path="/static")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")

ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "mp4", "webm", "ogg", "flac", "aac"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    calls = storage.list_calls()
    return render_template(
        "index.html", calls=calls, params=scoring.PARAMETERS,
        supabase_url=SUPABASE_URL, supabase_anon_key=SUPABASE_ANON_KEY,
    )


@app.route("/api/upload-url", methods=["POST"])
def api_upload_url():
    """Step 1: hand the browser a short-lived Supabase Storage upload token."""
    data = request.get_json(silent=True) or {}
    filename = (data.get("filename") or "").strip()
    if not filename or not allowed_file(filename):
        return jsonify({"error": f"Unsupported or missing file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400
    try:
        signed = audio_storage.create_signed_upload(filename)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Could not create upload URL: {e}"}), 500
    return jsonify(signed)


@app.route("/api/calls", methods=["GET", "POST"])
def api_calls():
    if request.method == "GET":
        return jsonify(storage.list_calls())

    # POST: Step 2, after the browser has already uploaded the file straight to
    # Supabase Storage. This body is tiny (just metadata), well under any limit.
    data = request.get_json(silent=True) or {}
    storage_path = (data.get("storage_path") or "").strip()
    original_name = (data.get("original_name") or "").strip()
    if not storage_path or not original_name:
        return jsonify({"error": "Missing storage_path or original_name."}), 400

    agent_name = (data.get("agent_name") or "").strip() or "Unassigned"
    call_date = (data.get("call_date") or "").strip()
    call_topic = (data.get("call_topic") or "").strip()

    call_id = storage.create_call(storage_path, original_name, agent_name, call_date, call_topic)
    return jsonify({"call_id": call_id, "redirect": url_for("report_card", call_id=call_id)})


@app.route("/api/status/<int:call_id>")
def api_status(call_id):
    call = storage.get_call(call_id)
    if not call:
        return jsonify({"error": "not found"}), 404
    return jsonify({
        "status": call["status"],
        "status_message": call["status_message"],
        "total_score": call["total_score"],
        "grade": call["grade"],
    })


@app.route("/api/process/transcribe/<int:call_id>", methods=["POST"])
def api_process_transcribe(call_id):
    """Short, idempotent step: download audio from Supabase Storage, run Whisper, save transcript."""
    call = storage.get_call(call_id)
    if not call:
        return jsonify({"error": "not found"}), 404
    try:
        storage.update_status(call_id, "transcribing", "Transcribing audio with Whisper...")
        ext = call["filename"].rsplit(".", 1)[-1] if "." in call["filename"] else "audio"
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp_path = tmp.name
        audio_storage.download_to_path(call["filename"], tmp_path)
        try:
            transcript = transcribe.transcribe(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        storage.save_transcript(call_id, transcript)
        storage.update_status(call_id, "transcribed", "Transcript ready - scoring next...")
        return jsonify({"status": "transcribed"})
    except transcribe.TranscriptionError as e:
        storage.update_status(call_id, "error", f"Transcription failed: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:  # noqa: BLE001
        storage.update_status(call_id, "error", f"Unexpected error during transcription: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/process/score/<int:call_id>", methods=["POST"])
def api_process_score(call_id):
    """Short, idempotent step: score the (already-transcribed) call."""
    call = storage.get_call(call_id)
    if not call:
        return jsonify({"error": "not found"}), 404
    if not call.get("transcript"):
        return jsonify({"error": "No transcript yet - call transcribe first."}), 400
    try:
        storage.update_status(call_id, "scoring", "Scoring call against the Voice rubric...")
        meta = {
            "agent_name": call.get("agent_name"),
            "call_date": call.get("call_date"),
            "call_topic": call.get("call_topic"),
        }
        scores, method = scoring.score_transcript(call["transcript"], meta)
        total = scoring.totals_from_scores(scores)
        grade, _pct = scoring.grade_for(total)
        storage.save_scores(call_id, scores, total, scoring.MAX_TOTAL, grade, method)
        storage.update_status(call_id, "done", "Scoring complete.")
        return jsonify({"status": "done", "total_score": total, "grade": grade})
    except Exception as e:  # noqa: BLE001
        storage.update_status(call_id, "error", f"Unexpected error during scoring: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/call/<int:call_id>")
def report_card(call_id):
    call = storage.get_call(call_id)
    if not call:
        return "Call not found", 404

    scores = call.get("scores") or {}
    rows = []
    for p in scoring.PARAMETERS:
        v = scores.get(p["key"], {"score": p["max"], "evidence": "Not yet scored.", "max": p["max"]})
        mx = v.get("max", p["max"])
        s = v.get("score", mx)
        pct = 100.0 * s / mx if mx else 0
        rows.append({
            "key": p["key"], "label": p["label"], "group": p["group"], "max": mx,
            "score": s, "pct": pct, "band": scoring.band_for_pct(pct),
            "evidence": v.get("evidence", ""),
        })

    recs = scoring.coaching_recommendations(scores) if scores else []
    pct = (100.0 * (call["total_score"] or 0) / (call["max_score"] or 100)) if call.get("grade") else None

    return render_template(
        "report_card.html", call=call, rows=rows, groups=scoring.GROUPS,
        recs=recs, max_total=scoring.MAX_TOTAL, pct=pct,
    )


@app.route("/call/<int:call_id>/update", methods=["POST"])
def update_call(call_id):
    call = storage.get_call(call_id)
    if not call:
        return jsonify({"error": "not found"}), 404

    scores = call.get("scores") or {}
    for p in scoring.PARAMETERS:
        key = p["key"]
        score_field = f"score_{key}"
        evidence_field = f"evidence_{key}"
        if score_field in request.form:
            try:
                s = float(request.form.get(score_field))
            except (TypeError, ValueError):
                s = scores.get(key, {}).get("score", p["max"])
            s = max(0.0, min(float(p["max"]), s))
            evidence = request.form.get(evidence_field, scores.get(key, {}).get("evidence", ""))
            scores[key] = {"score": s, "evidence": evidence, "max": p["max"]}

    total = scoring.totals_from_scores(scores)
    grade, _pct = scoring.grade_for(total)
    method = call.get("scoring_method") or "Heuristic draft"
    if "human-reviewed" not in method:
        method = f"{method} + human-reviewed"
    storage.save_scores(call_id, scores, total, scoring.MAX_TOTAL, grade, method)

    return redirect(url_for("report_card", call_id=call_id))


@app.route("/overview")
def overview():
    calls = [c for c in storage.list_calls() if c["status"] == "done"]

    param_avgs = []
    for p in scoring.PARAMETERS:
        vals = []
        for c in calls:
            v = (c.get("scores") or {}).get(p["key"])
            if v:
                vals.append(v.get("score", 0))
        avg = sum(vals) / len(vals) if vals else 0
        pct = 100.0 * avg / p["max"] if p["max"] else 0
        param_avgs.append({"label": p["label"], "avg": round(avg, 1), "max": p["max"],
                            "pct": round(pct, 1), "band": scoring.band_for_pct(pct)})

    ranked = sorted(calls, key=lambda c: (c["total_score"] or 0), reverse=True)
    total_scores = [c["total_score"] for c in calls if c["total_score"] is not None]
    avg_total = round(sum(total_scores) / len(total_scores), 1) if total_scores else 0
    pass_count = sum(1 for c in calls if (c["total_score"] or 0) >= 70)

    return render_template(
        "overview.html", calls=ranked, param_avgs=param_avgs, avg_total=avg_total,
        pass_count=pass_count, total_calls=len(calls), max_total=scoring.MAX_TOTAL,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, port=port)
