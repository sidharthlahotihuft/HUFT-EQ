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
import json
import os
import tempfile

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash

import audio_storage
import pii
import scoring
import speaker_label
import storage
import transcribe

load_dotenv()

app = Flask(__name__, static_folder="public/static", static_url_path="/static")
# Session signing key. MUST be set in production (Vercel env) so sessions stay
# valid across serverless instances; the fallback is for local dev only.
app.secret_key = os.environ.get("SECRET_KEY", "huft-care-portal-dev-secret-change-me")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")

# --- Authentication & roles -------------------------------------------------
# Roles: "manager" sees every call and the dashboard; "agent" sees only their
# own calls. Users come from the USERS_JSON env var (a JSON list); if unset, a
# single default admin manager is used so the app works out of the box.
#
# USERS_JSON example (set in .env / Vercel):
#   [{"email":"manager@headsupfortails.com","password":"...","role":"manager"},
#    {"email":"akhila@headsupfortails.com","password":"...","role":"agent","agent_name":"Akhila"}]
# Use "password_hash" (a werkzeug hash) instead of "password" to avoid plaintext.
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@headsupfortails.com").strip().lower()
_DEFAULT_PW_HASH = (
    "scrypt:32768:8:1$7FB5XSMO8fZZLM4k$b76aaeb390c7e293be82b87b9478543dee3bbe9"
    "72bd0bb6be9988544fdec1065b9d35b3490c1b7eee188e2d5ee273231e239b4162fd3e88c6569d90b64881203"
)
ADMIN_PASSWORD_HASH = os.environ.get("ADMIN_PASSWORD_HASH", _DEFAULT_PW_HASH)
ADMIN_PASSWORD_PLAIN = os.environ.get("ADMIN_PASSWORD")  # optional plaintext override


def _load_users():
    raw = os.environ.get("USERS_JSON")
    if raw:
        try:
            out = []
            for u in json.loads(raw):
                out.append({
                    "email": (u.get("email") or "").strip().lower(),
                    "password": u.get("password"),
                    "password_hash": u.get("password_hash"),
                    "role": (u.get("role") or "agent").strip().lower(),
                    "agent_name": (u.get("agent_name") or "").strip(),
                })
            out = [u for u in out if u["email"]]
            if out:
                return out
        except Exception as e:  # noqa: BLE001
            print(f"[auth] USERS_JSON parse failed, using default admin: {e}")
    return [{
        "email": ADMIN_EMAIL,
        "password": ADMIN_PASSWORD_PLAIN,
        "password_hash": None if ADMIN_PASSWORD_PLAIN else ADMIN_PASSWORD_HASH,
        "role": "manager", "agent_name": "",
    }]


USERS = _load_users()

# Endpoints reachable without a session (each does its own checks where needed).
PUBLIC_ENDPOINTS = {"login", "static", "api_cleanup"}


def _find_user(email):
    email = (email or "").strip().lower()
    for u in USERS:
        if u["email"] == email:
            return u
    return None


def _user_password_ok(u, pw):
    if u.get("password"):
        return pw == u["password"]
    if u.get("password_hash"):
        try:
            return check_password_hash(u["password_hash"], pw)
        except Exception:  # noqa: BLE001
            return False
    return False


def is_manager():
    return session.get("role") == "manager"


def _visible_calls(calls):
    """Managers see all; agents see only calls assigned to their agent_name."""
    if is_manager():
        return calls
    an = (session.get("agent_name") or "").strip().lower()
    return [c for c in calls if (c.get("agent_name") or "").strip().lower() == an]


def _can_view_call(call):
    if is_manager():
        return True
    an = (session.get("agent_name") or "").strip().lower()
    return (call.get("agent_name") or "").strip().lower() == an


@app.before_request
def _require_login():
    if request.endpoint in PUBLIC_ENDPOINTS or (request.path or "").startswith("/static"):
        return
    if not session.get("authed"):
        return redirect(url_for("login", next=request.path))


def _safe_next(target):
    """Only allow local redirects (prevent open-redirect via ?next=)."""
    if target and target.startswith("/") and not target.startswith("//"):
        return target
    return url_for("index")


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("authed"):
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        pw = request.form.get("password") or ""
        u = _find_user(email)
        if u and _user_password_ok(u, pw):
            session["authed"] = True
            session["email"] = u["email"]
            session["role"] = u["role"]
            session["agent_name"] = u.get("agent_name", "")
            return redirect(_safe_next(request.args.get("next")))
        error = "Invalid email or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "mp4", "webm", "ogg", "flac", "aac"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    calls = _visible_calls(storage.list_calls())
    return render_template(
        "index.html", calls=calls, params=scoring.DIMENSIONS,
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
    # An agent can only file calls under their own name (so they see them, and
    # can't attribute a call to someone else).
    if not is_manager() and session.get("agent_name"):
        agent_name = session["agent_name"]
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
        # Mask PII (phone, email, card/account, PIN) BEFORE any further AI or
        # storage, so sensitive customer data never reaches the scoring model
        # or the manager view in the clear.
        transcript = pii.mask_pii(transcript)
        # Ensure the transcript is split into "HUFT Agent:" / "Customer:" turns.
        # Deepgram already labels speakers; for plain Whisper this adds labels
        # via an LLM pass (verbatim, non-destructive). No-op without an LLM key.
        transcript = speaker_label.ensure_labeled(transcript)
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
        storage.update_status(call_id, "scoring", "Scoring call against the HUFT Care model...")
        meta = {
            "agent_name": call.get("agent_name"),
            "call_date": call.get("call_date"),
            "call_topic": call.get("call_topic"),
        }
        result, method = scoring.score_transcript(call["transcript"], meta)
        # display_score is 1-5 (or None when evidence is insufficient); rating_band is the label.
        total = result.get("display_score")
        band = result.get("rating_band")
        storage.save_scores(call_id, result, total, scoring.MAX_SCORE, band, method)
        storage.update_status(call_id, "done", "Scoring complete.")
        return jsonify({"status": "done", "total_score": total, "grade": band})
    except Exception as e:  # noqa: BLE001
        storage.update_status(call_id, "error", f"Unexpected error during scoring: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/call/<int:call_id>")
def report_card(call_id):
    call = storage.get_call(call_id)
    if not call:
        return "Call not found", 404
    if not _can_view_call(call):
        return "You don't have access to this call.", 403

    result = call.get("scores") or {}
    dims = result.get("dimensions") or {}
    rows = []
    for d in scoring.DIMENSIONS:
        v = dims.get(d["key"], {})
        score = v.get("score")
        applicable = v.get("applicable", True)
        rows.append({
            "key": d["key"], "label": d["label"], "weight": d["weight"],
            "question": d["question"], "plain": d.get("plain", d["question"]),
            "score": score, "applicable": applicable,
            "confidence": v.get("confidence"),
            "band": scoring.band_for_score(score) if applicable else "na",
            "evidence": v.get("evidence", ""),
        })

    recs = scoring.coaching_recommendations(result) if dims else []

    return render_template(
        "report_card.html", call=call, rows=rows, result=result,
        max_score=scoring.MAX_SCORE,
    )


@app.route("/call/<int:call_id>/update", methods=["POST"])
def update_call(call_id):
    call = storage.get_call(call_id)
    if not call:
        return jsonify({"error": "not found"}), 404
    if not _can_view_call(call):
        return "You don't have access to this call.", 403

    result = call.get("scores") or scoring.empty_result()
    dims = result.setdefault("dimensions", {})
    for d in scoring.DIMENSIONS:
        key = d["key"]
        entry = dims.setdefault(key, {"score": None, "applicable": True,
                                      "evidence_sufficient": False, "evidence": "",
                                      "weight": d["weight"]})
        # Applicability: checkbox "applicable_<key>" present == applicable.
        applicable = f"applicable_{key}" in request.form
        entry["applicable"] = applicable

        score_field = f"score_{key}"
        if score_field in request.form:
            raw = (request.form.get(score_field) or "").strip()
            if raw == "" or not applicable:
                entry["score"] = None
                entry["evidence_sufficient"] = False
            else:
                try:
                    s = max(1.0, min(scoring.MAX_SCORE, float(raw)))
                except (TypeError, ValueError):
                    s = entry.get("score")
                entry["score"] = s
                entry["evidence_sufficient"] = s is not None
        evidence_field = f"evidence_{key}"
        if evidence_field in request.form:
            entry["evidence"] = request.form.get(evidence_field, entry.get("evidence", ""))
        entry["weight"] = d["weight"]

    # Let a reviewer clear/keep the critical & high-severity flags.
    result["critical_failure"] = "critical_failure" in request.form
    result["high_severity_failure"] = "high_severity_failure" in request.form

    scoring.recompute(result)

    method = call.get("scoring_method") or "Manual"
    if "human-reviewed" not in method:
        method = f"{method} + human-reviewed"
    storage.save_scores(call_id, result, result.get("display_score"),
                        scoring.MAX_SCORE, result.get("rating_band"), method)

    return redirect(url_for("report_card", call_id=call_id))


def _call_day(call):
    """The date a call belongs to for filtering: its call_date, else the day it
    was created. Returns a 'YYYY-MM-DD' string or ''."""
    d = (call.get("call_date") or "").strip()
    if d:
        return d[:10]
    ca = (call.get("created_at") or "").strip()
    return ca[:10] if ca else ""


@app.route("/overview")
def overview():
    # The team dashboard is manager-only; agents are sent to their own calls.
    if not is_manager():
        return redirect(url_for("index"))
    calls = [c for c in storage.list_calls() if c["status"] == "done"]

    # --- Time filter (Manager dashboard) ---
    date_from = (request.args.get("from") or "").strip()
    date_to = (request.args.get("to") or "").strip()
    if date_from or date_to:
        def in_range(c):
            day = _call_day(c)
            if not day:
                return False
            if date_from and day < date_from:
                return False
            if date_to and day > date_to:
                return False
            return True
        calls = [c for c in calls if in_range(c)]

    param_avgs = []
    for d in scoring.DIMENSIONS:
        vals = []
        for c in calls:
            v = ((c.get("scores") or {}).get("dimensions") or {}).get(d["key"])
            # Average only applicable, scored dimensions (N/A excluded).
            if v and v.get("applicable") and v.get("score") is not None:
                vals.append(float(v["score"]))
        avg = sum(vals) / len(vals) if vals else 0
        pct = 100.0 * avg / scoring.MAX_SCORE if avg else 0
        param_avgs.append({"label": d["label"], "avg": round(avg, 2),
                           "max": scoring.MAX_SCORE, "n": len(vals),
                           "pct": round(pct, 1), "band": scoring.band_for_score(avg)})

    ranked = sorted(calls, key=lambda c: (c["total_score"] or 0), reverse=True)
    total_scores = [c["total_score"] for c in calls if c["total_score"] is not None]
    avg_total = round(sum(total_scores) / len(total_scores), 2) if total_scores else 0
    # "Pass" = displayed score >= 3.5 (Strong or better) on the 1-5 scale.
    pass_count = sum(1 for c in calls if (c["total_score"] or 0) >= 3.5)

    summary = scoring.manager_summary(calls)

    return render_template(
        "overview.html", calls=ranked, param_avgs=param_avgs, avg_total=avg_total,
        pass_count=pass_count, total_calls=len(calls), max_score=scoring.MAX_SCORE,
        summary=summary, date_from=date_from, date_to=date_to,
    )


RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "90"))
CRON_SECRET = os.environ.get("CRON_SECRET", "")


@app.route("/api/cleanup", methods=["GET", "POST"])
def api_cleanup():
    """Data-retention job: delete calls (and their audio) older than
    RETENTION_DAYS. Authorised either by a logged-in manager, or by a cron
    secret (Vercel Cron sends 'Authorization: Bearer <CRON_SECRET>'). This is
    the automated deletion policy — 90 days by default."""
    authed_manager = session.get("authed") and is_manager()
    supplied = (request.headers.get("Authorization", "").replace("Bearer ", "").strip()
                or request.headers.get("x-cron-secret", "").strip()
                or request.args.get("secret", "").strip())
    cron_ok = bool(CRON_SECRET) and supplied == CRON_SECRET
    if not (authed_manager or cron_ok):
        return jsonify({"error": "unauthorized"}), 403
    try:
        deleted = storage.purge_older_than(RETENTION_DAYS)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500
    return jsonify({"deleted": deleted, "retention_days": RETENTION_DAYS})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, port=port)
