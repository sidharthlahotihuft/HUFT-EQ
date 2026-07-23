"""
Persistence layer for the CS Call Quality Portal, backed by Supabase (Postgres + Storage).

Uses the Supabase service_role key server-side only (never exposed to the browser).
Row Level Security stays default-deny on the `calls` table - the service role key
bypasses RLS, and no other key is ever used to touch this table.

Run supabase/schema.sql once in your Supabase project's SQL editor before first use.
"""
import os
from datetime import datetime, timezone

_client = None


def get_client():
    """Lazily creates and caches the Supabase client (service role)."""
    global _client
    if _client is None:
        from supabase import create_client

        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set (see .env.example)."
            )
        _client = create_client(url, key)
    return _client


def init_db():
    """No-op: schema is managed via supabase/schema.sql, run once in the Supabase SQL editor."""
    return None


def create_call(filename, original_name, agent_name, call_date, call_topic):
    row = {
        "filename": filename,
        "original_name": original_name,
        "agent_name": agent_name,
        "call_date": call_date or None,
        "call_topic": call_topic,
        "status": "uploaded",
        "status_message": "Uploaded - ready to transcribe.",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    res = get_client().table("calls").insert(row).execute()
    return res.data[0]["id"]


def update_status(call_id, status, message=""):
    get_client().table("calls").update(
        {"status": status, "status_message": message}
    ).eq("id", call_id).execute()


def save_transcript(call_id, transcript):
    get_client().table("calls").update({"transcript": transcript}).eq("id", call_id).execute()


def save_scores(call_id, scores, total, max_score, grade, method):
    get_client().table("calls").update(
        {
            "scores_json": scores,
            "total_score": total,
            "max_score": max_score,
            "grade": grade,
            "scoring_method": method,
        }
    ).eq("id", call_id).execute()


def get_call(call_id):
    res = get_client().table("calls").select("*").eq("id", call_id).execute()
    rows = res.data or []
    return _normalize(rows[0]) if rows else None


def list_calls(order_by="created_at", desc=True):
    res = get_client().table("calls").select("*").order(order_by, desc=desc).execute()
    return [_normalize(r) for r in (res.data or [])]


def _normalize(row):
    row = dict(row)
    row["scores"] = row.get("scores_json") or {}
    return row
