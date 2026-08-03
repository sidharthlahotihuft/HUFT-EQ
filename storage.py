"""
Persistence layer for the CS Call Quality Portal, backed by Supabase (Postgres + Storage).

Uses the Supabase service_role key server-side only (never exposed to the browser).
Row Level Security stays default-deny on the `calls` table - the service role key
bypasses RLS, and no other key is ever used to touch this table.

Run supabase/schema.sql once in your Supabase project's SQL editor before first use.
"""
import json
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


def create_call(filename, original_name, agent_name, call_date, call_topic, content_hash=None):
    row = {
        "filename": filename,
        "original_name": original_name,
        "agent_name": agent_name,
        "call_date": call_date or None,
        "call_topic": call_topic,
        "content_hash": content_hash,
        "status": "uploaded",
        "status_message": "Uploaded - ready to transcribe.",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    res = get_client().table("calls").insert(row).execute()
    return res.data[0]["id"]


def find_by_hash(content_hash):
    """Return an existing call with the same audio fingerprint, or None. Used to
    detect and skip duplicate uploads (same recording uploaded twice)."""
    if not content_hash:
        return None
    try:
        res = (get_client().table("calls").select("id, original_name, status")
               .eq("content_hash", content_hash).limit(1).execute())
    except Exception:  # noqa: BLE001 - e.g. column not created yet -> treat as no match
        return None
    rows = res.data or []
    return rows[0] if rows else None


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
    sj = row.get("scores_json")
    # jsonb normally comes back as a dict, but guard against a string-encoded
    # value (older rows / manual edits) so aggregation never crashes the page.
    if isinstance(sj, str):
        try:
            sj = json.loads(sj)
        except Exception:  # noqa: BLE001
            sj = {}
    row["scores"] = sj if isinstance(sj, dict) else {}
    return row


def set_excluded(call_id, excluded):
    """Soft-remove a call from the dashboard and scoring aggregates (reversible).
    Stored as scores_json.excluded so no schema change is needed; the call and
    its report card are kept, just left out of team stats."""
    res = get_client().table("calls").select("scores_json").eq("id", call_id).execute()
    rows = res.data or []
    sj = (rows[0].get("scores_json") if rows else None) or {}
    sj["excluded"] = bool(excluded)
    get_client().table("calls").update({"scores_json": sj}).eq("id", call_id).execute()


def purge_older_than(days):
    """Data-retention: permanently delete calls created more than `days` ago,
    along with their audio files in Storage. Returns the number of calls deleted.
    Called by the scheduled /api/cleanup job."""
    from datetime import datetime, timedelta, timezone
    import audio_storage

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    client = get_client()
    res = client.table("calls").select("id, filename").lt("created_at", cutoff).execute()
    rows = res.data or []
    deleted = 0
    for r in rows:
        if r.get("filename"):
            try:
                audio_storage.delete_object(r["filename"])
            except Exception as e:  # noqa: BLE001 - keep going even if one audio file is gone
                print(f"[retention] could not delete audio {r['filename']}: {e}")
        client.table("calls").delete().eq("id", r["id"]).execute()
        deleted += 1
    return deleted
