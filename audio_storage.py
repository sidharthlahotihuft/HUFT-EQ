"""
Audio file handling via Supabase Storage.

Uploads go DIRECT from the browser to Supabase Storage using a short-lived signed
upload URL (see /api/upload-url in app.py) - this deliberately bypasses our own
Vercel function, because Vercel Functions cap request bodies at 4.5MB and call
recordings routinely exceed that. The backend only ever generates the token and,
later, downloads the file server-side (via the service role key) to feed it to
Whisper - outgoing downloads aren't subject to that inbound body-size limit.
"""
import os
import uuid

BUCKET = os.environ.get("SUPABASE_AUDIO_BUCKET", "call-recordings")


def create_signed_upload(original_name):
    """Returns {path, token, bucket} for the browser to upload directly to Supabase Storage."""
    from storage import get_client

    ext = original_name.rsplit(".", 1)[-1].lower() if "." in original_name else "bin"
    path = f"{uuid.uuid4().hex}.{ext}"
    resp = get_client().storage.from_(BUCKET).create_signed_upload_url(path)
    # supabase-py returns keys like "path"/"signed_url"/"token" (or camelCase depending on version)
    token = resp.get("token") or resp.get("signedUrl", {}).get("token")
    return {"bucket": BUCKET, "path": path, "token": token}


def download_to_path(storage_path, dest_path):
    """Downloads a file from Supabase Storage to a local path (used before transcription)."""
    from storage import get_client

    data = get_client().storage.from_(BUCKET).download(storage_path)
    with open(dest_path, "wb") as f:
        f.write(data)
    return dest_path
