"""
Audio transcription for the CS Call Quality Portal.

Preference order:
  1. OpenAI Whisper API ("whisper-1") if OPENAI_API_KEY is set - fast, accurate,
     handles most audio formats directly, no local model download.
  2. faster-whisper (local, offline, CPU-friendly reimplementation of Whisper)
     if the `faster_whisper` package is installed - no API key needed, but
     downloads a model (~150MB for "base") on first use and requires ffmpeg.
  3. Otherwise, raises a clear error telling the user how to enable one path.
"""
import os

_local_model = None  # lazy-loaded faster_whisper model, cached across calls


class TranscriptionError(RuntimeError):
    pass


def _transcribe_with_openai_api(audio_path):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    client = OpenAI(api_key=api_key)
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=os.environ.get("OPENAI_WHISPER_MODEL", "whisper-1"),
            file=f,
        )
    return resp.text


def _transcribe_with_faster_whisper(audio_path):
    global _local_model
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None
    if _local_model is None:
        model_size = os.environ.get("FASTER_WHISPER_MODEL", "base")
        compute_type = os.environ.get("FASTER_WHISPER_COMPUTE", "int8")
        _local_model = WhisperModel(model_size, compute_type=compute_type)
    segments, _info = _local_model.transcribe(audio_path, beam_size=5)
    return " ".join(seg.text.strip() for seg in segments).strip()


def transcribe(audio_path):
    """Returns the transcript text for the given audio file path."""
    for fn in (_transcribe_with_openai_api, _transcribe_with_faster_whisper):
        try:
            text = fn(audio_path)
        except Exception as e:  # noqa: BLE001
            raise TranscriptionError(f"{fn.__name__} failed: {e}") from e
        if text:
            return text
    raise TranscriptionError(
        "No transcription backend available. Set OPENAI_API_KEY (Vercel/Supabase deploys "
        "require this - local model files aren't practical in serverless). For local-only "
        "dev without an API key, `pip install faster-whisper` for offline transcription."
    )
