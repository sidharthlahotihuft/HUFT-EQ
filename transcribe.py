"""
Audio transcription for the HUFT CS Call Quality Portal.

Accuracy is the whole point here: a bad transcript makes every downstream
score wrong. So this module does four things beyond a naive Whisper call, each
targeting a common failure mode on real support-call audio:

  1. Domain prompt  - Whisper is given a hint listing the brand name and the
     vocabulary that shows up on HUFT calls (products, actions, Hindi/English
     support terms) so it stops mishearing "Head Up For Tails", product names,
     order IDs, agent names, etc. See WHISPER_PROMPT / DEFAULT_PROMPT below.
  2. Language hint  - support calls are often Hinglish (Hindi + English mixed).
     Set WHISPER_LANGUAGE ("en", "hi", ...) to stop Whisper guessing wrong on
     short/noisy clips. Left blank = auto-detect (best for genuinely mixed calls).
  3. temperature=0  - deterministic decoding. The default (non-zero) sampling
     is what produces looped/repeated phrases and invented words on silence.
  4. Long-call handling - the OpenAI Whisper API rejects files over 25MB.
     Longer recordings are automatically split into overlapping chunks (when
     ffmpeg/pydub are available) and stitched back together, with each chunk
     primed by the tail of the previous one so wording stays consistent.

Backend preference order:
  1. OpenAI Whisper API ("whisper-1") if OPENAI_API_KEY is set - fast, accurate,
     handles most audio formats directly, no local model download.
  2. faster-whisper (local, offline) if the `faster_whisper` package is
     installed - no API key needed, downloads a model on first use, needs ffmpeg.
  3. Otherwise, raises a clear error telling the user how to enable one path.
"""
import os

_local_model = None  # lazy-loaded faster_whisper model, cached across calls

# OpenAI's Whisper API hard-rejects uploads larger than 25MB. Stay a little
# under it to leave room for multipart overhead.
_MAX_API_BYTES = 24 * 1024 * 1024

# Default domain prompt. Whisper uses this as a style/vocabulary hint (max ~224
# tokens are actually used). List real names/terms so they're transcribed
# correctly instead of phonetically. Override with the WHISPER_PROMPT env var,
# and ideally extend it with your actual agent names and top product names.
# IMPORTANT: keep this a bare comma-separated vocabulary list, NOT full
# sentences. Whisper echoes its prompt when the audio has no clear speech, and
# a prompt made of grammatical sentences gets parroted back as a fluent-looking
# (but fake) transcript. A keyword list can't be looped into a plausible
# sentence, so silent/empty audio yields little or nothing instead of garbage,
# and the hallucination guard below can then catch it.
DEFAULT_PROMPT = (
    "Head Up For Tails, HUFT, order, delivery, refund, return, replacement, "
    "exchange, subscription, grooming, appointment, vet, pet food, kibble, "
    "treats, collar, leash, harness, bed, shampoo, order ID, tracking, courier, "
    "prepaid, COD, wallet, coupon, loyalty points, cancellation"
)

# When WHISPER_ROMANIZE is on we want Hindi written in the Latin alphabet the way
# people actually type Hinglish, not in Devanagari. Whisper tends to mimic the
# SCRIPT of its prompt, so a romanized-Hinglish keyword prompt nudges the whole
# transcript to come out romanized. Any Devanagari that still slips through is
# converted afterward by _romanize() as a safety net. Still a keyword list (no
# full sentences) so it can't be parroted into a fake transcript.
DEFAULT_PROMPT_ROMANIZED = (
    "Head Up For Tails, HUFT, namaste sir, ji haan, theek hai, aapka order, "
    "delivery kahan hai, refund, return, replacement, exchange, cancel, "
    "grooming appointment, pet food, kibble, treats, collar, leash, order ID, "
    "tracking, courier, prepaid, COD, coupon, main aapki madad karta hun, "
    "ek minute, ho jayega, dhanyavaad"
)


def _romanize_enabled():
    return (os.environ.get("WHISPER_ROMANIZE", "1").strip().lower()
            in ("1", "true", "yes", "on"))


def _romanize(text):
    """Convert any Devanagari spans in `text` to Latin (Hinglish). Latin/English
    text is left exactly as-is. This is a safety net - most output should already
    be romanized thanks to the romanized prompt. Returns text unchanged if the
    transliteration library isn't installed (so it degrades gracefully)."""
    if not text:
        return text
    import re
    if not re.search(r"[ऀ-ॿ]", text):
        return text  # nothing in Devanagari, skip
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except Exception:
        return text  # library missing -> leave Devanagari as-is rather than crash
    scheme = getattr(sanscript, os.environ.get("ROMANIZE_SCHEME", "OPTITRANS"),
                     sanscript.OPTITRANS)
    dev = re.compile(r"[ऀ-ॿ]+")
    return dev.sub(lambda m: transliterate(m.group(0), sanscript.DEVANAGARI, scheme), text)


def _looks_like_hallucination(text):
    """Whisper loops a short phrase when it can't find real speech (silence,
    hold music, corrupt/empty audio). Detect that so we fail loudly instead of
    feeding looped garbage into scoring. Returns True if the text is dominated
    by one repeated phrase."""
    if not text:
        return False
    words = text.split()
    if len(words) < 30:
        return False  # too short to judge; let it through
    # Compare the set of distinct sentences to the total: heavy repetition means
    # very few distinct sentences relative to how many there are.
    import re
    sentences = [s.strip().lower() for s in re.split(r"[.!?]+", text) if s.strip()]
    if len(sentences) >= 5:
        distinct = set(sentences)
        if len(distinct) / len(sentences) < 0.25:
            return True
    # Also catch low overall vocabulary diversity (same few words on a loop).
    if len(set(w.lower() for w in words)) / len(words) < 0.12:
        return True
    return False


class TranscriptionError(RuntimeError):
    pass


def _whisper_language():
    lang = (os.environ.get("WHISPER_LANGUAGE") or "").strip()
    return lang or None  # None -> let Whisper auto-detect


def _whisper_prompt():
    # Explicit empty string ("WHISPER_PROMPT=") disables the prompt; unset uses
    # the default (romanized variant when WHISPER_ROMANIZE is on).
    val = os.environ.get("WHISPER_PROMPT")
    if val is None:
        return DEFAULT_PROMPT_ROMANIZED if _romanize_enabled() else DEFAULT_PROMPT
    val = val.strip()
    return val or None


def _whisper_temperature():
    try:
        return float(os.environ.get("WHISPER_TEMPERATURE", "0"))
    except ValueError:
        return 0.0


def _call_openai_whisper(client, audio_path, model, prompt):
    """One API call for a single (already size-checked) file."""
    kwargs = {
        "model": model,
        "temperature": _whisper_temperature(),
    }
    lang = _whisper_language()
    if lang:
        kwargs["language"] = lang
    if prompt:
        kwargs["prompt"] = prompt
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(file=f, **kwargs)
    return (resp.text or "").strip()


def _split_audio(audio_path, chunk_ms=10 * 60 * 1000, overlap_ms=3000):
    """Split a large file into <=chunk_ms segments (with small overlap) as temp
    wav files. Returns a list of paths, or None if pydub/ffmpeg aren't available."""
    try:
        from pydub import AudioSegment
    except Exception:
        return None
    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception:
        return None  # ffmpeg missing or unreadable format
    import tempfile
    paths = []
    start = 0
    length = len(audio)
    while start < length:
        end = min(start + chunk_ms, length)
        seg = audio[start:end]
        fd, p = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        seg.export(p, format="wav")
        paths.append(p)
        if end >= length:
            break
        start = end - overlap_ms  # small overlap so words on the boundary aren't lost
    return paths


def _transcribe_with_deepgram(audio_path):
    """Preferred backend: Deepgram Nova-3 does verbatim transcription AND speaker
    diarization (HUFT Agent / Customer labels) in one call, tuned for Hinglish.
    Returns None when DEEPGRAM_API_KEY isn't set, so the Whisper paths still work."""
    if not os.environ.get("DEEPGRAM_API_KEY"):
        return None
    try:
        from transcribe_deepgram import transcribe_deepgram, DeepgramError
    except ImportError:
        return None
    try:
        return transcribe_deepgram(audio_path)
    except DeepgramError as e:
        raise TranscriptionError(str(e)) from e


def _transcribe_with_openai_api(audio_path):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    client = OpenAI(api_key=api_key)
    model = os.environ.get("OPENAI_WHISPER_MODEL", "whisper-1")
    base_prompt = _whisper_prompt()

    size = os.path.getsize(audio_path)
    if size <= _MAX_API_BYTES:
        return _call_openai_whisper(client, audio_path, model, base_prompt)

    # Too big for one API call -> chunk it.
    chunks = _split_audio(audio_path)
    if not chunks:
        raise TranscriptionError(
            f"Audio is {size / 1024 / 1024:.1f}MB, over the 25MB Whisper API "
            "limit, and it can't be auto-split here (ffmpeg/pydub not available). "
            "Install ffmpeg + `pip install pydub`, or upload a shorter/compressed "
            "recording (e.g. 64kbps mono MP3)."
        )
    parts = []
    try:
        for p in chunks:
            # Prime each chunk with the brand prompt + the tail of what we've
            # transcribed so far, so names and phrasing stay consistent.
            tail = " ".join(" ".join(parts).split()[-40:])
            prompt = (base_prompt or "")
            if tail:
                prompt = f"{prompt} {tail}".strip()
            parts.append(_call_openai_whisper(client, p, model, prompt or None))
    finally:
        for p in chunks:
            if os.path.exists(p):
                os.remove(p)
    return " ".join(t for t in parts if t).strip()


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
    segments, _info = _local_model.transcribe(
        audio_path,
        beam_size=5,
        temperature=0,
        language=_whisper_language(),      # None -> auto-detect
        initial_prompt=_whisper_prompt(),  # same domain vocabulary hint
        vad_filter=True,                   # skip long silences -> fewer hallucinations
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


def transcribe(audio_path):
    """Returns the transcript text for the given audio file path."""
    for fn in (_transcribe_with_deepgram, _transcribe_with_openai_api, _transcribe_with_faster_whisper):
        try:
            text = fn(audio_path)
        except TranscriptionError:
            raise
        except Exception as e:  # noqa: BLE001
            raise TranscriptionError(f"{fn.__name__} failed: {e}") from e
        if text:
            if _romanize_enabled():
                text = _romanize(text)
            if _looks_like_hallucination(text):
                raise TranscriptionError(
                    "Transcription produced looped/repeated text, which means "
                    "Whisper couldn't find clear speech in the audio. Common "
                    "causes: the recording is silent or near-silent, it's mostly "
                    "hold music/ringing, the volume is extremely low, or the file "
                    "is corrupt or in an unexpected format. Check that this call's "
                    "audio actually plays and contains audible conversation, then "
                    "re-run. (If it's a real but very quiet call, try normalizing/"
                    "amplifying the audio before upload.)"
                )
            return text
    raise TranscriptionError(
        "No transcription backend available. Set OPENAI_API_KEY (Vercel/Supabase deploys "
        "require this - local model files aren't practical in serverless). For local-only "
        "dev without an API key, `pip install faster-whisper` for offline transcription."
    )
