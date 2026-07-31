"""
Speaker labeling: ensure every transcript is split into
"HUFT Agent:" / "Customer:" lines.

Two cases:
  * Deepgram already returns speaker-labeled utterances (acoustic diarization) -
    such transcripts are detected and returned unchanged.
  * Plain Whisper returns one unlabeled block. Here we ask the LLM (the same
    ANTHROPIC_API_KEY / OPENAI_API_KEY used for scoring) to assign each turn to
    the agent or the customer WITHOUT changing any words - verbatim, just split
    and labeled. If no LLM key is available, the transcript is returned
    unchanged (labeling is best-effort, never destructive).

The agent is the HUFT customer-care representative (greets, represents HUFT,
drives the resolution); the customer is the person with the issue.
"""
import os
import re

_LABEL_RE = re.compile(r"^\s*(HUFT Agent|Agent|Customer|Caller|Speaker\s*\d+)\s*:",
                       re.IGNORECASE | re.MULTILINE)


def is_already_labeled(transcript):
    """True if the transcript already has speaker labels on multiple lines."""
    if not transcript:
        return False
    return len(_LABEL_RE.findall(transcript)) >= 2


def _prompt(transcript):
    return f"""Below is a verbatim transcript of a single phone call between a HUFT
(Head Up For Tails) customer-care AGENT and a CUSTOMER. It may be in English, Hindi or
Hinglish. Your ONLY job is to split it into turns and label each turn with the speaker.

Rules:
- Do NOT translate, summarise, correct, add or remove any words. Keep the text verbatim.
- Only decide who is speaking and insert a label at the start of each turn.
- The AGENT greets, represents HUFT, asks verifying questions and drives the resolution.
  The CUSTOMER describes the problem or request.
- Use exactly these labels: "HUFT Agent:" and "Customer:".
- Output one turn per line, nothing else (no commentary, no numbering).

Transcript:
\"\"\"
{transcript}
\"\"\"
"""


def _label_with_anthropic(transcript):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model, max_tokens=4000, temperature=0,
        messages=[{"role": "user", "content": _prompt(transcript)}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()


def _label_with_openai(transcript):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    model = os.environ.get("OPENAI_SCORING_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user", "content": _prompt(transcript)}],
    )
    return (resp.choices[0].message.content or "").strip()


def _sanity_ok(original, labeled):
    """Guard against the LLM rewriting the call: the labeled version must be
    labeled and not wildly different in length from the original word count."""
    if not labeled or not is_already_labeled(labeled):
        return False
    ow = len(original.split())
    # strip the labels before counting words
    lw = len(_LABEL_RE.sub("", labeled).split())
    if ow == 0:
        return False
    ratio = lw / ow
    return 0.6 <= ratio <= 1.4


def ensure_labeled(transcript):
    """Return a speaker-labeled transcript. Best-effort and non-destructive:
    returns the original text unchanged if already labeled or if labeling
    isn't possible."""
    if not transcript or is_already_labeled(transcript):
        return transcript
    if os.environ.get("SPEAKER_LABEL", "1").strip().lower() not in ("1", "true", "yes", "on"):
        return transcript
    for fn in (_label_with_anthropic, _label_with_openai):
        try:
            labeled = fn(transcript)
        except Exception as e:  # noqa: BLE001 - labeling must never break transcription
            print(f"[speaker_label] {fn.__name__} failed: {e}")
            labeled = None
        if labeled and _sanity_ok(transcript, labeled):
            return labeled
    return transcript  # give up gracefully, keep the raw transcript
