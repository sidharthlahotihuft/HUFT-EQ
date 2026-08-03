"""
Speaker labeling + light cleanup: turn a raw transcript into a clean,
turn-by-turn "HUFT Agent:" / "Customer:" conversation.

Three cases:
  * Deepgram returns speaker-labeled utterances (acoustic diarization). If they
    are cleanly separated, they're kept as-is.
  * Deepgram sometimes merges both people into a few huge blocks (poor
    diarization on mono/low-fidelity audio). We detect that ("under-segmented")
    and re-split it with the LLM.
  * Plain Whisper returns one unlabeled block -> the LLM splits it.

The LLM pass keeps the words but is allowed to tidy for readability: fix
obviously garbled capitalization of romanized Hindi words, and collapse an
immediately-repeated filler ("Hello. Hello. Hello." -> "Hello."). It must not
translate, summarise, or drop any substantive content. Needs an LLM key; without
one, the raw transcript is returned unchanged (never destructive).
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


def _under_segmented(transcript):
    """True if labeled but the turns are huge blocks (both speakers merged) -
    a sign diarization did a poor job and it should be re-split. Threshold:
    more than ~55 words per labeled turn on a reasonably long transcript."""
    labels = len(_LABEL_RE.findall(transcript))
    words = len(transcript.split())
    if labels < 2 or words < 120:
        return False
    return (words / labels) > 55


def _prompt(transcript):
    return f"""Below is a transcript of a single phone call between a HUFT
(Head Up For Tails) customer-care AGENT and a CUSTOMER, in English, Hindi or Hinglish.
Produce a clean, readable, turn-by-turn version.

Rules:
- Split the conversation into turns and label each with the speaker.
- Use exactly these labels: "HUFT Agent:" and "Customer:". One turn per line.
- The AGENT greets, represents HUFT, verifies identity and drives the resolution.
  The CUSTOMER describes the problem or request.
- KEEP the words. You MAY, for readability only:
    - fix obviously garbled capitalization of names/words (e.g. "sevi" -> "Sevi",
      "gayatri kaura" -> "Gayatri Kaur");
    - collapse an immediately-repeated filler ("Hello. Hello. Hello." -> "Hello.").
- Do NOT translate, summarise, reorder, or drop any substantive content, numbers,
  commitments or details. If a number is already masked as [number]/[email], keep it.

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
    """Guard against the LLM rewriting the call: the result must be labeled and
    not wildly different in length (light cleanup may trim a little)."""
    if not labeled or not is_already_labeled(labeled):
        return False
    ow = len(original.split())
    lw = len(_LABEL_RE.sub("", labeled).split())
    if ow == 0:
        return False
    ratio = lw / ow
    return 0.5 <= ratio <= 1.4


def ensure_labeled(transcript):
    """Return a clean, speaker-labeled transcript. Best-effort and
    non-destructive: returns the original if it's already clean or if labeling
    isn't possible."""
    if not transcript:
        return transcript
    # Already labeled AND well-segmented -> leave it alone.
    if is_already_labeled(transcript) and not _under_segmented(transcript):
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
