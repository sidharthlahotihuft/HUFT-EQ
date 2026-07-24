"""
Deepgram Nova-3 transcription + speaker diarization for the HUFT portal.

Why this exists: OpenAI Whisper gives one unlabeled stream of text. Deepgram
Nova-3 does verbatim transcription AND separates speakers in a single API call,
is tuned for Hindi + Hinglish code-switching (customer-support speech), and
runs as a plain HTTPS request - so it works on Vercel with no heavy models,
no GPU, and no extra infrastructure.

Output is a line-by-line transcript labeled "HUFT Agent:" / "Customer:", e.g.:

    HUFT Agent: Namaste, thank you for calling Head Up For Tails, main Priya baat kar rahi hun.
    Customer: Hi, mera order abhi tak nahi aaya.
    HUFT Agent: Main abhi check karti hun, aapka order ID bata dijiye.

Enable it by setting DEEPGRAM_API_KEY in your environment (.env / Vercel).
Config (all optional):
  DEEPGRAM_MODEL     default "nova-3"
  DEEPGRAM_LANGUAGE  default "multi" (best for Hinglish); or "hi", "en", ...
  DEEPGRAM_KEYTERMS  comma-separated terms to boost (defaults to a HUFT list)
  WHISPER_ROMANIZE   "1" (default) writes Hindi in Latin Hinglish; "0" keeps Devanagari
"""
import os
import re

import requests

_ENDPOINT = "https://api.deepgram.com/v1/listen"

# Terms to bias recognition toward (Nova-3 "keyterm" prompting). Extend with your
# real agent names and top product names for best accuracy.
DEFAULT_KEYTERMS = [
    "Head Up For Tails", "HUFT", "order ID", "delivery", "refund", "replacement",
    "exchange", "subscription", "grooming", "kibble", "treats", "collar", "leash",
    "harness", "shampoo", "COD", "prepaid", "coupon", "loyalty points", "courier",
]


class DeepgramError(RuntimeError):
    pass


def _romanize_maybe(text):
    """Reuse the portal's romanizer (Devanagari -> Hinglish) if available."""
    try:
        from transcribe import _romanize, _romanize_enabled
        return _romanize(text) if _romanize_enabled() else text
    except Exception:
        return text


def _keyterms():
    raw = os.environ.get("DEEPGRAM_KEYTERMS")
    if raw is None:
        return DEFAULT_KEYTERMS
    return [t.strip() for t in raw.split(",") if t.strip()]


_AGENT_SIGNALS = [
    "head up for tails", "huft", "thank you for calling", "how may i help",
    "how can i help", "how can i assist", "my name is", "baat kar rah",
    "aapki kya madad", "kaise madad", "main aapki", "main aapka", "order id",
    "please hold", "ek minute", "confirm kar", "dhanyavaad", "have a great day",
]


def _map_agent_customer(utterances):
    """Map Deepgram speaker numbers (0,1,...) to HUFT Agent / Customer.
    The speaker using more agent-signal phrases is the agent; ties break to
    whoever speaks first (agents open the call)."""
    texts, first_seen = {}, {}
    for i, u in enumerate(utterances):
        spk = u.get("speaker", 0)
        texts.setdefault(spk, []).append((u.get("transcript") or "").lower())
        first_seen.setdefault(spk, i)
    if not texts:
        return {}
    speakers = list(texts.keys())
    scores = {spk: sum(t.count(sig) for t in joined for sig in _AGENT_SIGNALS)
              for spk, joined in texts.items()}
    if len(speakers) == 1:
        return {speakers[0]: "HUFT Agent"}
    agent = max(speakers, key=lambda s: (scores.get(s, 0), -first_seen.get(s, 0)))
    mapping, others = {}, 0
    for spk in speakers:
        if spk == agent:
            mapping[spk] = "HUFT Agent"
        else:
            mapping[spk] = "Customer" if len(speakers) == 2 else f"Customer {others + 1}"
            others += 1
    return mapping


def transcribe_deepgram(audio_path):
    """Returns a speaker-labeled, line-by-line transcript, or None if no key set."""
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        return None

    params = [
        ("model", os.environ.get("DEEPGRAM_MODEL", "nova-3")),
        ("language", os.environ.get("DEEPGRAM_LANGUAGE", "multi")),
        ("diarize", "true"),
        ("punctuate", "true"),
        ("smart_format", "true"),
        ("utterances", "true"),
    ]
    for kt in _keyterms():
        params.append(("keyterm", kt))

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    try:
        resp = requests.post(
            _ENDPOINT,
            params=params,
            headers={"Authorization": f"Token {api_key}",
                     "Content-Type": "application/octet-stream"},
            data=audio_bytes,
            timeout=300,
        )
    except requests.RequestException as e:
        raise DeepgramError(f"Deepgram request failed: {e}") from e

    if resp.status_code == 400 and "keyterm" in resp.text.lower():
        # Keyterm prompting is Nova-3+ only; retry once without it.
        params = [p for p in params if p[0] != "keyterm"]
        resp = requests.post(
            _ENDPOINT, params=params,
            headers={"Authorization": f"Token {api_key}",
                     "Content-Type": "application/octet-stream"},
            data=audio_bytes, timeout=300,
        )
    if resp.status_code != 200:
        raise DeepgramError(f"Deepgram returned {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    utterances = (data.get("results") or {}).get("utterances") or []
    if not utterances:
        # Fall back to the flat transcript if diarized utterances are empty.
        alts = ((data.get("results") or {}).get("channels") or [{}])[0].get("alternatives") or [{}]
        flat = (alts[0].get("transcript") or "").strip()
        return _romanize_maybe(flat) if flat else None

    mapping = _map_agent_customer(utterances)
    lines, cur, buf = [], None, []
    for u in utterances:
        label = mapping.get(u.get("speaker", 0), f"Speaker {u.get('speaker', 0)}")
        text = _romanize_maybe((u.get("transcript") or "").strip())
        if not text:
            continue
        if label != cur:
            if buf:
                lines.append(f"{cur}: {' '.join(buf).strip()}")
            cur, buf = label, [text]
        else:
            buf.append(text)
    if buf:
        lines.append(f"{cur}: {' '.join(buf).strip()}")
    return "\n".join(lines).strip() or None
