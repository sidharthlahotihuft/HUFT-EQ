"""
PII masking for HUFT call transcripts.

Runs on the transcript immediately after transcription and BEFORE it is sent to
any downstream AI (speaker labeling, scoring) or stored — so sensitive customer
data (phone numbers, emails, card / bank / account numbers, PIN codes) never
reaches the scoring model or the manager view in the clear.

Best-effort and regex-based, tuned for Indian customer-care transcripts where
numbers usually appear as digits (Deepgram/Whisper transcribe spoken digits to
numerals). It is deliberately conservative about what it treats as an order ID
vs. a phone number. Toggle with MASK_PII=0.

Note: the transcription engine (Deepgram/Whisper) unavoidably sees the raw
audio; masking protects every step after that. For stricter needs, also enable
Deepgram's own redaction (redact=pci,pii,numbers) at the API layer.
"""
import os
import re

MASK = "****"
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# A run of 6+ digits, allowing spaces/dashes between them (e.g. "9876 543 210").
_DIGIT_RUN = re.compile(r"(?<!\w)(\+?\d[\d\s-]{4,}\d)(?!\w)")


def _classify_digits(raw):
    digits = re.sub(r"\D", "", raw)
    n = len(digits)
    if n >= 6:
        return MASK                   # phone / card / account / PIN / long number
    return raw                         # short (2-5 digits) — quantities, years, etc.


def mask_pii(text):
    """Return the transcript with emails and sensitive number sequences masked as
    ****. Speaker labels and ordinary words are untouched. (Names/addresses are
    additionally redacted by the LLM cleanup pass in speaker_label.py.)"""
    if not text:
        return text
    if os.environ.get("MASK_PII", "1").strip().lower() not in ("1", "true", "yes", "on"):
        return text
    text = _EMAIL.sub(MASK, text)
    text = _DIGIT_RUN.sub(lambda m: _classify_digits(m.group(1)), text)
    return text
