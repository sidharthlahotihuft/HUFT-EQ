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

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# A run of 6+ digits, allowing spaces/dashes between them (e.g. "9876 543 210").
_DIGIT_RUN = re.compile(r"(?<!\w)(\+?\d[\d\s-]{4,}\d)(?!\w)")


def _classify_digits(raw):
    digits = re.sub(r"\D", "", raw)
    n = len(digits)
    if n >= 12 and n <= 19:
        return "[card/account]"      # card or bank account number
    if n == 10 or (11 <= n <= 12 and (digits.startswith("0") or digits.startswith("91"))):
        return "[phone]"             # Indian mobile, with/without country code
    if n == 6:
        return "[pincode]"           # Indian PIN code
    if n >= 6:
        return "[number]"            # other long number (kept generic)
    return raw                        # short — leave (e.g. small quantities)


def mask_pii(text):
    """Return the transcript with emails and sensitive number sequences masked.
    Speaker labels ('HUFT Agent:' / 'Customer:') and ordinary words are untouched."""
    if not text:
        return text
    if os.environ.get("MASK_PII", "1").strip().lower() not in ("1", "true", "yes", "on"):
        return text
    text = _EMAIL.sub("[email]", text)
    text = _DIGIT_RUN.sub(lambda m: _classify_digits(m.group(1)), text)
    return text
