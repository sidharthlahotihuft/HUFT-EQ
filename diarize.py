#!/usr/bin/env python3
"""
Voice-based speaker diarization for HUFT call recordings.

Produces a line-by-line transcript labeled "HUFT Agent:" / "Customer:" using
real acoustic speaker separation (pyannote) on top of Whisper transcription.
Works on MONO recordings (both speakers on one track) - it separates them by
voice, not by channel.

This does NOT run on Vercel (heavy PyTorch + acoustic models). Run it on your
Mac or a dedicated worker/GPU box, then feed the labeled transcript into the
portal.

  transcription: faster-whisper  (verbatim, timestamps)
  speaker turns: pyannote.audio "speaker-diarization-3.1"
  merge:         each Whisper segment gets the speaker whose turn overlaps it most
  labeling:      the speaker who greets / uses brand + support phrases -> HUFT Agent

Setup (one time):
  pip install -r requirements-diarize.txt
  # 1. Make a free Hugging Face account, create a token (Settings -> Access Tokens)
  # 2. Accept the model terms (click "Agree") on BOTH pages while logged in:
  #      https://huggingface.co/pyannote/speaker-diarization-3.1
  #      https://huggingface.co/pyannote/segmentation-3.0
  # 3. Put the token in your .env as HF_TOKEN=hf_xxx

Usage:
  python diarize.py path/to/call.mp3
  python diarize.py path/to/call.mp3 --speakers 2      # force exactly 2 speakers
"""
import os
import re
import sys
import argparse

from dotenv import load_dotenv

load_dotenv()

# --- Reuse the portal's romanization if available (Devanagari -> Hinglish) ---
try:
    from transcribe import _romanize, _romanize_enabled
except Exception:
    def _romanize_enabled():
        return os.environ.get("WHISPER_ROMANIZE", "1").strip().lower() in ("1", "true", "yes", "on")

    def _romanize(text):
        if not text or not re.search(r"[ऀ-ॿ]", text):
            return text
        try:
            from indic_transliteration import sanscript
            from indic_transliteration.sanscript import transliterate
        except Exception:
            return text
        scheme = getattr(sanscript, os.environ.get("ROMANIZE_SCHEME", "OPTITRANS"), sanscript.OPTITRANS)
        return re.sub(r"[ऀ-ॿ]+",
                      lambda m: transliterate(m.group(0), sanscript.DEVANAGARI, scheme), text)


# Phrases that mark the CS agent side of the conversation (Hinglish + English).
_AGENT_SIGNALS = [
    "head up for tails", "huft", "thank you for calling", "how may i help",
    "how can i help", "how can i assist", "my name is", "speaking", "this side",
    "aapki kya madad", "kaise madad", "main aapki", "team", "sir", "ma'am", "maam",
    "order id", "kindly", "please hold", "line pe rahiye", "ek minute", "confirm karein",
    "aapka order", "dhanyavaad", "have a great day",
]


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def transcribe_segments(audio_path):
    """Whisper transcription with per-segment timestamps."""
    from faster_whisper import WhisperModel
    model_size = os.environ.get("FASTER_WHISPER_MODEL", "small")
    compute_type = os.environ.get("FASTER_WHISPER_COMPUTE", "int8")
    lang = (os.environ.get("WHISPER_LANGUAGE") or "").strip() or None
    model = WhisperModel(model_size, compute_type=compute_type)
    segments, info = model.transcribe(
        audio_path, beam_size=5, temperature=0, language=lang, vad_filter=True,
    )
    segs = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments if s.text.strip()]
    return segs, info.language


def diarize_turns(audio_path, num_speakers=None):
    """pyannote speaker turns: list of (start, end, speaker_label)."""
    from pyannote.audio import Pipeline
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit(
            "No Hugging Face token found. Set HF_TOKEN in your .env (see setup "
            "notes at the top of this file) - pyannote needs it to download the "
            "diarization model."
        )
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    # Nudge onto GPU if available (much faster); CPU works, just slower.
    try:
        import torch
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except Exception:
        pass
    kw = {}
    if num_speakers:
        kw["num_speakers"] = num_speakers
    diarization = pipeline(audio_path, **kw)
    turns = [(turn.start, turn.end, speaker)
             for turn, _, speaker in diarization.itertracks(yield_label=True)]
    return turns


def assign_speakers(segs, turns):
    """Give each Whisper segment the speaker whose turn overlaps it most."""
    for s in segs:
        best, best_ov = None, 0.0
        for (t0, t1, spk) in turns:
            ov = _overlap(s["start"], s["end"], t0, t1)
            if ov > best_ov:
                best, best_ov = spk, ov
        s["speaker"] = best or "SPEAKER_?"
    return segs


def label_agent_customer(segs):
    """Map raw pyannote labels (SPEAKER_00/01/...) to HUFT Agent / Customer.
    Heuristic: whichever speaker uses more agent-signal phrases is the agent;
    ties break to whoever speaks first (agents open the call)."""
    texts, first_seen = {}, {}
    for i, s in enumerate(segs):
        spk = s["speaker"]
        texts.setdefault(spk, []).append(s["text"].lower())
        first_seen.setdefault(spk, i)
    scores = {spk: sum(t.count(sig) for t in joined for sig in _AGENT_SIGNALS)
              for spk, joined in ((k, v) for k, v in texts.items())}
    speakers = list(texts.keys())
    if not speakers:
        return segs, {}
    if len(speakers) == 1:
        mapping = {speakers[0]: "HUFT Agent"}
    else:
        # Highest agent-score is the agent; break ties by who spoke first.
        agent = max(speakers, key=lambda spk: (scores.get(spk, 0), -first_seen.get(spk, 0)))
        mapping = {}
        others = 0
        for spk in speakers:
            if spk == agent:
                mapping[spk] = "HUFT Agent"
            else:
                mapping[spk] = "Customer" if len(speakers) == 2 else f"Customer {others + 1}"
                others += 1
    for s in segs:
        s["label"] = mapping.get(s["speaker"], s["speaker"])
    return segs, mapping


def build_transcript(segs):
    """Merge consecutive same-speaker segments into labeled lines."""
    lines, cur_label, buf = [], None, []
    for s in segs:
        text = _romanize(s["text"]) if _romanize_enabled() else s["text"]
        if s["label"] != cur_label:
            if buf:
                lines.append(f"{cur_label}: {' '.join(buf).strip()}")
            cur_label, buf = s["label"], [text]
        else:
            buf.append(text)
    if buf:
        lines.append(f"{cur_label}: {' '.join(buf).strip()}")
    return "\n".join(lines)


def diarized_transcript(audio_path, num_speakers=2):
    segs, lang = transcribe_segments(audio_path)
    if not segs:
        raise SystemExit("No speech found in the audio.")
    turns = diarize_turns(audio_path, num_speakers=num_speakers)
    segs = assign_speakers(segs, turns)
    segs, mapping = label_agent_customer(segs)
    return build_transcript(segs), lang, mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--speakers", type=int, default=2,
                    help="expected number of speakers (default 2; use 0 to auto-detect)")
    args = ap.parse_args()
    if not os.path.exists(args.audio):
        raise SystemExit(f"File not found: {args.audio}")

    num = args.speakers or None
    print(f"Transcribing + diarizing {args.audio} (speakers={num or 'auto'})...\n", flush=True)
    transcript, lang, mapping = diarized_transcript(args.audio, num_speakers=num)

    print(f"Detected language: {lang}")
    print(f"Speaker mapping  : {mapping}\n")
    print("=" * 72)
    print(transcript)
    print("=" * 72)

    out = os.path.splitext(os.path.basename(args.audio))[0] + ".diarized.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(transcript)
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
