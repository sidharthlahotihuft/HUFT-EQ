#!/usr/bin/env python3
"""
Quick local test of the transcription pipeline on ONE audio file.

Runs the SAME code the portal uses (transcribe.py) so what you see here is
exactly what would be stored and sent to scoring.

Usage:
    python test_transcribe.py path/to/call.mp3

Needs either:
  - OPENAI_API_KEY set in your .env (the real portal path), OR
  - `pip install faster-whisper` for offline transcription (first run downloads
    a model; set FASTER_WHISPER_MODEL=small or =medium in .env for better Hindi).
"""
import sys
import os
from dotenv import load_dotenv

load_dotenv()  # read .env from the current folder

if len(sys.argv) < 2:
    print("Usage: python test_transcribe.py path/to/audio.mp3")
    sys.exit(1)

audio = sys.argv[1]
if not os.path.exists(audio):
    print(f"File not found: {audio}")
    sys.exit(1)

import transcribe       # the portal's real transcription module
import speaker_label    # the portal's HUFT Agent / Customer labeling module

print(f"File      : {audio}")
print(f"Romanize  : {os.environ.get('WHISPER_ROMANIZE', '1')}  "
      f"Language: {os.environ.get('WHISPER_LANGUAGE') or 'auto'}  "
      f"Temp: {os.environ.get('WHISPER_TEMPERATURE', '0')}")
print("Transcribing (this is the exact portal output)...\n")

try:
    text = transcribe.transcribe(audio)
except transcribe.TranscriptionError as e:
    print("TRANSCRIPTION FAILED:\n", e)
    sys.exit(2)

# Ensure HUFT Agent / Customer labels (same step the deployed app runs). If
# Deepgram already diarized into >=2 speakers this is a no-op; otherwise it
# splits by who's speaking using the LLM (needs ANTHROPIC_API_KEY / OPENAI_API_KEY).
labeled = speaker_label.ensure_labeled(text)
if labeled == text and not speaker_label.is_already_labeled(text):
    print("(!) Speaker labels not applied - set ANTHROPIC_API_KEY (or OPENAI_API_KEY) "
          "so the Agent/Customer split can run.\n")
text = labeled

print("=" * 70)
print(text)
print("=" * 70)

out = os.path.splitext(os.path.basename(audio))[0] + ".transcript.txt"
with open(out, "w", encoding="utf-8") as f:
    f.write(text)
print(f"\nSaved to: {out}")
print(f"Words: {len(text.split())}")
