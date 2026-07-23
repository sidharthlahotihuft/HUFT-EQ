# HUFT EQ — CS Call Quality Portal

A web portal for HUFT Customer Care: upload a call recording, Whisper transcribes it, and it's automatically scored against the 100-point Voice rubric. Includes a per-call report card and a team overview dashboard.

Built to deploy on **Vercel** (Python/Flask runtime) + **Supabase** (Postgres for data, Storage for audio files). Design follows the existing Report Card / Scoring Matrix documents (navy/blue headers, green/yellow/red banded score cells, criterion + evidence tables).

## How it works

1. **Upload** — the browser uploads the audio file *directly* to Supabase Storage using a short-lived signed URL (Vercel Functions cap request bodies at 4.5MB, which most recordings exceed, so audio never passes through our own backend).
2. **Transcribe** — Whisper (OpenAI's Whisper API) turns the recording into text.
3. **Score** — the transcript is scored against all 28 line items (grouped into the parameters below), summing to 100 points. Scoring uses Claude or GPT if you provide an API key; otherwise a keyword-based heuristic gives a draft score so the app still works with zero AI keys.
4. **Report Card** — per-call page with overall score/grade, coaching recommendations, a full criterion table with evidence, and the transcript. Every score and evidence note is editable — a QA reviewer can correct the AI's first pass and save.
5. **Overview** — ranks all scored calls, shows team averages per parameter (chart + table), and pass-rate stats.

Transcription and scoring run as two short, separate steps triggered by the browser in sequence (rather than one background job) — Vercel functions are stateless, so this keeps each request well within its time limit and makes the pipeline resumable if a page is reloaded mid-process.

### The rubric (100 pts)

| Parameter | Points | | Parameter | Points |
|---|---|---|---|---|
| WOW | 10 | | Interruption | 3 |
| Greetings | 3 | | Language Style Matching | 3 |
| Identify The Concern | 3 | | Active Listening | 3 |
| Empathy/Apology | 3 | | Imposed Behavior | 2 |
| Verify | 3 | | Command Over Tone | 3 |
| Probing Questions | 3 | | Command Over Language | 3 |
| Paraphrasing | 3 | | Procedure | 3 |
| Hold Procedure | 3 | | Navigation | 3 |
| Unhold Procedure | 3 | | Tagging & Notes | 3 |
| TAT | 3 | | First Call Resolution | 3 |
| Resolution | 3 | | Profiling Pet/Human | 3 |
| Summarization | 3 | | Unprofessional (deduction) | 8 |
| Additional Assistance | 3 | | Incomplete Information | 3 |
| Closure | 3 | | Misleading/Incorrect Info (deduction) | 8 |

Grade bands: **A+** 90–100 · **A** 80–89 · **B** 70–79 · **C** 60–69 · **D** 50–59 · **F** below 50.

## Set up Supabase

1. Create a project at [supabase.com](https://supabase.com).
2. In the SQL Editor, run **`supabase/schema.sql`** from this repo — it creates the `calls` table (RLS enabled, no public policies — only the service role key can touch it) and a private `call-recordings` Storage bucket.
3. In Project Settings → API, copy the **Project URL**, **anon/public key**, and **service_role key**.

## Set up locally

Requires Python 3.10+.

```bash
cd huft-call-scoring-portal
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:
- `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY` (required).
- `OPENAI_API_KEY` for Whisper transcription (required for a real deployment — local Whisper models aren't practical in serverless; for local-only dev without a key you can `pip install faster-whisper` instead).
- `ANTHROPIC_API_KEY` (or reuse `OPENAI_API_KEY` for scoring) so calls are scored by an LLM instead of the keyword heuristic. Optional — every score is editable by hand either way.

Run it:

```bash
python app.py
```

Open **http://localhost:5001**.

## Deploy to Vercel

1. Push this repo to GitHub/GitLab/Bitbucket.
2. In Vercel, **Import Project** and select the repo. Vercel auto-detects the Flask app (`app.py` at the repo root) — no build config needed.
3. In Project Settings → Environment Variables, add everything from `.env.example` (`SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_AUDIO_BUCKET`, `OPENAI_API_KEY`, and optionally `ANTHROPIC_API_KEY`).
4. Deploy. `vercel.json` sets a 120s max duration for the function (Whisper + LLM scoring calls easily fit within that); raise it if you routinely score long calls and are on a plan that allows it.
5. This app has no login/auth built in — it's meant for internal use. Consider adding [Vercel Deployment Protection](https://vercel.com/docs/deployment-protection) or basic auth in front of it before sharing the URL widely.

## Notes & limitations

- Some criteria (Tagging & Notes in the CRM, true tone/prosody, factual accuracy of policy statements) can't be fully verified from a transcript alone — the AI scorer is told this and asked for its best evidence-based judgment, and the heuristic fallback flags these clearly as needing manual review. That's why every score on the report card is editable: treat the first pass as an AI-assisted draft, not a final verdict.
- Audio files live in Supabase Storage (private bucket, accessed only via signed URLs and the service role key); call data lives in Supabase Postgres. Nothing is stored on Vercel itself, which is required since serverless functions have no persistent disk.
- To reset all data: truncate the `calls` table and empty the `call-recordings` bucket in the Supabase dashboard.

## Project structure

```
app.py               Flask routes (upload-url issuing, transcribe/score steps, pages)
scoring.py            Rubric definition + heuristic & LLM scoring
transcribe.py          Whisper transcription (OpenAI API, with optional local fallback)
storage.py              Supabase Postgres persistence (calls table)
audio_storage.py         Supabase Storage helpers (signed upload URLs, server-side download)
supabase/schema.sql       Run once in the Supabase SQL editor
vercel.json                Function config (maxDuration)
templates/                  Jinja2 pages (upload, report card, overview)
public/static/css/style.css  Brand styling (served from Vercel's CDN in production)
```
