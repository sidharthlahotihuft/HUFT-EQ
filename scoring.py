"""
Scoring engine for the CS Call Quality Portal.

Defines the 28-line-item rubric (grouped into 26 named parameters, with
"Hold/Unhold & TAT" split into its 3 constituent sub-checks) that sums to 100
points, exactly matching the scoring matrix supplied for the Voice channel.

Two scoring paths are supported:
  1. LLM scoring (preferred) - sends the transcript + rubric to Claude
     (ANTHROPIC_API_KEY) or, if that's not configured, OpenAI (OPENAI_API_KEY)
     and asks for a structured per-criterion score + evidence quote.
  2. Heuristic fallback - simple keyword/pattern matching so the portal is
     still fully functional with zero API keys configured (useful for demos
     or offline use). Heuristic scores are clearly labeled as a draft and
     every score is editable by a human reviewer on the report card page.
"""
import json
import os
import re

# ---------------------------------------------------------------------------
# 1. The rubric
# ---------------------------------------------------------------------------
# Each item: key, label, group (for display), max points, and a short
# description of what "full marks" looks like (used in the LLM prompt).
PARAMETERS = [
    dict(key="wow", label="WOW Factor", group="Overall Impression", max=10,
         desc="Call includes a genuine, memorable moment of delight beyond standard service "
              "(e.g. going out of the way for the pet/customer, a thoughtful unscripted touch)."),
    dict(key="greetings", label="Greetings", group="Opening", max=3,
         desc="Warm, branded greeting at the start of the call (and closing), sets a friendly tone."),
    dict(key="identify_concern", label="Identify The Concern", group="Opening", max=3,
         desc="Agent clearly and quickly identifies the customer's actual concern before acting."),
    dict(key="empathy_apology", label="Empathy/Apology", group="Opening", max=3,
         desc="Sincere, specific empathy or apology offered before moving to policy/procedure."),
    dict(key="verify", label="Verify", group="Discovery", max=3,
         desc="Agent verifies customer/order/account identity per process before taking action."),
    dict(key="probing_questions", label="Probing Questions", group="Discovery", max=3,
         desc="Agent asks relevant open-ended/clarifying questions to fully understand the issue."),
    dict(key="paraphrasing", label="Paraphrasing", group="Discovery", max=3,
         desc="Agent paraphrases/confirms understanding back to the customer."),
    dict(key="hold_procedure", label="Hold Procedure", group="Hold/Unhold & TAT", max=3,
         desc="Agent asks permission clearly before placing the customer on hold."),
    dict(key="unhold_procedure", label="Unhold Procedure", group="Hold/Unhold & TAT", max=3,
         desc="Agent thanks the customer for waiting / holding when returning to the line."),
    dict(key="tat", label="TAT (Turnaround Time)", group="Hold/Unhold & TAT", max=3,
         desc="Agent communicates a clear, accurate turnaround time / resolution timeline."),
    dict(key="resolution", label="Resolution", group="Resolution", max=3,
         desc="Issue is actually resolved or a clear resolution path is set in motion on the call."),
    dict(key="summarization", label="Summarization", group="Resolution", max=3,
         desc="Agent summarizes what was discussed/agreed before closing."),
    dict(key="additional_assistance", label="Additional Assistance", group="Resolution", max=3,
         desc="Agent proactively asks if there's anything else they can help with."),
    dict(key="closure", label="Closure", group="Resolution", max=3,
         desc="Call ends with a warm, professional, branded closing."),
    dict(key="interruption", label="Interruption", group="Conduct", max=3,
         desc="Agent does not talk over or cut off the customer."),
    dict(key="language_style_matching", label="Language Style Matching", group="Conduct", max=3,
         desc="Agent mirrors the customer's language/register (formality, code-switching) appropriately."),
    dict(key="active_listening", label="Active Listening", group="Conduct", max=3,
         desc="Agent demonstrates active listening (acknowledgements, no repeated questions, relevant follow-ups)."),
    dict(key="imposed_behavior", label="Imposed Behavior", group="Conduct", max=2,
         desc="Agent does not impose opinions/pressure the customer into a decision."),
    dict(key="command_tone", label="Command Over Tone", group="Delivery", max=3,
         desc="Tone is confident, warm, and appropriately modulated throughout."),
    dict(key="command_language", label="Command Over Language", group="Delivery", max=3,
         desc="Clear grammar, vocabulary and articulation; easy to understand."),
    dict(key="procedure", label="Procedure", group="Process Adherence", max=3,
         desc="Agent follows the correct SOP/process steps for this call type."),
    dict(key="navigation", label="Navigation", group="Process Adherence", max=3,
         desc="Agent navigates systems/tools efficiently without excessive delay or repetition."),
    dict(key="tagging_notes", label="Tagging & Notes", group="Process Adherence", max=3,
         desc="Call is correctly tagged and notes are logged (verify in CRM; not always inferable from audio alone)."),
    dict(key="first_call_resolution", label="First Call Resolution", group="Process Adherence", max=3,
         desc="Issue is resolved within this single call with no unnecessary follow-up required."),
    dict(key="profiling", label="Profiling Pet/Human", group="Process Adherence", max=3,
         desc="Agent references/updates relevant pet and customer profile details during the call."),
    dict(key="unprofessional", label="Unprofessional", group="Red Flags (deduction)", max=8,
         desc="Deduction item. Full marks unless the agent is rude, dismissive, sarcastic or unprofessional."),
    dict(key="incomplete_info", label="Incomplete Information", group="Red Flags (deduction)", max=3,
         desc="Deduction item. Full marks unless the agent gives incomplete information the customer needed."),
    dict(key="misleading_info", label="Misleading / Incorrect Information", group="Red Flags (deduction)", max=8,
         desc="Deduction item. Full marks unless the agent gives misleading or factually incorrect information."),
]

MAX_TOTAL = sum(p["max"] for p in PARAMETERS)  # == 100
PARAM_BY_KEY = {p["key"]: p for p in PARAMETERS}
GROUPS = list(dict.fromkeys(p["group"] for p in PARAMETERS))  # preserve order


def grade_for(total, max_total=MAX_TOTAL):
    pct = 100.0 * total / max_total if max_total else 0
    if pct >= 90:
        return "A+", pct
    if pct >= 80:
        return "A", pct
    if pct >= 70:
        return "B", pct
    if pct >= 60:
        return "C", pct
    if pct >= 50:
        return "D", pct
    return "F", pct


def band_for_pct(pct):
    """green / yellow / red banding used for cell shading, matching the report-card style."""
    if pct >= 80:
        return "good"
    if pct >= 50:
        return "warn"
    return "bad"


def empty_scores():
    return {p["key"]: {"score": p["max"], "evidence": "Not yet scored.", "max": p["max"]} for p in PARAMETERS}


def totals_from_scores(scores: dict):
    total = 0.0
    for p in PARAMETERS:
        v = scores.get(p["key"], {})
        s = v.get("score", 0) or 0
        try:
            s = float(s)
        except (TypeError, ValueError):
            s = 0.0
        s = max(0.0, min(float(p["max"]), s))
        total += s
    return total


# ---------------------------------------------------------------------------
# 2. LLM scoring (preferred path)
# ---------------------------------------------------------------------------
def _rubric_prompt(transcript, meta):
    lines = []
    for p in PARAMETERS:
        lines.append(f'- key="{p["key"]}" | "{p["label"]}" (max {p["max"]} pts): {p["desc"]}')
    rubric_text = "\n".join(lines)

    meta_text = "\n".join(f"{k}: {v}" for k, v in meta.items() if v)

    return f"""You are a senior QA analyst scoring a customer service phone call transcript for HUFT
(Heads Up For Tails), a pet care brand. Score the call strictly against the rubric below.
Each item must be scored between 0 and its max. For the three "Red Flags (deduction)" items
(unprofessional, incomplete_info, misleading_info), start from full marks and deduct only if you
find real evidence of that problem in the transcript.

Call metadata:
{meta_text or "(none provided)"}

Rubric:
{rubric_text}

Transcript:
\"\"\"
{transcript}
\"\"\"

Return ONLY a JSON object (no markdown, no commentary) shaped exactly like this, with one entry
per rubric key:
{{
  "wow": {{"score": <number 0-10>, "evidence": "<one or two sentence justification with a quote if possible>"}},
  "greetings": {{"score": <number 0-3>, "evidence": "..."}},
  ... (one entry for every key listed in the rubric above)
}}
"""


def _extract_json(text):
    text = text.strip()
    # strip markdown code fences if present
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(match.group(0))


def _score_with_anthropic(transcript, meta):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=4000,
        temperature=0,
        messages=[{"role": "user", "content": _rubric_prompt(transcript, meta)}],
    )
    text = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")
    return _extract_json(text)


def _score_with_openai(transcript, meta):
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
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": _rubric_prompt(transcript, meta)}],
    )
    text = resp.choices[0].message.content
    return _extract_json(text)


def llm_score(transcript, meta=None):
    """Try Claude first, then OpenAI. Returns (scores_dict, method_label) or (None, None)."""
    meta = meta or {}
    for fn, label in ((_score_with_anthropic, "Claude"), (_score_with_openai, "OpenAI GPT")):
        try:
            raw = fn(transcript, meta)
        except Exception as e:  # noqa: BLE001 - want to fall back on any provider error
            raw = None
            print(f"[scoring] {label} scoring failed: {e}")
        if raw:
            scores = {}
            for p in PARAMETERS:
                item = raw.get(p["key"]) or {}
                score = item.get("score", p["max"])
                try:
                    score = float(score)
                except (TypeError, ValueError):
                    score = p["max"]
                score = max(0.0, min(float(p["max"]), score))
                scores[p["key"]] = {
                    "score": score,
                    "evidence": item.get("evidence", "").strip() or "No evidence text returned.",
                    "max": p["max"],
                }
            return scores, f"AI-scored ({label})"
    return None, None


# ---------------------------------------------------------------------------
# 3. Heuristic fallback (no API key required)
# ---------------------------------------------------------------------------
_PATTERNS = {
    "greetings": [r"\bwelcome to\b", r"\bthank you for calling\b", r"\bgood (morning|afternoon|evening)\b",
                  r"\bhow (can|may) i help\b"],
    "identify_concern": [r"\bwhat.{0,15}(issue|problem|concern|help)\b", r"\btell me (more )?about\b",
                          r"\bhow can i assist\b"],
    "empathy_apology": [r"\bsorry\b", r"\bapolog", r"\bi understand\b", r"\bi can imagine\b"],
    "verify": [r"\bcan you confirm\b", r"\bverify\b", r"\bmay i (know|have) your\b", r"\border id\b",
               r"\bregistered (email|mobile|number)\b"],
    "paraphrasing": [r"\bso (what|if) you.{0,10}(saying|mean)\b", r"\bjust to confirm\b",
                      r"\bif i understand correctly\b"],
    "hold_procedure": [r"\bmay i (please )?place you (on|call on) hold\b", r"\bcan i put you on hold\b",
                        r"\bplease hold\b"],
    "unhold_procedure": [r"\bthank you for (holding|waiting)\b", r"\bthanks for (holding|waiting)\b",
                          r"\bsorry to keep you waiting\b", r"\bappreciate your patience\b"],
    "tat": [r"\b\d{1,3}\s?(-|to)?\s?\d{0,3}\s?(hour|hr|day)s?\b", r"\bwithin \d+\b"],
    "resolution": [r"\bresolved\b", r"\bhas been (processed|initiated)\b", r"\bi have (processed|initiated)\b",
                   r"\breplacement will be\b", r"\brefund (has been|is) initiated\b"],
    "summarization": [r"\bto summarize\b", r"\bjust to recap\b", r"\bso to confirm what we discussed\b"],
    "additional_assistance": [r"\banything else\b", r"\bis there (anything|something) else\b"],
    "closure": [r"\bhave a (great|nice|good) day\b", r"\bthank you for calling\b", r"\bcheers\b",
                r"\btake care\b"],
    "active_listening": [r"\bi see\b", r"\bgot it\b", r"\bunderstood\b", r"\bnoted\b", r"\bi hear you\b"],
}

_NEGATIVE_PATTERNS = {
    "unprofessional": [r"\bwhatever\b", r"\bnot my problem\b", r"\bi don'?t care\b", r"\bcalm down\b",
                        r"\bthat'?s not my (job|department)\b"],
    "incomplete_info": [r"\bi don'?t (know|have that)\b", r"\bnot sure\b(?!.{0,40}(check|find out|confirm))"],
    "misleading_info": [],  # left for LLM / human review - heuristic can't reliably detect factual accuracy
}


def _hit_count(patterns, text):
    return sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))


def heuristic_score(transcript, meta=None):
    text = transcript or ""
    scores = {}
    for p in PARAMETERS:
        key, mx = p["key"], p["max"]

        if key in _NEGATIVE_PATTERNS:
            hits = _hit_count(_NEGATIVE_PATTERNS[key], text)
            score = max(0, mx - hits * (mx / 2))
            evidence = (f"{hits} potential red-flag phrase(s) matched by keyword scan."
                        if hits else "No red-flag language detected by keyword scan (heuristic default — verify manually).")
            scores[key] = {"score": score, "evidence": evidence, "max": mx}
            continue

        if key in _PATTERNS:
            hits = _hit_count(_PATTERNS[key], text)
            if hits >= 2:
                score = mx
            elif hits == 1:
                score = round(mx * 0.7, 1)
            else:
                score = round(mx * 0.4, 1)
            evidence = (f"{hits} matching phrase(s) found by keyword scan."
                        if hits else "No matching phrases found by keyword scan — likely missing or needs manual review.")
            scores[key] = {"score": score, "evidence": evidence, "max": mx}
            continue

        # Items that really need audio/tone/CRM context and can't be reliably
        # inferred from text alone: default to a neutral 70% with a clear flag.
        score = round(mx * 0.7, 1)
        scores[key] = {
            "score": score,
            "evidence": "Heuristic default (no reliable text-only signal) — please review manually or connect an AI scoring key.",
            "max": mx,
        }
    return scores


def score_transcript(transcript, meta=None):
    """Main entry point used by the app. Returns (scores, method_label)."""
    meta = meta or {}
    scores, method = llm_score(transcript, meta)
    if scores:
        return scores, method
    return heuristic_score(transcript, meta), "Heuristic draft (no AI scoring key configured)"


def coaching_recommendations(scores, top_n=3):
    """Pick the weakest-percentage criteria and return short coaching notes."""
    ranked = []
    for p in PARAMETERS:
        v = scores.get(p["key"], {})
        mx = v.get("max", p["max"]) or p["max"]
        s = v.get("score", mx)
        pct = 100.0 * s / mx if mx else 100
        ranked.append((pct, p, v))
    ranked.sort(key=lambda t: t[0])
    out = []
    for pct, p, v in ranked[:top_n]:
        if pct >= 80:
            continue
        out.append({
            "label": p["label"],
            "pct": round(pct, 0),
            "tip": f'Coach on "{p["label"]}": {p["desc"]}',
            "evidence": v.get("evidence", ""),
        })
    return out
