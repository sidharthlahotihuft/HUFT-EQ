"""
Scoring engine for the HUFT Care Intelligence Call-Care Coach.

Implements the model defined in the "HUFT Care Intelligence" master brief
(v2.0), Sections 4 and 8-13:

  * Six care dimensions scored 1-5 (or N/A), weighted:
        Customer understood 20%, Pet acknowledged 15%, Empathy 15%,
        Ownership 20%, Resolution 20%, Warmth 10%.
  * Overall = weighted average over APPLICABLE dimensions only. A dimension may
    be Not Applicable (excluded from the denominator) - the agent is never
    punished for, e.g., not mentioning a pet when no pet is relevant.
  * Insufficient-evidence guard: if fewer than 3 dimensions are applicable with
    sufficient evidence (or transcript confidence is too low), no overall score
    is produced - the call is routed for human review instead of guessing.
  * Critical / high-severity caps: a confirmed critical failure (unsafe medical
    or product guidance, deception, abusive conduct, privacy breach, refusal to
    escalate a serious pet-safety issue) caps the displayed score at 2.0; a
    high-severity failure caps it at 3.0. Safety and deception must never be
    averaged away by otherwise-polite behaviour.
  * Rich post-call output (Section 10): one-line assessment, what went well,
    biggest missed moment (with timestamp), a "try instead" sentence, resolution
    status, commitments, risk/alerts, policy check, and one coaching tag.

Two scoring paths:
  1. LLM scoring (preferred) - Claude (ANTHROPIC_API_KEY) or OpenAI
     (OPENAI_API_KEY), prompted with the HUFT judgment layer and asked for a
     structured JSON evaluation. Every score is editable by a human reviewer.
  2. Fallback - when no LLM key is configured, the call is marked
     "insufficient_evidence / human review required" rather than fabricating
     keyword-based judgments (the brief explicitly forbids confident guesses).
"""
import json
import os
import re

# ---------------------------------------------------------------------------
# 1. The six care dimensions (Brief Section 4, Tables 8, 10, 12)
# ---------------------------------------------------------------------------
DIMENSIONS = [
    dict(
        key="customer_understood", label="Customer understood", weight=20,
        question="Did the agent correctly understand and reflect the real problem?",
        anchor5="Reflects both the request and the underlying concern; minimal repetition.",
        anchor3="Understands the task but misses the deeper concern.",
        anchor1="Misunderstands, interrupts, or makes the customer repeat without need.",
        positive="Accurate paraphrase; relevant questions; no needless repetition.",
        reducing="Wrong issue, premature solution, repeated questions already answered.",
        na_guard="Do not require an explicit paraphrase when a simple request is correctly actioned.",
    ),
    dict(
        key="pet_acknowledged", label="Pet acknowledged", weight=15,
        question="When relevant, did the agent recognise the pet's comfort, safety, wellbeing or emotional significance?",
        anchor5="Naturally responds to the pet's relevant comfort, safety, health or significance.",
        anchor3="Mentions the pet generically or late.",
        anchor1="Ignores a clear pet concern, trivialises it or gives unsafe guidance.",
        positive="Responds to stated comfort, health, safety, behaviour, grief or joy.",
        reducing="Ignores, minimises or contradicts a relevant pet concern.",
        na_guard="N/A only when no pet context is relevant. Pet-name use alone is not evidence.",
    ),
    dict(
        key="empathy", label="Empathy", weight=15,
        question="Did the response specifically acknowledge inconvenience, disappointment, worry or grief?",
        anchor5="Specific, timely and proportionate.",
        anchor3="Generic acknowledgment such as “I understand.”",
        anchor1="No acknowledgment, blame, defensiveness or emotional mismatch.",
        positive="Specific and proportionate acknowledgment at the right moment.",
        reducing="Generic script, emotional mismatch, blame or defensiveness.",
        na_guard="N/A for wrong number/spam; brevity is not lack of empathy.",
    ),
    dict(
        key="ownership", label="Ownership", weight=20,
        question="Did the agent take responsibility for moving the issue forward?",
        anchor5="Clear next steps, owner and timeline; reduces customer effort.",
        anchor3="Some help, but hand-off or follow-up is vague.",
        anchor1="Passes responsibility, refuses to help or leaves no path forward.",
        positive="Clear next step, owner, dependency and timeline; warm hand-off.",
        reducing="Deflection, avoidable transfer, vague callback or customer asked to chase.",
        na_guard="Do not penalise limits outside the agent's authority when a useful path is created.",
    ),
    dict(
        key="resolution", label="Resolution", weight=20,
        question="Was the information accurate, complete, clear and policy-aligned?",
        anchor5="Accurate, complete, concise and confirmed.",
        anchor3="Mostly correct but timeline/conditions unclear.",
        anchor1="Wrong policy, unsupported promise, no resolution or avoidable confusion.",
        positive="Correct, complete, concise, policy-grounded outcome and confirmation.",
        reducing="Wrong facts, omitted conditions, unsupported certainty, unresolved confusion.",
        na_guard="Mark cannot-verify — not low — when the policy source is unavailable.",
    ),
    dict(
        key="warmth", label="Warmth", weight=10,
        question="Did the customer feel cared for rather than processed?",
        anchor5="Attentive, respectful and naturally human throughout.",
        anchor3="Polite but transactional.",
        anchor1="Cold, rushed, sarcastic, dismissive or argumentative.",
        positive="Respectful attention, responsive language, suitable pace and human close.",
        reducing="Cold processing, impatience, sarcasm, argument or inappropriate cheerfulness.",
        na_guard="Do not infer coldness from accent, low pitch, pauses, grammar or concise speech.",
    ),
]

DIM_BY_KEY = {d["key"]: d for d in DIMENSIONS}
TOTAL_WEIGHT = sum(d["weight"] for d in DIMENSIONS)  # 100

# Scenario postures the AI classifies before applying expectations (Section 6).
SCENARIOS = [
    "Delivery delay", "Return / size issue", "Food refusal / palatability",
    "Possible adverse reaction", "Injury / illness", "Grooming complaint",
    "Lost pet", "Pet death / grief", "Billing / refund",
    "Happy feedback / birthday", "Abusive caller", "Product recommendation",
    "Escalation request", "Repeat complaint", "Wrong number / spam / administrative",
    "Other",
]

# If fewer than this many dimensions are applicable+evidenced, don't score.
MIN_APPLICABLE = 3
MAX_SCORE = 5.0

# --- Rating bands (Brief Table 9) ---------------------------------------------
def rating_band_for(display_score, critical=False, high_severity=False, status="scored"):
    if status == "insufficient_evidence" or display_score is None:
        return "Insufficient evidence"
    if critical:
        return "Critical failure"
    if high_severity:
        return "High-severity issue"
    if display_score >= 4.5:
        return "Excellent"
    if display_score >= 3.5:
        return "Strong"
    if display_score >= 2.5:
        return "Adequate"
    if display_score >= 1.5:
        return "Weak"
    return "Unacceptable"


def band_for_score(score):
    """green / yellow / red banding for a 1-5 dimension score (cell shading)."""
    if score is None:
        return "na"
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "na"
    if s >= 4:
        return "good"
    if s >= 2.5:
        return "warn"
    return "bad"


# ---------------------------------------------------------------------------
# 2. Overall score computation (Brief Table 13)
# ---------------------------------------------------------------------------
def compute_overall(dimensions, transcript_confidence=1.0,
                    confidence_threshold=0.35,
                    critical_failure=False, high_severity_failure=False):
    """
    dimensions: {key: {"score": 1-5|None, "applicable": bool,
                        "evidence_sufficient": bool, ...}}
    Returns (overall_score|None, display_score|None, score_status).
    Never lowers a score merely because transcript confidence is weak - in that
    case the finding is withheld and the call is routed for human review.
    """
    applicable = []
    for d in DIMENSIONS:
        v = dimensions.get(d["key"], {})
        if v.get("applicable") and v.get("evidence_sufficient") and v.get("score") is not None:
            applicable.append((d, v))

    if len(applicable) < MIN_APPLICABLE or transcript_confidence < confidence_threshold:
        return None, None, "insufficient_evidence"

    num = sum(float(v["score"]) * d["weight"] for d, v in applicable)
    den = sum(d["weight"] for d, v in applicable)
    overall = round(num / den, 2) if den else None

    display = overall
    if overall is not None:
        if critical_failure:
            display = min(overall, 2.0)
        elif high_severity_failure:
            display = min(overall, 3.0)
    return overall, display, "scored"


def recompute(result):
    """Recompute overall/display/band in place after a human edits dimension
    scores or applicability on the report card. Returns the updated result."""
    dims = result.get("dimensions", {})
    crit = bool(result.get("critical_failure"))
    high = bool(result.get("high_severity_failure"))
    conf = result.get("transcript_confidence", 1.0)
    overall, display, status = compute_overall(
        dims, transcript_confidence=conf,
        critical_failure=crit, high_severity_failure=high)
    result["overall_score"] = overall
    result["display_score"] = display
    result["score_status"] = status
    result["rating_band"] = rating_band_for(display, crit, high, status)
    return result


def empty_result():
    """Placeholder result before scoring has run."""
    dims = {}
    for d in DIMENSIONS:
        dims[d["key"]] = {
            "score": None, "applicable": True, "evidence_sufficient": False,
            "confidence": None, "evidence": "Not yet scored.", "weight": d["weight"],
        }
    return {
        "one_line_assessment": "", "scenario": "", "dimensions": dims,
        "overall_score": None, "display_score": None, "score_status": "not_scored",
        "rating_band": "Not scored", "what_went_well": [],
        "biggest_missed_moment": {"summary": "", "timestamp": ""}, "try_instead": "",
        "resolution_status": "", "commitments": [], "risks_alerts": [],
        "policy_check": {"status": "", "note": ""}, "coaching_tag": "",
        "failure_attribution": "", "critical_failure": False,
        "high_severity_failure": False, "transcript_confidence": 1.0,
    }


# ---------------------------------------------------------------------------
# 3. LLM scoring (preferred path)
# ---------------------------------------------------------------------------
def _dimension_block():
    out = []
    for d in DIMENSIONS:
        out.append(
            f'- key="{d["key"]}" | {d["label"]} (weight {d["weight"]}%)\n'
            f'    Core question: {d["question"]}\n'
            f'    5 = {d["anchor5"]}\n'
            f'    3 = {d["anchor3"]}\n'
            f'    1 = {d["anchor1"]}\n'
            f'    Positive evidence: {d["positive"]}\n'
            f'    Score-reducing: {d["reducing"]}\n'
            f'    N/A guard: {d["na_guard"]}'
        )
    return "\n".join(out)


def _prompt(transcript, meta):
    meta_text = "\n".join(f"{k}: {v}" for k, v in meta.items() if v) or "(none provided)"
    scenarios = ", ".join(SCENARIOS)
    return f"""You are the HUFT Care Intelligence Coach evaluating a customer-care phone call for
Head Up For Tails (HUFT), an Indian pet-care brand. Pets are family: every call is about a
person and an animal they love, not merely a ticket. Judge care, accuracy and clarity across
English, Hindi and Hinglish equally.

Follow these rules strictly:
- Score SIX dimensions from 1 to 5, OR mark a dimension "applicable": false when the behaviour
  was not reasonably called for (it is then excluded from the overall score). Never punish an
  agent for not mentioning a pet when no pet is relevant.
- A score of 5 requires affirmative evidence, not merely the absence of failure. A 3 is
  competent/adequate, not a punishment.
- Cite transcript evidence (quote or paraphrase, with an approximate timestamp) for every score.
- Do NOT infer emotion as fact from tone alone; say "the customer appears..." unless they name it.
- Never score accent, gender, region, grammar, code-switching or vocabulary. Brevity is not
  coldness or lack of empathy.
- Do NOT invent policy, product or timeline facts. If a claim's correctness depends on a HUFT
  policy you were not given, set policy_check.status to "cannot_verify" and do not mark Resolution
  low solely for that.
- Do NOT lower any score merely because transcript quality is poor; instead set that dimension's
  "evidence_sufficient": false.
- Set "critical_failure": true ONLY for confirmed unsafe medical/product guidance, deception,
  abusive conduct, privacy breach, or refusal to escalate a serious pet-safety issue.
  Set "high_severity_failure": true for e.g. ignored repeated escalation, unsupported financial
  promise, unresolved high-emotion case.

First classify the scenario (one of: {scenarios}), then apply the appropriate care posture.

Call metadata:
{meta_text}

The six dimensions, with anchors and fairness guards:
{_dimension_block()}

Transcript (may include "HUFT Agent:" / "Customer:" speaker labels and timestamps):
\"\"\"
{transcript}
\"\"\"

Return ONLY a JSON object (no markdown, no commentary) shaped EXACTLY like this:
{{
  "one_line_assessment": "<plain-language, balanced, specific summary of the call>",
  "scenario": "<one scenario from the list>",
  "dimensions": {{
    "customer_understood": {{"score": <1-5 or null>, "applicable": <true|false>, "evidence_sufficient": <true|false>, "confidence": <0.0-1.0>, "evidence": "<quote/paraphrase + ~timestamp>"}},
    "pet_acknowledged":    {{"score": <1-5 or null>, "applicable": <true|false>, "evidence_sufficient": <true|false>, "confidence": <0.0-1.0>, "evidence": "..."}},
    "empathy":             {{"score": <1-5 or null>, "applicable": <true|false>, "evidence_sufficient": <true|false>, "confidence": <0.0-1.0>, "evidence": "..."}},
    "ownership":           {{"score": <1-5 or null>, "applicable": <true|false>, "evidence_sufficient": <true|false>, "confidence": <0.0-1.0>, "evidence": "..."}},
    "resolution":          {{"score": <1-5 or null>, "applicable": <true|false>, "evidence_sufficient": <true|false>, "confidence": <0.0-1.0>, "evidence": "..."}},
    "warmth":              {{"score": <1-5 or null>, "applicable": <true|false>, "evidence_sufficient": <true|false>, "confidence": <0.0-1.0>, "evidence": "..."}}
  }},
  "what_went_well": ["<one or two evidence-backed strengths>"],
  "biggest_missed_moment": {{"summary": "<the single most valuable improvement>", "timestamp": "<~mm:ss>"}},
  "try_instead": "<one natural sentence appropriate to that exact moment>",
  "resolution_status": "<resolved|partially_resolved|unresolved|unclear> - <short reason>",
  "commitments": [{{"action": "...", "owner": "...", "due": "...", "in_ticket": "<yes|no|unclear>"}}],
  "risks_alerts": [{{"severity": "<critical|high|medium|coaching>", "summary": "...", "evidence": "..."}}],
  "policy_check": {{"status": "<aligned|possible_mismatch|cannot_verify>", "note": "..."}},
  "coaching_tag": "<one theme only, e.g. 'specific empathy' or 'clearer closure'>",
  "failure_attribution": "<agent|policy|knowledge|technology|fulfilment|product|customer|shared>",
  "critical_failure": <true|false>,
  "high_severity_failure": <true|false>
}}
"""


def _extract_json(text):
    text = text.strip()
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
        model=model, max_tokens=4000, temperature=0,
        messages=[{"role": "user", "content": _prompt(transcript, meta)}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
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
        model=model, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "user", "content": _prompt(transcript, meta)}],
    )
    return _extract_json(resp.choices[0].message.content)


def _coerce_dimension(raw):
    """Validate/normalise one dimension entry from the LLM."""
    applicable = bool(raw.get("applicable", True))
    score = raw.get("score", None)
    if score is not None:
        try:
            score = max(1.0, min(5.0, float(score)))
        except (TypeError, ValueError):
            score = None
    evidence_sufficient = bool(raw.get("evidence_sufficient", score is not None))
    conf = raw.get("confidence", None)
    if conf is not None:
        try:
            conf = max(0.0, min(1.0, float(conf)))
        except (TypeError, ValueError):
            conf = None
    if not applicable:
        score, evidence_sufficient = None, False
    return {
        "score": score, "applicable": applicable,
        "evidence_sufficient": evidence_sufficient, "confidence": conf,
        "evidence": (raw.get("evidence") or "").strip() or "No evidence text returned.",
    }


def _assemble(raw, method_label):
    result = empty_result()
    result["one_line_assessment"] = (raw.get("one_line_assessment") or "").strip()
    result["scenario"] = (raw.get("scenario") or "").strip()

    raw_dims = raw.get("dimensions") or {}
    for d in DIMENSIONS:
        entry = _coerce_dimension(raw_dims.get(d["key"]) or {})
        entry["weight"] = d["weight"]
        result["dimensions"][d["key"]] = entry

    result["what_went_well"] = raw.get("what_went_well") or []
    bmm = raw.get("biggest_missed_moment") or {}
    result["biggest_missed_moment"] = {
        "summary": (bmm.get("summary") or "").strip(),
        "timestamp": (bmm.get("timestamp") or "").strip(),
    }
    result["try_instead"] = (raw.get("try_instead") or "").strip()
    result["resolution_status"] = (raw.get("resolution_status") or "").strip()
    result["commitments"] = raw.get("commitments") or []
    result["risks_alerts"] = raw.get("risks_alerts") or []
    pc = raw.get("policy_check") or {}
    result["policy_check"] = {"status": (pc.get("status") or "").strip(),
                              "note": (pc.get("note") or "").strip()}
    result["coaching_tag"] = (raw.get("coaching_tag") or "").strip()
    result["failure_attribution"] = (raw.get("failure_attribution") or "").strip()
    result["critical_failure"] = bool(raw.get("critical_failure"))
    result["high_severity_failure"] = bool(raw.get("high_severity_failure"))
    result["scoring_method"] = method_label
    recompute(result)
    return result


def llm_score(transcript, meta=None):
    meta = meta or {}
    for fn, label in ((_score_with_anthropic, "Claude"), (_score_with_openai, "OpenAI GPT")):
        try:
            raw = fn(transcript, meta)
        except Exception as e:  # noqa: BLE001 - fall back on any provider error
            raw = None
            print(f"[scoring] {label} scoring failed: {e}")
        if raw:
            return _assemble(raw, f"AI-scored ({label})"), f"AI-scored ({label})"
    return None, None


# ---------------------------------------------------------------------------
# 4. Fallback when no LLM key is configured
# ---------------------------------------------------------------------------
def _needs_review_result():
    """The brief forbids fabricated judgments. With no LLM available, return a
    clearly-flagged 'needs review' result rather than fake keyword scores."""
    result = empty_result()
    note = ("No AI scoring key configured. This call needs an AI evaluation "
            "(ANTHROPIC_API_KEY or OPENAI_API_KEY) and/or human review - the "
            "HUFT model does not produce keyword-based guesses.")
    result["one_line_assessment"] = note
    result["score_status"] = "insufficient_evidence"
    result["rating_band"] = "Insufficient evidence"
    for k in result["dimensions"]:
        result["dimensions"][k]["evidence"] = "Awaiting AI evaluation / human review."
    return result


def score_transcript(transcript, meta=None):
    """Main entry point. Returns (result_dict, method_label)."""
    meta = meta or {}
    result, method = llm_score(transcript, meta)
    if result:
        return result, method
    return _needs_review_result(), "Needs review (no AI scoring key configured)"


# ---------------------------------------------------------------------------
# 5. Coaching helper (single priority per Section 10 "Next coaching tag")
# ---------------------------------------------------------------------------
def coaching_recommendations(result, top_n=3):
    """Weakest applicable dimensions -> short coaching notes. Complements the
    single coaching_tag the LLM already returns."""
    dims = result.get("dimensions", {})
    ranked = []
    for d in DIMENSIONS:
        v = dims.get(d["key"], {})
        if not v.get("applicable") or v.get("score") is None:
            continue
        ranked.append((float(v["score"]), d, v))
    ranked.sort(key=lambda t: t[0])
    out = []
    for score, d, v in ranked[:top_n]:
        if score >= 4:
            continue
        out.append({
            "label": d["label"], "score": score,
            "tip": f'Coach on "{d["label"]}": {d["question"]}',
            "evidence": v.get("evidence", ""),
        })
    return out
