# Cell 5: Prompt templates

# Agent 1: Semantic-Linguistic Grounding
SEMANTIC_UNDERSTANDING_PROMPT = """Build a compact semantic-linguistic grounding of the English dialogue for Chinese summarization.

Goal: preserve the meaning needed for a faithful human summary, not every utterance.

Capture:
- participants, coreference, and roles
- speech act and predicate-argument structure
- final state, plan, decision, request, refusal, commitment, problem, preference, or outcome
- time/condition, polarity, modality, discourse role, and grounding status
- central attitude/emotion only when explicit and summary-relevant

Rules:
- Merge related turns into one event; prefer final resolved state.
- For yes/no exchanges, use the answer; rejected propositions are established + negated.
- For requests/imperatives, separate requester from expected actor.
- Do not turn unresolved, uncertain, conditional, or implicit content into settled fact.
- Use implicit only when strongly entailed; never guess motives, emotions, or visual details.
- Grounding acts support interpretation; routine acknowledgments are not summary events.
- Extract 1-5 events; simple 2-3 turn dialogues usually need 1 core event.
- Set include_in_summary=true only for core events.
- Keep summary_role under 12 words.
- Output valid JSON only.

Label guide:
- grounding_act: clarification resolves ambiguity; follow_up asks for more information; acknowledgment signals understanding/empathy; answer resolves a question; confirmation confirms a proposal/fact; correction fixes an assumption.
- salience: core belongs in the summary; supporting explains a core event; background is normally omitted.
- modality: actual happened; planned/intended/requested/possible/obligatory did not necessarily happen.

Schema:
{
  "participants": [
    {"name": "speaker name", "role_or_relation": "known relation or null"}
  ],
  "semantic_grounding": [
    {
      "event_id": 1,
      "speech_act": "request | answer | commitment | decision | refusal | plan_change | cancellation | offer | information_update | problem | preference | agreement | attitude | emotion | correction | accusation",
      "grounding_act": "none | clarification | follow_up | acknowledgment | answer | confirmation | correction",
      "common_ground_status": "established | clarified | unresolved | implicit",
      "salience": "core | supporting | background",
      "include_in_summary": true,
      "meaning": "concise event meaning",
      "actor": "person or null",
      "action": "base verb phrase",
      "object": "specific object/result or null",
      "recipient": "person or null",
      "time_or_condition": "time, condition, or null",
      "polarity": "affirmed | negated | uncertain | conditional",
      "modality": "actual | planned | intended | requested | possible | obligatory | unknown",
      "discourse_role": "final_outcome | main | cause | condition | contrast | background",
      "summary_role": "why it matters",
      "evidence": ["1-2 short quotes"]
    }
  ],
  "final_outcome": "one concise sentence"
}

Input dialogue:
{dialogue}

JSON output:
"""

# Agent 2: Event Selection and Chinese Summary Generation
SUMMARY_GENERATION_PROMPT = """Write a concise Chinese summary from the semantic-linguistic representation.

You cannot see the original dialogue. Use only the representation, final_outcome, and evidence.

Rules:
- Select 1-3 events for most dialogues; use up to 4 only for complex or multi-party dialogues.
- Prefer include_in_summary=true, salience=core, and discourse_role=final_outcome/main.
- Preserve central actor/action/object/recipient/time_or_condition.
- Preserve common_ground_status, polarity, and modality when they affect meaning.
- Do not present unresolved, uncertain, conditional, implicit, planned, intended, requested, possible, or obligatory content as completed fact.
- Use supporting events only for necessary context; omit background and routine acknowledgments.
- Do not add details beyond the representation, final_outcome, or evidence.
- Use natural Chinese; translate personal names when natural, but keep brands, acronyms, titles, URLs, apps, and technical terms in common form.
- Output valid JSON only.

Schema:
{
  "selected_events": [
    {"event_id": 1, "selection_reason": "core final outcome | needed context | central condition"}
  ],
  "summary_zh": "Chinese summary"
}

Input semantic-linguistic representation:
{semantic_representation}

JSON output:
"""

# Agent 3: Grounded Verification and Revision
REVISION_PROMPT = """Verify the draft Chinese summary against the original dialogue and semantic-linguistic representation.

Revise only if necessary.

Rules:
- Trust the original dialogue over the representation or selected_events.
- Check missing key information, final outcome, hallucination, grounding, actor/action/object/recipient/time/condition, polarity, modality, outdated plans, visual details, and Chinese clarity.
- Remove unsupported interpretation, intensified emotion, unnecessary reaction, or non-central background.
- Do not remove central information just to shorten the summary.
- Do not rewrite the summary from scratch.
- Verify and minimally revise the draft; do not generate a new summary directly from the dialogue.
- Add new information only when it is central, clearly supported, and missing from the draft.
- Keep brands, acronyms, titles, URLs, apps, and technical terms in common form.
- If already faithful, informative, concise, and clear, keep it exactly unchanged.
- Use short issue tags; revision_reason under 12 words.
- Output valid JSON only.

Allowed issue tags:
"hallucination", "wrong_actor", "wrong_action", "wrong_object",
"wrong_recipient", "wrong_time", "wrong_condition", "wrong_polarity",
"wrong_modality", "missing_final_outcome", "missing_key_event",
"outdated_plan", "ungrounded_event", "low_salience_event",
"unsupported_interpretation", "unsupported_visual_detail", "too_verbose",
"awkward_chinese", "name_translation_error", "term_translation_error"

Schema:
{
  "needs_revision": false,
  "issues_identified": [],
  "events_dropped": [],
  "revision_reason": "",
  "summary_zh_final": "Chinese summary"
}

Original English dialogue:
{dialogue}

Semantic-linguistic representation:
{semantic_representation}

Draft Chinese summary and selected events:
{summary_points}

JSON output:
"""
