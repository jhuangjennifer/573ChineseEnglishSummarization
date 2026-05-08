# Cell 5: Prompt templates

# Agent 1: Semantic Understanding
SEMANTIC_UNDERSTANDING_PROMPT = """Analyze the English dialogue and extract only the key semantic events needed for the final summary.

Focus on:
- participants
- speaker intent
- actor/action/recipient relations
- final decisions, refusals, commitments, changed plans, or cancellations
- the final outcome of the dialogue

Rules:
- Do not analyze every utterance.
- Ignore greetings, jokes, reactions, and closings unless they affect the outcome.
- Resolve ambiguity from context.
- For imperatives, the actor is usually the listener, not the speaker.
  Example: if A says "Just text him" to B, B is the actor.
- Do not infer visual details from placeholders such as <file_photo>, <file_gif>, or <file>.

Extract 1-6 key events. Use fewer events for simple dialogues.
Write final_outcome as one concise sentence.

Output valid JSON only.

Schema:
{
  "participants": ["speaker names"],
  "semantic_grounding": [
    {
      "event_id": 1,
      "speaker": "speaker name",
      "speech_act": "short label",
      "intended_meaning": "concise interpretation",
      "actor": "person or null",
      "action": "concise action",
      "object": "object or null",
      "recipient": "person or null",
      "evidence": ["short quote"]
    }
  ],
  "final_outcome": "one concise sentence"
}

Input dialogue:
{dialogue}

JSON output:
"""

# Agent 2: Event Selection and Chinese Summary Generation
SUMMARY_GENERATION_PROMPT = """You will receive a structured semantic representation of an English dialogue.

Select the most important events and write a concise Chinese summary.

Rules:
- Select only 1-3 salient events.
- Prioritize the final outcome, changed plans, decisions, refusals, commitments, and important actor/action relations.
- If an earlier plan is later changed, summarize the final updated state.
- Do not invent details from placeholders.
- Use natural Chinese.
- Match the style of a short factual dataset summary.
- For simple dialogues, keep the summary short.
- For complex dialogues, include only necessary details.

Output valid JSON only.

Schema:
{
  "selected_events": [
    {
      "event_id": 1,
      "key_event": "short description"
    }
  ],
  "summary_zh": "Chinese summary"
}

Input semantic representation:
{semantic_representation}

JSON output:
"""

# Agent 3: Concise Verification and Revision
REVISION_PROMPT = """Verify the draft Chinese summary against the original English dialogue.

Revise only if necessary.

Check for:
- hallucination
- missing final outcome
- wrong actor/action/recipient
- missing key event
- outdated plan
- unsupported visual detail
- awkward Chinese
- unnecessary verbosity

Rules:
- Trust the original dialogue if it conflicts with selected_events.
- Do not rewrite merely for style.
- If the draft is correct, keep it exactly unchanged.
- Be extremely concise.
- Use short issue tags only.
- revision_reason must be under 12 words.
- Output valid JSON only.

Allowed issue tags:
"hallucination", "wrong_actor", "wrong_action", "wrong_recipient",
"missing_final_outcome", "missing_key_event", "outdated_plan",
"too_verbose", "awkward_chinese", "unsupported_visual_detail"

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

Draft Chinese summary and selected events:
{summary_points}

JSON output:
"""