# Cell 5: Prompt templates

# Agent 1: Semantic Understanding
SEMANTIC_UNDERSTANDING_PROMPT = """Analyze the English dialogue and extract only the key semantic events needed for the final summary.

Focus on:
- participants
- actor/action/recipient relations
- final decisions, refusals, commitments, changed plans, or cancellations
- the final outcome of the dialogue
- whether each event is summary-worthy

Rules:
- Do not analyze every utterance.
- Ignore greetings, jokes, reactions, and closings unless they affect the outcome.
- Do not create separate events for reactions, doubts, confirmations, or emotional emphasis unless they change the final outcome.
- Merge multiple turns into one event when they support the same resolved fact, plan, decision, request, refusal, or commitment.
- Prefer the final resolved state over intermediate negotiation steps.
- Resolve ambiguity from context.
- For imperatives, the actor is usually the listener, not the speaker.
  Example: if A says "Just text him" to B, B is the actor.
- Use concise base verb phrases for actions, such as "catch evening flight", not "catching evening flight".
- Make objects specific enough for summary, such as "evening flight home", not just "flight".
- Do not infer visual details from placeholders such as <file_photo>, <file_gif>, or <file>.

Salience:
- Mark an event "core" if it should normally appear in a reference summary.
- Mark an event "supporting" if it is true and helps interpret a core event but should usually be omitted from a short summary.
- Mark an event "background" if it is contextual and should not be selected unless needed for coherence.
- Set include_in_summary to true only for core events.

Evidence:
- Include 1-2 short quotes that directly support the event.
- Evidence is for grounding and revision, not for adding extra details.

Extract 1-5 key events. Use fewer events for simple dialogues.
Write final_outcome as one concise sentence.

Output valid JSON only.

Schema:
{
  "participants": ["speaker names"],
  "semantic_grounding": [
    {
      "event_id": 1,
      "speech_act": "request | commitment | decision | refusal | plan_change | cancellation | offer | information_update | problem | preference",
      "salience": "core | supporting | background",
      "include_in_summary": true,
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
- Prefer events where include_in_summary is true and salience is "core".
- Use supporting events only if they are needed to make the core event understandable.
- Do not include events that are true but not summary-worthy.
- Prioritize the final outcome, changed plans, decisions, refusals, commitments, and important actor/action relations.
- If an earlier plan is later changed, summarize the final updated state.
- Selected events must be supported by their evidence or by the final_outcome.
- Do not invent details from placeholders.
- Use natural Chinese.

STRICT TRANSLATION RULE:
- You MUST translate ALL English proper nouns and speaker names into standard Chinese characters.
- ABSOLUTELY NO English letters or names should appear in the final Chinese summary.

Conciseness:
- For simple dialogues, prefer 20-50 Chinese characters.
- For complex dialogues, allow up to 80 Chinese characters.

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
- selected events not grounded in evidence
- missing final outcome
- wrong actor/action/recipient
- missing key event
- outdated plan
- non-summary-worthy event included
- unsupported visual detail
- awkward Chinese
- unnecessary verbosity
- untranslated English names or proper nouns in the draft Chinese summary

Rules:
- Trust the original dialogue if it conflicts with selected_events.
- Use the evidence fields to check whether selected events are grounded in the dialogue.
- Drop selected events that are true but not important enough for a short reference-style summary.
- Do not rewrite merely for style.
- If the draft is correct, keep it exactly unchanged.
- Be extremely concise.
- Use short issue tags only.
- revision_reason must be under 12 words.
- Output valid JSON only.

Conciseness:
- For simple dialogues, prefer 20-50 Chinese characters.
- For complex dialogues, allow up to 80 Chinese characters.

Allowed issue tags:
"hallucination", "wrong_actor", "wrong_action", "wrong_recipient",
"missing_final_outcome", "missing_key_event", "outdated_plan",
"ungrounded_event", "low_salience_event", "too_verbose", "awkward_chinese",
"unsupported_visual_detail", "untranslated_names"

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