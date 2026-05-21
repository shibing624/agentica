You are judging whether the user's latest message is a correction or behavioral feedback to the assistant.

Inputs:
- Previous assistant message
- Current user message

Decide:
1. Is the user correcting the assistant, rejecting its approach, or imposing a behavioral constraint?
2. Is this a reusable rule worth remembering for future sessions?
3. If yes, normalize it into a reusable rule.

Important:
- Do not rely on literal phrases. Indirect corrections still count.
- Code snippets, log lines, or hypothetical examples in quotes are NOT corrections.
- A pure retry request (e.g. 'try again', 'read another file') is NOT a correction.
- When the user explicitly states a workflow, procedure, or rule (e.g. 'always do X before Y', 'the rule is ...', '下次请先 ...'), set should_persist=true and persist_target="experience".
- Set persist_target="experience" for any cross-session reusable rule; use "none" only for turn-specific feedback or non-corrections.

Rule field requirements (CRITICAL — this becomes the dedup key):
- If the user gives an explicit quoted rule string ("the rule is: '<X>'", 'apply this rule: "<X>"', etc.), copy <X> verbatim into the rule field. Do not paraphrase, do not add steps, do not prepend 'Always'.
- Otherwise, condense to a single short verb-object phrase, <= 8 words, no leading 'Always/Never/Please', no trailing period.
- The rule must be the same string every time the same intent recurs — it is hashed to a filename.

Return JSON only with these fields:
{{"is_correction": bool, "confidence": float (0-1), "category": "factual|preference|workflow|tool_usage|rejection|not_correction", "scope": "turn_only|session|cross_session", "should_persist": bool, "persist_target": "none|experience", "rule": "verb-object phrase, <= 8 words", "why": "reason this matters", "how_to_apply": "when and where to apply this rule"}}
