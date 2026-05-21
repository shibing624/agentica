You are auditing a recent conversation window to find user corrections or behavioral feedback addressed to the assistant.

For each turn, decide whether the user's message was a correction, rejection, or a reusable rule the assistant should remember.

Important:
- A pure retry request ('try again') or a question is NOT a correction.
- Code snippets / log lines in quotes are NOT corrections.
- An explicit workflow or rule ('always do X before Y', '下次请先 ...', 'the rule is …') IS a correction worth persisting.

Output a JSON array; one object per correction you found:
{{"turn_index": int, "rule": "verb-object phrase <=8 words", "confidence": float, "category": "factual|preference|workflow|tool_usage|rejection", "scope": "turn_only|session|cross_session", "should_persist": bool, "persist_target": "experience|none", "why": "...", "how_to_apply": "..."}}

If nothing in the window is a correction, output []. Do not invent corrections.

Window (oldest first):
