You are deciding whether ONE of the experience cards below should be upgraded into a reusable SKILL.md file.

A SKILL.md is a "don't step on this landmine again" note, NOT a "how to do X" tutorial. Every user-correction card with repeat_count >= 3 is a rule the user reinforced multiple times - strongly prefer install_shadow for the highest-repeat correction card unless it is genuinely a one-off preference. Tool-error cards alone are NOT skills, but a matching correction card next to them IS.

Decision recipe:
1. Pick the candidate with the highest repeat_count whose type is "correction".
2. If its repeat_count >= 3, return action=install_shadow.
3. Skip only if (a) only tool_error candidates are high-repeat, (b) an existing generated skill already covers this rule, or (c) the rule is a one-off preference with no procedural content.

Return JSON only:
{"action": "ignore|install_shadow", "skill_name": "kebab-case-slug", "source_experience": "title of the source experience card", "reason": "why this deserves to be a skill", "skill_md": "full SKILL.md content (see format below)"}

## skill_md format (gotcha-first, NOT textbook)

The skill_md string MUST NOT be wrapped in ```yaml or any other code fence. It MUST start with '-' (the opening '---').

FRONTMATTER (minimal, exactly 3 keys):
---
name: <kebab-case slug, equals skill_name above>
description: <one sentence, <=25 words>
when-to-use: <comma-separated keywords for discovery>
---

BODY STRUCTURE (strict, in this order):
1. One-line summary (<=30 words).
2. ## Gotchas (REQUIRED, MUST have >=2 items).
   - Each gotcha = one observed failure + the fix.
   - Format: '[warning] <symptom>: <root cause>. <minimal fix>'
   - Every gotcha MUST be traceable to evidence in the cards / raw events shown above. Do NOT invent gotchas.
3. ## Minimal Example (<=10 lines, real params, no '# TODO' placeholders, no '<your_value_here>').
4. ## Source (auto-filled by the system, leave a blank section).

FORBIDDEN (will be auto-rejected):
- Sections named 'Overview' / 'When To Use' / 'Workflow' / 'Failure Recovery' (these are textbook fluff).
- Generic steps the agent could derive from reading docs.
- Unverified claims (every gotcha must trace to a real event).
- Placeholder code: '# TODO', '<your_*_here>', 'FIXME', 'pass  # implement'.
- Skeleton code blocks with <10 chars per line on average.

Remember: a skill captures lessons that can ONLY be learned by actually running the tool and getting burned. If you cannot point to a concrete failure event for a gotcha, do NOT include it.
