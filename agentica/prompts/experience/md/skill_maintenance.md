You are maintaining a generated SKILL.md that recently failed multiple times.

Decide whether this skill should be repaired or discarded. Prefer repair only when the failures point to a local fix in the skill instructions. Discard when the method is obsolete, misleading, conflicts with newer guidance, or depends on a removed tool/API.

Return JSON only:
{"decision": "repair|discard", "reason": "...", "revised_skill_md": "full repaired SKILL.md when decision=repair"}

If the skill cannot be repaired, you may also reply with a line starting with DISCARD followed by the reason.
