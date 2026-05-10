You are evaluating a shadow-installed generated skill based on its runtime performance episodes.

Signals to weigh (the most important first):
1. gotchas_hit_count > 0 - strong evidence the skill saved the agent from documented landmines. Lean toward PROMOTE.
2. new_gotchas_seen > 0 - the skill is working but incomplete. Lean toward REVISE and rewrite the gotchas section to cover them.
3. consecutive_failures or low success rate without any gotchas_hit - the skill might be misleading. Lean toward ROLLBACK.
4. Otherwise, KEEP_SHADOW until more data accumulates.

Decisions:
- keep_shadow: not enough data yet, keep running
- promote: skill is performing well, promote to full status
- revise: skill idea is good but needs changes - prefer returning section_updates so the system can patch the current SKILL.md instead of rewriting the whole file
- rollback: skill is causing problems, disable it

Return JSON only:
{"decision": "keep_shadow|promote|revise|rollback", "reason": "...", "section_updates": {"summary": "...", "gotchas": ["...", "..."], "minimal_example": "..."} (preferred for revise, otherwise null), "revised_skill_md": "..." (legacy fallback, only if decision is revise)}
