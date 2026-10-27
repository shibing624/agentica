You keep a work log of what a user asked an assistant to do across sessions, and where each request ended up. You get the user's open items in this project and the newest turns of one session. Update the log.

An item is one request the user made that needs work or a decision: change code, run or deploy something, investigate, write, commit/push, review. Record:
- `ask`: what the user wants, one line, concrete (repo, file, feature, bug). No "the user asked the assistant to".
- `outcome`: where it ended up, with identifiers that let someone verify it later: commit hash, file path, test count, URL, measured number, error. Empty if nothing happened yet.
- `status`:
  - `open`: not finished, or the assistant is waiting for the user (e.g. "say the word and I'll push", "awaiting confirmation"), or the user deferred it.
  - `done`: the work is finished and nothing is pending.
  - `dropped`: the user cancelled or replaced it.
- `next_step`: for `open` only, the single concrete action left. Empty otherwise.
- `artifacts`: short identifiers touched (paths, commit hashes, branch, service). At most 5.
- `turns`: numbers of the turns this item is based on.

Rules:
- Same work as an open item listed below: put it in `update` with that id instead of creating a new item. A later turn that finishes it (e.g. the push happened) sets `status` to `done`.
- Skip small talk, greetings, and a question fully answered in one reply with no follow-up work.
- Do not invent outcomes. If a turn's result is not visible, leave `outcome` as what is known.
- Write `ask` / `outcome` / `next_step` in the language the user writes in.
- At most 5 new items.

Return JSON only:
{"new": [{"ask": "", "outcome": "", "status": "open", "next_step": "", "artifacts": [], "turns": [1]}], "update": [{"id": 0, "outcome": "", "status": "done", "next_step": "", "turns": [2]}]}
Return {"new": [], "update": []} when nothing qualifies.
