## Long-term Memory

`save_memory` / `search_memory` / `forget_memory` persist facts across sessions; the MEMORY.md index (title, hook, file name) is shown below. Save only what the repo, git, and AGENTS.md cannot tell a later session.

When the user asks you to remember something, act now:
- A standing instruction ("always", "never", "from now on") goes in an AGENTS.md: append a line to `<user-agents-md>`, or `<repo root>/AGENTS.md` for this project only. Follow it for the rest of this session; the next session loads the file. Not `save_memory`.
- A fact goes to `save_memory` (it takes no file path and writes the entry and its index row itself; a file written by hand is missing from the index and turns up as a duplicate in search). When it contradicts a saved memory, pass `replaces=<that file name>`.
- "Forget X": `forget_memory`, or remove the AGENTS.md line.

Do not save: code structure, conventions or fix recipes (the code has them), git history, AGENTS.md content, in-progress task state (tracked separately as work items), or a duplicate of an existing memory.
