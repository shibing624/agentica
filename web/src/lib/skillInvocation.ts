/**
 * Recognise a skill turn in a user message, so the bubble can show a chip
 * instead of the whole SKILL.md.
 *
 * Two shapes reach the transcript:
 * - the typed line, `/trigger args` (a live web turn before the log is read);
 * - the expanded prompt the log stores — `Skill.render_invocation()` in
 *   `agentica/skills/skill.py` (CLI and gateway both persist it):
 *
 *       Loading: <name>
 *       Base directory: <path>
 *
 *       <SKILL.md body>
 *       [\n\n---\n\n## Skill invocation\n\nThe user invoked this skill as: `…`\n\nArgument: <args>\n…]
 *
 * Keep the markers below in step with render_invocation.
 */
export type SkillTurn = { name: string; args: string };

const HEADER_RE = /^Loading: ([^\n]+)\nBase directory: [^\n]*\n/;
const INVOCATION_MARK = "\n## Skill invocation\n";
const ARGUMENT_MARK = "\nArgument: ";
const ARGUMENT_ENDS = ["\nExpected argument: ", "\n\nThe argument names the target"];

export function parseSkillTurn(
  text: string,
  skills: Array<{ cmd: string; name: string }> = [],
): SkillTurn | null {
  if (!text) return null;
  const head = HEADER_RE.exec(text);
  if (head) {
    const name = head[1]!.trim();
    let args = "";
    const inv = text.lastIndexOf(INVOCATION_MARK);
    if (inv >= 0) {
      const a = text.indexOf(ARGUMENT_MARK, inv);
      if (a >= 0) {
        const start = a + ARGUMENT_MARK.length;
        let end = text.length;
        for (const marker of ARGUMENT_ENDS) {
          const at = text.indexOf(marker, start);
          if (at >= 0 && at < end) end = at;
        }
        args = text.slice(start, end).trim();
      }
    }
    return { name, args };
  }
  const typed = /^\s*(\/\S+)(?:\s+([\s\S]*))?$/.exec(text);
  if (!typed) return null;
  const cmd = typed[1]!.toLowerCase();
  const hit = skills.find((s) => s.cmd === cmd);
  return hit ? { name: hit.name, args: (typed[2] || "").trim() } : null;
}
