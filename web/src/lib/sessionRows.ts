export const COLLAPSED_SESSION_LIMIT = 5;

export type SessionContentHit = {
  turn: number;
  role: "user" | "assistant";
  snippet: string;
};

export type SessionRow = {
  id: string;
  session: {
    title: string;
    preview?: string;
    running?: boolean;
    pinned?: boolean;
  };
};

export function sessionMatches(
  row: SessionRow,
  projectName: string,
  query: string,
  contentHits?: SessionContentHit[],
) {
  const q = query.trim().toLocaleLowerCase();
  if (!q) return true;
  if (contentHits && contentHits.length > 0) return true;
  return [row.session.title, row.session.preview || "", projectName]
    .some((value) => value.toLocaleLowerCase().includes(q));
}

function snippetAround(text: string, needle: string, before = 24, after = 56): string {
  const flat = text.replace(/\s+/g, " ").trim();
  const idx = flat.toLocaleLowerCase().indexOf(needle);
  if (idx < 0) return flat.length > 80 ? flat.slice(0, 80) + "…" : flat;
  const start = Math.max(0, idx - before);
  const end = Math.min(flat.length, idx + needle.length + after);
  return (start > 0 ? "…" : "") + flat.slice(start, end) + (end < flat.length ? "…" : "");
}

function stripWindowPreamble(content: string): string {
  if (!content.startsWith("<context_window>")) return content;
  for (const m of ["\n</dropped_span>\n", "\n</session_notes>\n", "\n</context_window>\n"]) {
    const at = content.indexOf(m);
    if (at >= 0) {
      let rest = content.slice(at + m.length).replace(/^\n+/, "");
      if (rest.startsWith("New context window started")) {
        const nl = rest.indexOf("\n");
        rest = nl >= 0 ? rest.slice(nl + 1).replace(/^\n+/, "") : "";
      }
      return rest;
    }
  }
  return "";
}

/** Search user queries and assistant answers inside a hydrated `msgs` list. */
export function searchMsgsConversation(
  msgs: Array<{ role: string; content: string; steer?: boolean; automated?: boolean; compact?: unknown }>,
  olderTurns: number,
  query: string,
  maxHits = 12,
): SessionContentHit[] {
  const needle = query.trim().replace(/\s+/g, " ").toLocaleLowerCase();
  if (!needle) return [];
  const hits: SessionContentHit[] = [];
  let turn = olderTurns || 0;
  const seen = new Set<string>();
  for (const m of msgs) {
    if (m.compact) continue;
    if (m.role === "user") {
      if (m.automated) continue;
      if (!m.steer) turn += 1;
      const clean = (m.content || "").trim();
      if (!clean) continue;
      const key = `${Math.max(1, turn)}:user`;
      if (!seen.has(key) && clean.replace(/\s+/g, " ").toLocaleLowerCase().includes(needle)) {
        seen.add(key);
        hits.push({ turn: Math.max(1, turn), role: "user", snippet: snippetAround(clean, needle) });
      }
    } else if (m.role === "assistant") {
      const clean = (m.content || "").trim();
      if (!clean) continue;
      const key = `${Math.max(1, turn)}:assistant`;
      if (!seen.has(key) && clean.replace(/\s+/g, " ").toLocaleLowerCase().includes(needle)) {
        seen.add(key);
        hits.push({ turn: Math.max(1, turn), role: "assistant", snippet: snippetAround(clean, needle) });
      }
    }
    if (hits.length >= maxHits) break;
  }
  return hits;
}

/** Search raw JSONL events (client-side fallback when gateway is not restarted). */
export function searchEventsConversation(events: unknown[], query: string, maxHits = 8): SessionContentHit[] {
  const needle = query.trim().replace(/\s+/g, " ").toLocaleLowerCase();
  if (!needle) return [];
  const hits: SessionContentHit[] = [];
  let turn = 0;
  const seen = new Set<string>();
  for (const raw of events) {
    if (!raw || typeof raw !== "object") continue;
    const e = raw as Record<string, unknown>;
    const type = String(e.type || "");
    if (type === "user") {
      if (e.automated === true) continue;
      const content = String(e.content || "");
      const isSteer = content.includes("__RELAYED__");
      let clean = "";
      if (!isSteer) {
        clean = stripWindowPreamble(content).trim();
        if (!clean) continue;
        turn += 1;
      } else {
        clean = content.replace(/__RELAYED__/g, "").trim();
      }
      const key = `${Math.max(1, turn)}:user`;
      if (!seen.has(key) && clean.replace(/\s+/g, " ").toLocaleLowerCase().includes(needle)) {
        seen.add(key);
        hits.push({ turn: Math.max(1, turn), role: "user", snippet: snippetAround(clean, needle) });
      }
    } else if (type === "assistant") {
      const clean = String(e.content || "").trim();
      if (!clean) continue;
      const key = `${Math.max(1, turn)}:assistant`;
      if (!seen.has(key) && clean.replace(/\s+/g, " ").toLocaleLowerCase().includes(needle)) {
        seen.add(key);
        hits.push({ turn: Math.max(1, turn), role: "assistant", snippet: snippetAround(clean, needle) });
      }
    }
    if (hits.length >= maxHits) break;
  }
  return hits;
}

export function collapseSessionRows<T extends SessionRow>(
  rows: T[],
  currentId: string | null,
  expanded: boolean,
  limit = COLLAPSED_SESSION_LIMIT,
): { visible: T[]; hiddenCount: number } {
  if (expanded) return { visible: rows, hiddenCount: 0 };
  let idleCount = 0;
  const visible = rows.filter((row) => {
    if (row.id === currentId || row.session.running || row.session.pinned) return true;
    idleCount += 1;
    return idleCount <= limit;
  });
  return { visible, hiddenCount: rows.length - visible.length };
}
