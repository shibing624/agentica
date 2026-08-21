import { useEffect, useRef, useState } from "react";
import { useStrings } from "../i18n";
import { IconBook } from "../icons";

export type SlashItem = { cmd: string; name: string; desc: string; kind: "command" | "skill" };

export function slashQuery(text: string): string | null {
  const m = text.match(/^(\/[^\s]*)$/);
  return m ? m[1].toLowerCase() : null;
}

export function webSlashItems(skills: Array<Record<string, any>>, S: {
  slashGoal: string; slashCompact: string; slashQueue: string;
}): SlashItem[] {
  const out: SlashItem[] = [
    { cmd: "/goal", name: "/goal", desc: S.slashGoal, kind: "command" },
    { cmd: "/queue", name: "/queue", desc: S.slashQueue, kind: "command" },
    { cmd: "/compact", name: "/compact", desc: S.slashCompact, kind: "command" },
  ];
  for (const sk of skills || []) {
    if (sk.is_hidden || sk.user_invocable === false) continue;
    const cmd = String(sk.slash || sk.trigger || "").toLowerCase();
    if (!cmd.startsWith("/")) continue;
    out.push({ cmd, name: sk.name || cmd, desc: sk.description || "", kind: "skill" });
  }
  return out;
}

export function filterSlashItems(items: SlashItem[], query: string): SlashItem[] {
  return items.filter((it) => it.cmd.startsWith(query) || it.cmd.slice(1).startsWith(query.slice(1)));
}

export function invocableSkills(skills: Array<Record<string, any>>): Array<{ cmd: string; name: string; desc: string }> {
  const out: Array<{ cmd: string; name: string; desc: string }> = [];
  for (const sk of skills || []) {
    if (sk.is_hidden || sk.user_invocable === false) continue;
    const cmd = String(sk.slash || sk.trigger || "").toLowerCase();
    if (!cmd.startsWith("/")) continue;
    out.push({ cmd, name: String(sk.name || cmd), desc: String(sk.description || "") });
  }
  return out;
}

export function SlashMenu({
  items, active, onPick,
}: {
  items: SlashItem[];
  active: number;
  onPick: (item: SlashItem) => void;
}) {
  const S = useStrings();
  const listRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = listRef.current?.querySelector(".quick-item.active");
    if (el) el.scrollIntoView({ block: "nearest" });
  }, [active]);
  if (!items.length) {
    return <div className="quick-dd open slash-menu"><div className="quick-empty">{S.chat.slashEmpty}</div></div>;
  }
  return (
    <div className="quick-dd open slash-menu">
      <div className="quick-list" ref={listRef}>
        {items.map((it, i) => (
          <button
            type="button"
            key={it.kind + it.cmd}
            className={"quick-item quick-item-ref" + (i === active ? " active" : "")}
            title={it.desc}
            onMouseDown={(e) => { e.preventDefault(); onPick(it); }}
          >
            {it.kind === "skill" && <span className="quick-icon"><IconBook /></span>}
            <span className="quick-item-text">
              <span>{it.name}</span>
              {it.desc ? <span className="quick-item-desc">{it.desc}</span> : null}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

export function SkillsPicker({
  skills, onPick,
}: {
  skills: Array<Record<string, any>>;
  onPick: (cmd: string) => void;
}) {
  const S = useStrings();
  const [q, setQ] = useState("");
  const needle = q.trim().toLowerCase();
  const items = invocableSkills(skills).filter((it) => {
    if (!needle) return true;
    return it.name.toLowerCase().includes(needle)
      || it.cmd.toLowerCase().includes(needle)
      || it.desc.toLowerCase().includes(needle);
  });
  return (
    <div className="quick-dd open skills-dd">
      <input
        className="quick-search"
        value={q}
        autoFocus
        placeholder={S.chat.skillSearch}
        onChange={(e) => setQ(e.target.value)}
        onKeyDown={(e) => e.stopPropagation()}
      />
      <div className="quick-list">
        {!items.length && <div className="quick-empty">{S.chat.skillNone}</div>}
        {items.map((it) => (
          <button
            type="button"
            key={it.cmd}
            className="quick-item quick-item-ref"
            title={it.desc}
            onMouseDown={(e) => { e.preventDefault(); onPick(it.cmd); }}
          >
            <span className="quick-icon"><IconBook /></span>
            <span className="quick-item-text">
              <span>{it.name}</span>
              {it.desc ? <span className="quick-item-desc">{it.desc}</span> : null}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
