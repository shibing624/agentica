import { useEffect, useRef, useState } from "react";
import { useStrings } from "../i18n";
import type { JSX, KeyboardEvent, Ref } from "react";
import {
  IconBook, IconCompact, IconDownload, IconModel, IconPaperclip, IconQueue, IconShield, IconTarget,
} from "../icons";
import type { CommandId, MenuGroup, MenuItem } from "../lib/composerCommands";

export function slashQuery(text: string): string | null {
  const m = text.match(/^(\/[^\s]*)$/);
  return m ? m[1].toLowerCase() : null;
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

const ICONS: Record<CommandId, () => JSX.Element> = {
  file: IconPaperclip, goal: IconTarget, queue: IconQueue, compact: IconCompact,
  permission: IconShield, model: IconModel, export: IconDownload,
};

/**
 * One grouped command list (deepseek-harness style): a section title, then
 * rows of icon, label, slash name in grey and a one-line description on the
 * right. Used by the `+` menu and the `/` menu; ``active`` indexes the
 * flattened rows so arrow keys walk across sections.
 */
export function CommandMenu({
  groups, active, onPick, onHover, className = "", emptyText, menuRef, onKeyDown,
}: {
  groups: MenuGroup[];
  active: number;
  onPick: (item: MenuItem) => void;
  onHover?: (index: number) => void;
  className?: string;
  emptyText?: string;
  menuRef?: Ref<HTMLDivElement>;
  onKeyDown?: (e: KeyboardEvent<HTMLDivElement>) => void;
}) {
  const listRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = listRef.current?.querySelector(".cmd-item.active");
    if (el) el.scrollIntoView({ block: "nearest" });
  }, [active]);
  let index = -1;
  return (
    <div className={"cmd-menu " + className} role="menu" ref={menuRef} tabIndex={-1} onKeyDown={onKeyDown}>
      <div className="cmd-list" ref={listRef}>
        {!groups.length && emptyText && <div className="quick-empty">{emptyText}</div>}
        {groups.map((g) => (
          <div className="cmd-group" key={g.title} role="group" aria-label={g.title}>
            <div className="cmd-group-title">{g.title}</div>
            {g.items.map((it) => {
              index += 1;
              const i = index;
              const Icon = it.id ? ICONS[it.id] : null;
              return (
                <button
                  type="button"
                  role="menuitem"
                  key={it.key}
                  className={"cmd-item" + (it.kind === "skill" ? " skill" : "") + (i === active ? " active" : "")}
                  title={it.desc || undefined}
                  onMouseDown={(e) => { e.preventDefault(); onPick(it); }}
                  onMouseMove={() => { if (i !== active) onHover?.(i); }}
                >
                  {Icon && <span className="cmd-icon"><Icon /></span>}
                  <span className="cmd-label">{it.label}</span>
                  {it.kind === "command" && <span className="cmd-slash">{it.cmd.slice(1)}</span>}
                  {it.desc && <span className="cmd-desc">{it.desc}</span>}
                </button>
              );
            })}
          </div>
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
