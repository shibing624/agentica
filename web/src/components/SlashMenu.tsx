import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { useStrings } from "../i18n";
import type { JSX, KeyboardEvent, Ref } from "react";
import {
  IconBook, IconCompact, IconDownload, IconModel, IconPaperclip, IconQueue, IconShield, IconTarget,
} from "../icons";
import { filterPicker, type CommandId, type MenuGroup, type MenuItem, type PickerGroup } from "../lib/composerCommands";
export type { PickerGroup, PickerOption } from "../lib/composerCommands";

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
  file: IconPaperclip, skill: IconBook, goal: IconTarget, queue: IconQueue, compact: IconCompact,
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

/**
 * deepseek-harness style picker that opens upward over the composer: an
 * optional search box on top, grey group titles, one row per option. Arrow
 * keys / Enter / Esc work from the search box (or the list when there is
 * none); the cursor starts on the selected option.
 */
export function ComposerPicker({
  groups, onPick, onClose, searchPlaceholder, emptyText, footer, menuRef,
}: {
  groups: PickerGroup[];
  onPick: (key: string) => void;
  onClose: () => void;
  /** Present → show a search box. */
  searchPlaceholder?: string;
  emptyText: string;
  footer?: { label: string; onClick: () => void };
  menuRef?: Ref<HTMLDivElement>;
}) {
  const [query, setQuery] = useState("");
  const shown = filterPicker(groups, query);
  const flat = shown.flatMap((g) => g.options);
  const [active, setActive] = useState(() => Math.max(0, flat.findIndex((o) => o.selected)));
  const listRef = useRef<HTMLDivElement>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  useLayoutEffect(() => {
    // Opens upward: cap it to the room above the composer (minus the top bar)
    // so the search box never ends up off-screen; only the list scrolls.
    const el = boxRef.current;
    const anchor = el?.offsetParent as HTMLElement | null;
    if (el && anchor) {
      const room = anchor.getBoundingClientRect().top - 64;
      el.style.maxHeight = Math.max(160, Math.min(460, room)) + "px";
    }
    (searchRef.current || boxRef.current)?.focus();
  }, []);
  useEffect(() => {
    listRef.current?.querySelector(".cmd-item.active")?.scrollIntoView({ block: "nearest" });
  }, [active]);
  const cur = Math.min(active, Math.max(0, flat.length - 1));
  const onKeyDown = (e: KeyboardEvent<HTMLElement>) => {
    if (e.nativeEvent.isComposing) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (flat.length) setActive((cur + (e.key === "ArrowDown" ? 1 : -1) + flat.length) % flat.length);
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (flat[cur]) onPick(flat[cur].key);
    } else if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      onClose();
    }
  };
  let index = -1;
  return (
    <div className="cmd-menu picker-menu" role="dialog" tabIndex={-1}
         ref={(el) => {
           (boxRef as { current: HTMLDivElement | null }).current = el;
           if (typeof menuRef === "function") menuRef(el);
           else if (menuRef) (menuRef as { current: HTMLDivElement | null }).current = el;
         }}
         onKeyDown={searchPlaceholder ? undefined : onKeyDown}>
      {searchPlaceholder && (
        <input ref={searchRef} className="picker-search" value={query} placeholder={searchPlaceholder}
               onChange={(e) => { setQuery(e.target.value); setActive(0); }} onKeyDown={onKeyDown} />
      )}
      <div className="cmd-list" ref={listRef} role="listbox">
        {!flat.length && <div className="quick-empty">{emptyText}</div>}
        {shown.map((g) => (
          <div className="cmd-group" key={g.title} role="group" aria-label={g.title}>
            <div className="cmd-group-title">{g.title}</div>
            {g.options.map((o) => {
              index += 1;
              const i = index;
              return (
                <button type="button" role="option" aria-selected={!!o.selected} key={o.key}
                        className={"cmd-item" + (i === cur ? " active" : "") + (o.selected ? " selected" : "")}
                        onMouseDown={(e) => { e.preventDefault(); onPick(o.key); }}
                        onMouseMove={() => { if (i !== cur) setActive(i); }}>
                  {o.icon && <span className="cmd-icon">{o.icon}</span>}
                  <span className="cmd-label">{o.label}</span>
                  {o.sub && <span className="cmd-desc">{o.sub}</span>}
                  {o.selected && <span className="picker-check">✓</span>}
                </button>
              );
            })}
          </div>
        ))}
      </div>
      {footer && (
        <button type="button" className="picker-foot" onMouseDown={(e) => { e.preventDefault(); footer.onClick(); }}>
          {footer.label}
        </button>
      )}
    </div>
  );
}
