import { useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { useStrings } from "../i18n";
import { measureUserBubbleFold } from "../lib/userFold";
import { Dialog } from "./Dialog";

/** A user bubble's body, folded to a few lines when it is long (a pasted log or
 *  document). Folding is decided by the rendered height, so it follows the
 *  bubble width and font instead of a character count. `forceOpen` unfolds it
 *  when a chat search lands on this message. */
export function CollapsibleUserText({ children, forceOpen }: { children: ReactNode; forceOpen?: boolean }) {
  const S = useStrings();
  const ref = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [overflows, setOverflows] = useState(false);
  const [limit, setLimit] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => {
      const cs = getComputedStyle(el);
      const res = measureUserBubbleFold(el.scrollHeight, cs.lineHeight, cs.fontSize);
      setLimit(res.limit);
      setOverflows(res.overflows);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  useLayoutEffect(() => {
    if (forceOpen) setOpen(true);
  }, [forceOpen]);
  const folded = overflows && !open;
  return (
    <>
      <div ref={ref} className={"user-text" + (folded ? " folded" : "")}
           style={folded ? { maxHeight: limit } : undefined}>{children}</div>
      {overflows && (
        <button type="button" className="user-fold-toggle" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
          {open ? S.chat.userTextLess : S.chat.userTextMore}
        </button>
      )}
    </>
  );
}

/** `/skill args` in a user bubble: the skill name underlined, then the argument.
 *  Click the name to read its SKILL.md in the right-hand panel, instead of the
 *  whole prompt in the bubble.
 *
 *  `source` is the original user text. Recognition is deliberately strict, but
 *  a pasted document can still look like a skill header — keep the text one
 *  click away so a false positive never hides or loses what the user typed. */
export function SkillTurn({ name, args, source, onOpen }: {
  name: string;
  args: string;
  source?: string;
  onOpen: (name: string) => void;
}) {
  const S = useStrings();
  const [showSource, setShowSource] = useState(false);
  return (
    <>
      <button type="button" className="skill-ref" title={S.chat.skillChipTip} onClick={() => onOpen(name)}>
        /{name}
      </button>
      {args && <span className="skill-args">{args}</span>}
      {source && (
        <>
          <button type="button" className="skill-source-toggle" onClick={() => setShowSource((v) => !v)}>
            {showSource ? S.chat.skillSourceHide : S.chat.skillSourceShow}
          </button>
          {showSource && <pre className="skill-source">{source}</pre>}
        </>
      )}
    </>
  );
}

/** Dislike chips, in display order. Ids are what the session log and the stats db store. */
export const FEEDBACK_TAG_IDS = [
  "task-result",
  "instruction-following",
  "product-interaction",
  "service-stability",
  "resource-cost",
  "security-privacy-permission",
  "other",
] as const;

export type FeedbackTagId = (typeof FEEDBACK_TAG_IDS)[number];

/** Thumbs-down: one or more tags, plus an optional note. Tags alone can be submitted. */
export function FeedbackDialog({ initial, initialTags, onSubmit, onClose }: {
  initial: string;
  initialTags: string[];
  onSubmit: (comment: string, tags: string[]) => void;
  onClose: () => void;
}) {
  const S = useStrings();
  const [text, setText] = useState(initial);
  const [tags, setTags] = useState<string[]>(initialTags);
  const send = () => onSubmit(text, tags);
  const toggle = (id: string) => {
    setTags((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id]));
  };
  return (
    <Dialog
      wide
      title={S.chat.feedbackTitle}
      onClose={onClose}
      footer={
        <>
          <button className="dp-btn" onClick={onClose}>{S.common.cancel}</button>
          <button className="dp-btn primary" onClick={send}>{S.chat.feedbackSend}</button>
        </>
      }
    >
      <div className="feedback-tags" role="group" aria-label={S.chat.feedbackCategories}>
        {FEEDBACK_TAG_IDS.map((id) => (
          <button key={id} type="button"
                  className={"feedback-tag" + (tags.includes(id) ? " on" : "")}
                  aria-pressed={tags.includes(id)}
                  onClick={() => toggle(id)}>
            {S.chat.feedbackTags[id]}
          </button>
        ))}
      </div>
      <textarea className="feedback-text" autoFocus rows={5} value={text}
                placeholder={S.chat.feedbackPlaceholder}
                aria-label={S.chat.feedbackPlaceholder}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); send(); }
                }} />
    </Dialog>
  );
}
