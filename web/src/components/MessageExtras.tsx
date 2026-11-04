import { useState } from "react";
import { useStrings } from "../i18n";
import { Dialog } from "./Dialog";

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

/** Thumbs-down note: what was wrong. Sending empty keeps just the vote. */
export function FeedbackDialog({ initial, onSubmit, onClose }: {
  initial: string;
  onSubmit: (comment: string) => void;
  onClose: () => void;
}) {
  const S = useStrings();
  const [text, setText] = useState(initial);
  return (
    <Dialog
      title={S.chat.feedbackTitle}
      onClose={onClose}
      footer={
        <>
          <button className="dp-btn" onClick={onClose}>{S.common.cancel}</button>
          <button className="dp-btn primary" onClick={() => onSubmit(text)}>{S.chat.feedbackSend}</button>
        </>
      }
    >
      <textarea className="feedback-text" autoFocus rows={5} value={text}
                placeholder={S.chat.feedbackPlaceholder}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); onSubmit(text); }
                }} />
    </Dialog>
  );
}
