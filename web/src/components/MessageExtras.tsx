import { lazy, Suspense, useEffect, useState } from "react";
import * as api from "../api";
import { useStrings } from "../i18n";
import { IconBook, IconCopy } from "../icons";
import { showToast } from "../store";
import { Dialog } from "./Dialog";

const ChatMarkdown = lazy(() => import("./ChatMarkdown").then((m) => ({ default: m.ChatMarkdown })));

/** `/skill args` in a user bubble: the skill name underlined, then the argument.
 *  Click the name to read its SKILL.md, instead of the whole prompt in the bubble.
 *
 *  `source` is the original user text. Recognition is deliberately strict, but
 *  a pasted document can still look like a skill header — keep the text one
 *  click away so a false positive never hides or loses what the user typed. */
export function SkillTurn({ name, args, source }: { name: string; args: string; source?: string }) {
  const S = useStrings();
  const [open, setOpen] = useState(false);
  const [showSource, setShowSource] = useState(false);
  return (
    <>
      <button type="button" className="skill-chip" title={S.chat.skillChipTip} onClick={() => setOpen(true)}>
        <IconBook />
        <span>{name}</span>
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
      {open && <SkillPreview name={name} onClose={() => setOpen(false)} />}
    </>
  );
}

function SkillPreview({ name, onClose }: { name: string; onClose: () => void }) {
  const S = useStrings();
  const [doc, setDoc] = useState<{ content: string; path: string; description: string } | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let cancelled = false;
    void api.fetchSkillDetail(name).then(({ ok, data }) => {
      if (cancelled) return;
      if (!ok || !data) { setFailed(true); return; }
      setDoc({
        content: String(data.content || ""),
        path: String(data.path || ""),
        description: String(data.description || ""),
      });
    });
    return () => { cancelled = true; };
  }, [name]);
  return (
    <Dialog title={S.chat.skillPreviewTitle(name)} onClose={onClose} wide>
      {failed && <div className="skill-preview-empty">{S.chat.skillLoadFailed}</div>}
      {!doc && !failed && <div className="skill-preview-empty">…</div>}
      {doc && (
        <div className="skill-preview">
          {doc.path && (
            <div className="skill-preview-path">
              <code>{doc.path}/SKILL.md</code>
              <button type="button" title={S.common.copy}
                      onClick={() => { void navigator.clipboard.writeText(`${doc.path}/SKILL.md`).then(() => showToast(S.common.copied)); }}>
                <IconCopy />
              </button>
            </div>
          )}
          {doc.description && <p className="skill-preview-desc">{doc.description}</p>}
          <div className="skill-preview-body">
            <Suspense fallback={<pre>{doc.content}</pre>}>
              <ChatMarkdown text={doc.content} streaming={false} />
            </Suspense>
          </div>
        </div>
      )}
    </Dialog>
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
