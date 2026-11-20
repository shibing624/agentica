import { lazy, Suspense, useId, useLayoutEffect, useRef, useState } from "react";
import { useStrings } from "../i18n";
import { IconChevronRight, IconStop } from "../icons";
import { isImeKey } from "../lib/ime";
import { draftAnswer, questionSubmission, type QuestionAnswer, type QuestionDraft, type QuestionRequest } from "../lib/questionComposer";

const ChatMarkdown = lazy(() => import("./ChatMarkdown").then((m) => ({ default: m.ChatMarkdown })));

export function QuestionComposer({ q, busy, stopping, onAnswer, onEngage, onStop }: {
  q: QuestionRequest;
  busy: boolean;
  stopping: boolean;
  onAnswer: (answer: QuestionAnswer) => void;
  onEngage: () => Promise<boolean>;
  onStop: () => void;
}) {
  const S = useStrings().chat;
  const titleId = useId();
  const promptId = useId();
  const hintId = useId();
  const body = useRef<HTMLDivElement>(null);
  const prompt = useRef<HTMLDivElement>(null);
  const engaging = useRef(false);
  const [held, setHeld] = useState(false);
  const [index, setIndex] = useState(0);
  const [drafts, setDrafts] = useState<QuestionDraft[]>(() => q.items.map(() => ({ picked: [], text: "", skipped: false })));
  const [incomplete, setIncomplete] = useState(false);
  const item = q.items[index];
  const draft = drafts[index];
  const last = index === q.items.length - 1;
  const disabled = busy || stopping;
  const waiting = q.engaged || held;
  const placeholder = item.options.length ? S.questionOther : S.questionReply;

  useLayoutEffect(() => {
    if (body.current) body.current.scrollTop = 0;
    prompt.current?.focus({ preventScroll: true });
  }, [index]);

  function engage() {
    if (!q.canEngage || waiting || engaging.current) return;
    engaging.current = true;
    void onEngage().then((ok) => {
      engaging.current = false;
      if (ok) setHeld(true);
    });
  }

  function goTo(i: number) {
    engage();
    setIndex(i);
    setIncomplete(false);
  }

  function update(next: QuestionDraft) {
    engage();
    const updated = drafts.map((d, i) => i === index ? next : d);
    setDrafts(updated);
    setIncomplete(false);
    return updated;
  }

  function advance(next = drafts) {
    if (disabled) return;
    if (!last) {
      goTo(index + 1);
      return;
    }
    const result = questionSubmission(q, next);
    if ("missingIndex" in result) {
      goTo(result.missingIndex);
      setIncomplete(true);
    } else {
      onAnswer(result.answer);
    }
  }

  function choose(option: string) {
    const picked = item.multiSelect
      ? draft.picked.includes(option) ? draft.picked.filter((value) => value !== option) : [...draft.picked, option]
      : [option];
    update({ picked, text: "", skipped: false });
    if (!item.multiSelect && !last) goTo(index + 1);
  }

  return (
    <section className="question-card" aria-labelledby={titleId}>
      <header className="question-head">
        <div className="question-heading">
          <h2 id={titleId}>{S.questionTitle}</h2>
          {item.header && <span className="question-topic">{item.header}</span>}
        </div>
        {q.canEngage && (waiting
          ? <span className="question-wait" role="status">{S.questionWaiting}</span>
          : <button className="question-quiet" type="button" disabled={disabled} onClick={engage}>{S.questionTakeTime}</button>)}
      </header>
      <div className="question-body" ref={body}>
        <div className="question-prompt" id={promptId} tabIndex={-1} ref={prompt}>
          <Suspense fallback={item.prompt}><ChatMarkdown text={item.prompt} /></Suspense>
        </div>
        {item.options.length > 0 && (
          <div className="question-opts" role="group" aria-labelledby={promptId}>
            {item.multiSelect && <span className="question-mode">{S.questionMulti}</span>}
            {item.options.map((opt, i) => {
              const selected = draft.picked.includes(opt);
              return (
                <button key={i} type="button" disabled={disabled} className="question-opt"
                        role={item.multiSelect ? "checkbox" : undefined}
                        aria-checked={item.multiSelect ? selected : undefined}
                        aria-pressed={item.multiSelect ? undefined : selected} onClick={() => choose(opt)}>
                  <span className={"question-opt-marker" + (item.multiSelect ? " checkbox" : "")} aria-hidden="true">
                    {selected ? <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round"><path d="m4 8 2.6 2.6L12 5.2" /></svg>
                      : item.multiSelect ? null : i + 1}
                  </span>
                  <span>{opt}</span>
                </button>
              );
            })}
          </div>
        )}
        <div className="question-field">
          <div aria-hidden="true">{draft.text + "\n"}</div>
          <textarea rows={2} value={draft.text} aria-label={placeholder} aria-describedby={hintId}
                    placeholder={placeholder} disabled={disabled}
                    onPointerDown={engage}
                    onChange={(e) => update({ text: e.currentTarget.value, picked: [], skipped: false })}
                    onKeyDown={(e) => {
                      if (e.key !== "Enter" || e.shiftKey || isImeKey(e.nativeEvent) || !draftAnswer(draft)) return;
                      e.preventDefault();
                      advance();
                    }} />
        </div>
        <p className="question-hint" id={hintId}>{S.questionNewline}</p>
      </div>
      <footer className="question-footer">
        {q.items.length > 1 && (
          <div className="question-pager">
            <button type="button" className="question-icon question-prev" title={S.questionPrevious} aria-label={S.questionPrevious}
                    onClick={() => goTo(index - 1)} disabled={disabled || index === 0}><IconChevronRight /></button>
            <span role="status">{index + 1} / {q.items.length}</span>
            <button type="button" className="question-icon" title={S.questionNext} aria-label={S.questionNext}
                    onClick={() => goTo(index + 1)} disabled={disabled || last}><IconChevronRight /></button>
          </div>
        )}
        {incomplete && <span className="question-feedback" role="status">{S.questionIncomplete}</span>}
        {!incomplete && draft.skipped && <span className="question-feedback">{S.questionSkipped}</span>}
        <div className="question-actions">
          {q.batch && <button type="button" className="question-quiet" disabled={disabled}
                               onClick={() => advance(update({ picked: [], text: "", skipped: true }))}>{S.questionSkip}</button>}
          <button type="button" className="question-primary" disabled={disabled || (!draft.skipped && !draftAnswer(draft))}
                  onClick={() => advance()}>{busy ? S.questionSending : last ? S.questionSend : S.questionContinue}</button>
          <button type="button" className="question-icon question-stop" disabled={disabled}
                  title={S.stop} aria-label={S.stop} onClick={onStop}><IconStop /></button>
        </div>
      </footer>
    </section>
  );
}