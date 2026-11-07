/**
 * Commit / commit+push a round's edits from the review tab (Codex app style):
 * a split button, then a dialog that lists the round's files (all checked),
 * drafts the message on the side-task model, and shows where a push goes.
 * Only the checked files are committed; anything else dirty stays out.
 */
import { useEffect, useRef, useState } from "react";
import * as api from "../api";
import { useStrings } from "../i18n";
import { Dialog } from "../components/Dialog";
import { IconChevronDown } from "../icons";
import type { FileEdits } from "../lib/fileDiff";
import { showToast } from "../store";

function detail(res: api.ApiResult): string {
  const d = res.data as { detail?: unknown } | null;
  return typeof d?.detail === "string" ? d.detail : `HTTP ${res.status}`;
}

function CommitDialog({
  root, files, push: initialPush, onClose,
}: { root: string; files: FileEdits[]; push: boolean; onClose: () => void }) {
  const S = useStrings();
  const G = S.git;
  const [status, setStatus] = useState<api.GitStatus | null>(null);
  const [statusError, setStatusError] = useState("");
  const [picked, setPicked] = useState<Set<string>>(() => new Set(files.map((f) => f.path)));
  const [message, setMessage] = useState("");
  const [drafting, setDrafting] = useState(false);
  const [busy, setBusy] = useState(false);
  const [push, setPush] = useState(initialPush);
  const draftedOnce = useRef(false);
  const all = files.map((f) => f.path);

  useEffect(() => {
    let alive = true;
    void api.gitStatusApi(root, all).then((res) => {
      if (!alive) return;
      if (res.ok && res.data) setStatus(res.data);
      else setStatusError(detail(res));
    });
    return () => { alive = false; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [root]);

  // Paths from the round are relative to the session dir; git reports them
  // relative to the repo root. Match on the suffix.
  const matches = (list: string[] | undefined, p: string) => (list ?? []).some((c) => c === p || c.endsWith("/" + p));
  const isChanged = (p: string) => !status || matches(status.changed, p);
  const isIgnored = (p: string) => !!status && matches(status.ignored, p);
  const chosen = all.filter((p) => picked.has(p) && isChanged(p));

  const draft = async () => {
    if (!chosen.length) return;
    setDrafting(true);
    const res = await api.gitMessageApi(root, chosen);
    setDrafting(false);
    if (res.ok && res.data?.message) setMessage(res.data.message);
    else showToast(G.draftFailed(detail(res)), 4000);
  };

  // Draft once the status says which files actually have changes.
  useEffect(() => {
    if (!status || draftedOnce.current || !chosen.length) return;
    draftedOnce.current = true;
    void draft();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  const submit = async () => {
    if (!chosen.length || !message.trim() || busy) return;
    setBusy(true);
    const res = await api.gitCommitApi(root, chosen, message, push);
    setBusy(false);
    if (!res.ok || !res.data) {
      showToast(G.commitFailed(detail(res)), 5000);
      return;
    }
    const { sha, pushed, pushError } = res.data;
    if (push && !pushed) showToast(G.pushFailed(sha, pushError || ""), 6000);
    else showToast(pushed ? G.pushed(sha) : G.committed(sha), 3000);
    onClose();
  };

  const close = () => { if (!busy) onClose(); };
  return (
    <Dialog
      wide
      title={G.title}
      onClose={close}
      footer={
        <>
          <button type="button" className="dp-btn" onClick={close} disabled={busy}>{S.common.cancel}</button>
          <button type="button" className="dp-btn primary" disabled={busy || !chosen.length || !message.trim()}
                  onClick={() => void submit()}>
            {busy ? G.working : push ? G.commitPush : G.commit}
          </button>
        </>
      }
    >
      <div className="git-dialog" onKeyDown={(e) => {
        if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); void submit(); }
      }}>
        {status && (
          <div className="git-branch" title={status.root}>
            {status.branch || G.detached}
            {status.upstream && <span className="git-upstream"> → {status.upstream}</span>}
          </div>
        )}
        {statusError && <p className="git-error">{statusError}</p>}
        {status && !chosen.length && !statusError && (
          <p className="git-warn">{files.every((f) => isIgnored(f.path)) ? G.allIgnored : G.nothingToCommit}</p>
        )}
        <div className="git-files">
          {files.map((f) => {
            const has = isChanged(f.path);
            return (
              <label key={f.path} className={"git-file" + (has ? "" : " clean")}
                     title={has ? f.path : isIgnored(f.path) ? G.ignored : G.noChanges}>
                <input type="checkbox" checked={has && picked.has(f.path)} disabled={!has}
                       onChange={(e) => setPicked((prev) => {
                         const next = new Set(prev);
                         if (e.target.checked) next.add(f.path); else next.delete(f.path);
                         return next;
                       })} />
                <span className="git-file-path">{f.path}</span>
                {!has && <span className="git-file-tag">{isIgnored(f.path) ? G.ignoredTag : G.cleanTag}</span>}
              </label>
            );
          })}
        </div>
        <div className="git-msg">
          <textarea value={message} onChange={(e) => setMessage(e.target.value)} rows={5}
                    placeholder={drafting ? G.drafting : G.messagePlaceholder} disabled={busy} autoFocus />
          <button type="button" className="git-draft" disabled={drafting || busy || !chosen.length}
                  onClick={() => void draft()}>
            {drafting ? G.drafting : G.draft}
          </button>
        </div>
        <label className="git-push">
          <input type="checkbox" checked={push} onChange={(e) => setPush(e.target.checked)} disabled={busy} />
          <span>{status?.upstream ? G.pushTo(status.upstream, status.ahead) : G.pushNew(status?.branch || "")}</span>
        </label>
        {push && status && status.behind > 0 && <p className="git-warn">{G.behind(status.behind)}</p>}
      </div>
    </Dialog>
  );
}

/** The review header's split button: Commit, with Commit and Push in the menu. */
export function CommitBar({ root, files }: { root: string; files: FileEdits[] }) {
  const S = useStrings();
  const G = S.git;
  const [menu, setMenu] = useState(false);
  const [dialog, setDialog] = useState<null | { push: boolean }>(null);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!menu) return;
    const onDown = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setMenu(false); };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [menu]);
  return (
    <>
      <div className="git-split" ref={ref}>
        <button type="button" className="git-split-main" onClick={() => setDialog({ push: false })}>{G.commit}</button>
        <button type="button" className={"git-split-arrow" + (menu ? " on" : "")} aria-haspopup="menu" aria-expanded={menu}
                aria-label={G.more} onClick={() => setMenu((v) => !v)}>
          <IconChevronDown />
        </button>
        {menu && (
          <div className="rv-menu git-menu" role="menu">
            <button type="button" role="menuitem" className="rv-item" onClick={() => { setMenu(false); setDialog({ push: false }); }}>
              {G.commit}
            </button>
            <button type="button" role="menuitem" className="rv-item" onClick={() => { setMenu(false); setDialog({ push: true }); }}>
              {G.commitPush}
            </button>
          </div>
        )}
      </div>
      {dialog && <CommitDialog root={root} files={files} push={dialog.push} onClose={() => setDialog(null)} />}
    </>
  );
}
