/**
 * One round's edits, dsh's review tab: a file selector (every file the reply
 * changed, with its +/- counts) over the selected file's diff, drawn like
 * dsh's FileDiff: a hunk header per
 * hunk, old/new line numbers, red/green rows with a marker bar, unified or
 * side-by-side, wrap on or off. Every tool call that touched the file is one
 * group of hunks, in the order they ran.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import * as api from "../api";
import { useStrings } from "../i18n";
import { IconChevronDown, IconExternal, IconFileCode, IconFinder, IconSplit, IconWrap } from "../icons";
import {
  diffCounts, hunkHeader, hunkRows, oneSided, renderedEdits, splitRows, MAX_RENDERED_LINES, type DiffHunk, type FileEdits,
} from "../lib/fileDiff";
import { showToast, useAppState } from "../store";
import { CommitBar } from "./CommitBar";

const SPLIT_KEY = "ag_diff_split";
const WRAP_KEY = "ag_diff_wrap";

function useStoredFlag(key: string): [boolean, () => void] {
  const [on, setOn] = useState(() => localStorage.getItem(key) === "1");
  const toggle = () => setOn((v) => {
    localStorage.setItem(key, v ? "0" : "1");
    return !v;
  });
  return [on, toggle];
}

function Hunk({ hunk, split }: { hunk: DiffHunk; split: boolean }) {
  return (
    <section className="fd-hunk">
      <div className="fd-hunk-head">{hunkHeader(hunk)}</div>
      {split
        ? splitRows(hunk).map((row, i) => (
          <div key={i} className="fd-split">
            {[row.left, row.right].map((cell, side) => (
              <span key={side} className={"fd-cell " + (cell ? "fd-" + cell.kind : "fd-empty")}>
                <span className="fd-no">{cell?.no ?? ""}</span>
                <span className="fd-text">{cell?.text ?? ""}</span>
              </span>
            ))}
          </div>
        ))
        : hunkRows(hunk).map((row, i) => (
          <div key={i} className={"fd-line fd-" + row.kind}>
            <span className="fd-no">{row.old ?? ""}</span>
            <span className="fd-no">{row.new ?? ""}</span>
            <span className="fd-sign">{row.kind === "add" ? "+" : row.kind === "del" ? "-" : " "}</span>
            <span className="fd-text">{row.text}</span>
          </div>
        ))}
    </section>
  );
}

function Counts({ edits }: { edits: DiffHunk[][] }) {
  const { added, removed } = diffCounts(edits);
  return (
    <span className="mfc-diff">
      <span className="add">+{added.toLocaleString()}</span>
      <span className="del">-{removed.toLocaleString()}</span>
    </span>
  );
}

/** dsh's review selector: the selected file's path, a menu of every changed file. */
function FileSelect({ files, path, onSelect }: { files: FileEdits[]; path: string; onSelect: (path: string) => void }) {
  const S = useStrings();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false); };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("mousedown", onDown); document.removeEventListener("keydown", onKey); };
  }, [open]);
  return (
    <div className="rv-select" ref={ref}>
      <button type="button" className={"rv-select-btn" + (open ? " on" : "")} aria-haspopup="menu" aria-expanded={open}
              title={path} aria-label={S.files.reviewSelectFile} onClick={() => setOpen((v) => !v)}>
        <span className="rv-select-path"><bdi dir="ltr">{path}</bdi></span>
        <IconChevronDown />
      </button>
      {open && (
        <div className="rv-menu" role="menu">
          {files.map((f) => (
            <button key={f.path} type="button" role="menuitemradio" aria-checked={f.path === path}
                    className={"rv-item" + (f.path === path ? " on" : "")} title={f.path}
                    onClick={() => { onSelect(f.path); setOpen(false); }}>
              <span className="rv-item-path">{f.path}</span>
              <Counts edits={f.edits} />
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function DiffView({
  root, files, path, onSelect, onOpenFile,
}: {
  root: string;
  turn: number;
  files: FileEdits[];
  path: string;
  onSelect: (path: string) => void;
  onOpenFile: (path: string) => void;
}) {
  const S = useStrings();
  const D = S.files;
  const canGit = useAppState().accountRole === "admin";
  const [splitPref, toggleSplit] = useStoredFlag(SPLIT_KEY);
  const [wrap, toggleWrap] = useStoredFlag(WRAP_KEY);
  const edits = (files.find((f) => f.path === path) ?? files[0])?.edits ?? [];
  // Only additions (a new file) or only deletions: one column, whatever was picked.
  const split = splitPref && !oneSided(edits);
  const shown = useMemo(() => renderedEdits(edits), [edits]);
  const abs = root.replace(/\/+$/, "") + "/" + path;
  const open = (app: "finder" | "reveal") => {
    void (app === "reveal" ? api.revealPathApi(abs) : api.openPathApi(abs, "finder"))
      .then(({ ok }) => { if (!ok) showToast(S.chat.openFailed, 2500); });
  };
  return (
    <div className="ws-browser fd-root">
      <div className="ws-preview-bar">
        {files.length > 1
          ? <FileSelect files={files} path={path} onSelect={onSelect} />
          : <span className="ws-preview-name rv-single" title={path}>{path}</span>}
        <span className="rv-counts"><Counts edits={edits} /></span>
        <div className="pv-actions">
          {canGit && <CommitBar root={root} files={files} />}
          <button type="button" className={"pv-icon" + (splitPref ? " on" : "")} aria-pressed={splitPref}
                  title={splitPref ? D.diffUnified : D.diffSplit} aria-label={D.diffSplit} onClick={toggleSplit}>
            <IconSplit />
          </button>
          <button type="button" className={"pv-icon" + (wrap ? " on" : "")} aria-pressed={wrap}
                  title={wrap ? D.diffNoWrap : D.diffWrap} aria-label={D.diffWrap} onClick={toggleWrap}>
            <IconWrap />
          </button>
          <button type="button" className="pv-icon" title={D.diffOpenFile} aria-label={D.diffOpenFile}
                  onClick={() => onOpenFile(path)}>
            <IconFileCode />
          </button>
          <button type="button" className="pv-icon" title={S.chat.openExternal} aria-label={S.chat.openExternal}
                  onClick={() => open("finder")}>
            <IconExternal />
          </button>
          <button type="button" className="pv-icon" title={S.chat.revealInFolder} aria-label={S.chat.revealInFolder}
                  onClick={() => open("reveal")}>
            <IconFinder />
          </button>
        </div>
      </div>
      <div className={"fd-body" + (wrap ? " wrap" : "")} data-view={split ? "split" : "unified"}>
        {shown.truncated && <p className="fd-note">{D.diffTruncated(MAX_RENDERED_LINES)}</p>}
        {shown.edits.map((hunks, e) => (
          <div key={e} className="fd-edit">
            {shown.edits.length > 1 && <div className="fd-edit-head">{D.diffEdit(e + 1, shown.edits.length)}</div>}
            {hunks.map((h, i) => <Hunk key={i} hunk={h} split={split} />)}
          </div>
        ))}
      </div>
    </div>
  );
}
