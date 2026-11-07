/**
 * SKILL.md of an installed skill in the right-hand panel, opened from a skill
 * reference in the composer or a user bubble. Same toolbar and views as a
 * Markdown file in the workspace browser.
 */
import { lazy, Suspense, useCallback, useEffect, useState } from "react";
import * as api from "../api";
import { CodeFrame } from "../components/CodeFrame";
import { useStrings } from "../i18n";
import { HIGHLIGHT_LIMIT } from "../lib/highlight";
import { PlainText, PreviewActions, type ViewMode } from "./PreviewActions";

const ChatMarkdown = lazy(() => import("../components/ChatMarkdown").then((m) => ({ default: m.ChatMarkdown })));

type SkillDoc = {
  name: string;
  description: string;
  path: string;
  content: string;
  trigger: string;
  argumentHint: string;
  location: string;
};

const MODES: ViewMode[] = ["markdown", "code", "text"];

export function SkillPane({ name }: { name: string }) {
  const S = useStrings();
  const [doc, setDoc] = useState<SkillDoc | null>(null);
  const [failed, setFailed] = useState(false);
  const [mode, setMode] = useState<ViewMode>("markdown");
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    const { ok, data } = await api.fetchSkillDetail(name);
    if (!ok || !data) return null;
    return {
      name: String(data.name || name),
      description: String(data.description || ""),
      path: String(data.path || ""),
      content: String(data.content || ""),
      trigger: String(data.trigger || ""),
      argumentHint: String(data.argument_hint || ""),
      location: String(data.location || ""),
    };
  }, [name]);

  useEffect(() => {
    let cancelled = false;
    setDoc(null);
    setFailed(false);
    setMode("markdown");
    void load().then((d) => {
      if (cancelled) return;
      if (d) setDoc(d); else setFailed(true);
    });
    return () => { cancelled = true; };
  }, [load]);

  const refresh = () => {
    setRefreshing(true);
    void load().then((d) => {
      if (d) { setDoc(d); setFailed(false); } else setFailed(true);
    }).finally(() => setRefreshing(false));
  };

  const file = doc?.path ? `${doc.path}/SKILL.md` : "";
  const fields: Array<[string, string]> = doc
    ? ([
        ["name", doc.name],
        ["description", doc.description],
        ["trigger", doc.trigger],
        ["argument", doc.argumentHint],
        ["location", doc.location],
      ] as Array<[string, string]>).filter(([, v]) => v)
    : [];

  return (
    <div className="ws-browser">
      <div className="ws-preview-bar">
        <span className="ws-preview-name" title={file || name}>/{name}</span>
        <PreviewActions
          modes={MODES}
          mode={mode}
          onMode={setMode}
          onRefresh={refresh}
          refreshing={refreshing}
          absPath={file || undefined}
        />
      </div>
      <div className={"ws-preview-body" + (mode === "code" && doc ? " source" : "")}>
        {failed && <p className="ws-muted">{S.chat.skillLoadFailed}</p>}
        {!doc && !failed && <p className="ws-muted">{S.common.loading}</p>}
        {doc && mode === "markdown" && (
          <div className="md-doc">
            {fields.length > 0 && (
              <dl className="skill-pane-meta">
                {fields.map(([k, v]) => (
                  <div key={k} className="skill-pane-field">
                    <dt>{k}</dt>
                    <dd>{v}</dd>
                  </div>
                ))}
              </dl>
            )}
            <Suspense fallback={<pre>{doc.content}</pre>}>
              <ChatMarkdown text={doc.content} streaming={false} />
            </Suspense>
          </div>
        )}
        {doc && mode === "code" && (
          <CodeFrame code={doc.content} language="markdown" highlight={doc.content.length <= HIGHLIGHT_LIMIT} />
        )}
        {doc && mode === "text" && <PlainText text={doc.content} />}
      </div>
    </div>
  );
}
