import { useEffect, useState } from "react";
import * as api from "../api";
import { useStrings } from "../i18n";
import { fmtCost, fmtN } from "../lib/format";
import { bump, getState, saveSessions, type Session } from "../store";

export type UsageSection = {
  label: string;
  tokens: number;
  share: number;
};

export type SessionUsage = {
  model: string;
  /** The budget occupancy is judged against: the compact cap when set. */
  window: number;
  context_tokens: number;
  percent_full: number;
  messages: number;
  api_calls: number;
  cost_usd: number;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  cache_hit_percent: number | null;
  sections: UsageSection[];
};

export function applySessionUsage(sessionId: string, usage: SessionUsage) {
  const sess = getState().sessions[sessionId];
  if (sess) {
    sess.contextTokens = usage.context_tokens;
    sess.costUsd = usage.cost_usd;
    saveSessions();
  }
  // The ring divides occupancy by this, and occupancy is measured against the
  // working budget — so the working budget is what belongs here. Publishing the
  // provider limit would put the two halves of the ratio in different units.
  if (usage.window) getState().serverContextWindow = usage.window;
  bump();
}

export function ContextUsageTip({ sessionId, fallback }: {
  sessionId: string | null;
  fallback: Session | null;
}) {
  const S = useStrings();
  const [usage, setUsage] = useState<SessionUsage | null>(null);
  const [loading, setLoading] = useState(!!sessionId);

  useEffect(() => {
    if (!sessionId) {
      setLoading(false);
      return;
    }
    let cancelled = false;
    setLoading(true);
    void api.fetchSessionUsage(sessionId).then(({ ok, data }) => {
      if (cancelled) return;
      setLoading(false);
      if (ok && data) {
        setUsage(data as SessionUsage);
        applySessionUsage(sessionId, data as SessionUsage);
      }
    });
    return () => { cancelled = true; };
  }, [sessionId]);

  // The ratio is against the working budget (the cap when set), because that
  // is the one that fills and triggers compaction. /config and /usage are
  // where the model's hard limit is spelled out.
  const windowSize = usage?.window || getState().serverContextWindow || 128000;
  const contextTokens = usage?.context_tokens ?? fallback?.contextTokens ?? fallback?.lastInputTokens ?? 0;
  const messages = usage?.messages ?? fallback?.msgs.length ?? 0;
  const apiCalls = usage?.api_calls ?? fallback?.requests ?? 0;
  const cost = usage?.cost_usd ?? fallback?.costUsd ?? 0;
  const inputTokens = usage?.input_tokens ?? fallback?.tokIn ?? 0;
  const outputTokens = usage?.output_tokens ?? fallback?.tokOut ?? 0;
  const sections = (usage?.sections || []).filter((sec) => sec.tokens > 0);
  const pct = Math.min(100, Math.round((contextTokens / windowSize) * 100));
  // Bar length is the occupancy; sections split it in proportion.
  const sectionTotal = sections.reduce((n, sec) => n + sec.tokens, 0);
  const segments = sectionTotal > 0
    ? sections.map((sec, i) => ({ key: sec.label, tint: SECTION_TINTS[i % SECTION_TINTS.length], width: pct * sec.tokens / sectionTotal }))
    : [{ key: "total", tint: undefined, width: pct }];

  return (
    <div className="ctx-tip" role="dialog" aria-label={S.chat.ctxWindowTitle}>
      <div className="ctx-tip-head">
        <span className="ctx-tip-headline">{S.chat.ctxUsed}</span>
        <span className="ctx-tip-pct">{pct}%</span>
        <span className="ctx-tip-figures">~{fmtN(contextTokens)} / {fmtN(windowSize)}</span>
      </div>
      <div className="ctx-bar">
        {segments.filter((seg) => seg.width > 0).map((seg) => (
          <div key={seg.key} className="ctx-bar-seg"
               style={{ width: `${seg.width}%`, ...(seg.tint ? { ["--tint" as any]: seg.tint } : {}) }} />
        ))}
      </div>
      {loading && !usage && <div className="ctx-tip-muted">{S.common.loading}</div>}
      {sections.length > 0 && (
        <dl className="ctx-secs">
          {sections.map((sec, i) => (
            <div className="ctx-sec" key={sec.label}>
              <dt className="ctx-sec-label">
                <span className="ctx-swatch" style={{ ["--tint" as any]: SECTION_TINTS[i % SECTION_TINTS.length] }} />
                {S.chat.ctxSection(sec.label)}
              </dt>
              <dd className="ctx-sec-n">~{fmtN(sec.tokens)}</dd>
            </div>
          ))}
        </dl>
      )}
      <div className="ctx-tip-title">{S.chat.ctxSessionUsage}</div>
      <dl className="ctx-details">
        <dt>{S.chat.ctxMessages}</dt><dd>{messages}</dd>
        <dt>{S.chat.ctxApiCalls}</dt><dd>{apiCalls}</dd>
        <dt>{S.chat.ctxInput}</dt><dd>{fmtN(inputTokens)}</dd>
        {!!usage?.cache_read_tokens && (
          <>
            <dt>{S.chat.ctxCached}</dt>
            <dd>
              {fmtN(usage.cache_read_tokens)}
              {usage.cache_hit_percent != null ? ` / ${usage.cache_hit_percent.toFixed(1)}%` : ""}
            </dd>
          </>
        )}
        {!!usage?.cache_write_tokens && (
          <><dt>{S.chat.ctxCacheWrite}</dt><dd>{fmtN(usage.cache_write_tokens)}</dd></>
        )}
        <dt>{S.chat.ctxOutput}</dt><dd>{fmtN(outputTokens)}</dd>
        <dt>{S.chat.ctxCost}</dt><dd>~{fmtCost(cost)}</dd>
      </dl>
      {apiCalls === 0 && !loading && (
        <div className="ctx-tip-muted">{S.chat.ctxNoCalls}</div>
      )}
    </div>
  );
}

const SECTION_TINTS = ["#8b9bb4", "#a78bfa", "#f59e0b", "#34d399", "#60a5fa", "#f472b6", "#2dd4bf", "#3b82f6"];
