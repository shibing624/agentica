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

  const windowSize = usage?.window || getState().serverContextWindow || 128000;
  const contextTokens = usage?.context_tokens ?? fallback?.contextTokens ?? fallback?.lastInputTokens ?? 0;
  const messages = usage?.messages ?? fallback?.msgs.length ?? 0;
  const apiCalls = usage?.api_calls ?? fallback?.requests ?? 0;
  const cost = usage?.cost_usd ?? fallback?.costUsd ?? 0;
  const inputTokens = usage?.input_tokens ?? fallback?.tokIn ?? 0;
  const outputTokens = usage?.output_tokens ?? fallback?.tokOut ?? 0;
  const sections = usage?.sections || [];

  return (
    <div className="ctx-tip">
      <div className="ctx-tip-header">
        {S.chat.ctxWindowTitle}
        <span className="ctx-tip-sub">
          {fmtN(contextTokens)} / {fmtN(windowSize)} tok
        </span>
      </div>
      {loading && !usage && <div className="ctx-tip-muted">{S.common.loading}</div>}
      <div className="ctx-tip-row"><span>{S.chat.ctxMessages}</span><span>{messages}</span></div>
      <div className="ctx-tip-row"><span>{S.chat.ctxApiCalls}</span><span>{apiCalls}</span></div>
      <div className="ctx-tip-row"><span>{S.chat.ctxCost}</span><span>~{fmtCost(cost)}</span></div>
      <div className="ctx-tip-row"><span>{S.chat.ctxInput}</span><span>{fmtN(inputTokens)}</span></div>
      {!!usage?.cache_read_tokens && (
        <div className="ctx-tip-row">
          <span>{S.chat.ctxCached}</span>
          <span>
            {fmtN(usage.cache_read_tokens)}
            {usage.cache_hit_percent != null ? ` / ${usage.cache_hit_percent.toFixed(1)}%` : ""}
          </span>
        </div>
      )}
      <div className="ctx-tip-row ctx-tip-total"><span>{S.chat.ctxOutput}</span><span>{fmtN(outputTokens)}</span></div>
      {apiCalls === 0 && !loading && (
        <div className="ctx-tip-muted">{S.chat.ctxNoCalls}</div>
      )}
      {sections.length > 0 && (
        <div className="ctx-secs">
          {sections.map((sec) => (
            <div className="ctx-sec" key={sec.label}>
              <span className="ctx-sec-label">{S.chat.ctxSection(sec.label)}</span>
              <span className="ctx-sec-n">{fmtN(sec.tokens)} tok</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
