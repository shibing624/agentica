import { useEffect, useState } from "react";
import * as api from "../api";
import { Dialog } from "../components/Dialog";
import { getStrings, useStrings, type Strings } from "../i18n";
import { showToast } from "../store";

type EnvVar = { name: string; example: string };
type ChannelRow = {
  id: string;
  extra: string | null;
  env: EnvVar[];
  recommended: boolean;
  docs_anchor: string;
  configured: boolean;
  connected: boolean;
};
type ChannelsPayload = {
  web_url: string;
  listen: { host: string; port: number; loopback: boolean };
  guide_url: string;
  catalog: ChannelRow[];
};
type QrState = { id: string; png: string; expiresAt: number };

/** Web URL + IM how-to. Used by the settings tab and the `/assistant` page. */
export function AssistantPanel() {
  const S = useStrings();
  const [data, setData] = useState<ChannelsPayload | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  const reload = () =>
    api.fetchChannels().then(({ ok, data: body }) => {
      if (ok && body) setData(body as ChannelsPayload);
    });

  useEffect(() => {
    void api.fetchChannels().then(({ ok, data: body }) => {
      if (ok && body) setData(body as ChannelsPayload);
      else setFailed(true);
    });
  }, []);

  if (failed) return <p className="muted">{S.settings.readFailed}</p>;
  if (!data) return <p className="muted">{S.common.loading}</p>;

  const web = data.catalog.find((c) => c.id === "web");
  const im = data.catalog.filter((c) => c.id !== "web");
  const how = openId ? data.catalog.find((c) => c.id === openId) : null;

  return (
    <>
      <p className="asst-intro">{S.assistant.subtitle}</p>

      <div className="asst-section-title">{S.assistant.webSection}</div>
      {web && (
        <WebCard
          url={data.web_url}
          listen={data.listen}
          guideUrl={data.guide_url + web.docs_anchor}
        />
      )}

      <div className="asst-section-title">{S.assistant.imSection}</div>
      <p className="asst-im-hint">{S.assistant.imHint}</p>
      <div className="asst-list">
        {im.map((ch) => ch.id === "wechat" ? (
          <WeChatRow
            key={ch.id}
            ch={ch}
            guideUrl={data.guide_url + ch.docs_anchor}
            onRefresh={() => void reload()}
          />
        ) : (
          <ImRow
            key={ch.id}
            ch={ch}
            guideUrl={data.guide_url + ch.docs_anchor}
            onConfigure={() => setOpenId(ch.id)}
          />
        ))}
      </div>
      {how && (
        <HowToDialog
          ch={how}
          guideUrl={data.guide_url + how.docs_anchor}
          onClose={() => setOpenId(null)}
        />
      )}
    </>
  );
}

export function AssistantPage() {
  const S = useStrings();
  return (
    <div className="main">
      <div className="topbar">
        <h3 className="page-title">{S.assistant.title}</h3>
      </div>
      <div className="page-body asst-body">
        <AssistantPanel />
      </div>
    </div>
  );
}

function WebCard({
  url, listen, guideUrl,
}: {
  url: string; listen: ChannelsPayload["listen"]; guideUrl: string;
}) {
  const S = useStrings();
  const copy = S.assistant.channels.web;
  const open = async () => {
    const { ok, data } = await api.openUrlApi(url);
    if (!ok) showToast((data as any)?.detail || S.assistant.openUrlFailed, 3000);
  };
  return (
    <div className="asst-row">
      <div className="asst-row-main">
        <div className="asst-row-head">
          <span className="asst-name">{copy.title}</span>
          <span className="asst-badge on">{S.assistant.on}</span>
        </div>
        <div className="asst-desc">{copy.desc}</div>
        <div className="asst-url-row">
          <button type="button" className="config-path-link asst-url-link" onClick={() => void open()} title={url}>
            {url}
          </button>
          <button className="cron-act" onClick={() => void copyText(url)}>{S.common.copy}</button>
        </div>
        <div className="asst-meta">
          {S.assistant.bind(listen.host, listen.port)}
          {" · "}
          {listen.loopback ? S.assistant.loopbackHint : S.assistant.lanHint(listen.host, listen.port)}
        </div>
      </div>
      <div className="asst-row-actions">
        <a className="asst-guide" href={guideUrl} target="_blank" rel="noreferrer">{S.assistant.guide}</a>
      </div>
    </div>
  );
}

function WeChatRow({
  ch, guideUrl, onRefresh,
}: {
  ch: ChannelRow; guideUrl: string; onRefresh: () => void;
}) {
  const S = useStrings();
  const copy = channelCopy(S, ch.id);
  const badge = statusBadge(S, ch);
  const [qr, setQr] = useState<QrState | null>(null);
  const [left, setLeft] = useState(0);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!qr) return;
    const tick = () => {
      const n = Math.max(0, Math.ceil((qr.expiresAt - Date.now()) / 1000));
      setLeft(n);
      if (n <= 0) setQr(null);
    };
    tick();
    const t = window.setInterval(tick, 1000);
    return () => window.clearInterval(t);
  }, [qr]);

  useEffect(() => {
    if (!qr) return;
    const id = qr.id;
    let cancelled = false;
    const poll = async () => {
      const { ok, data } = await api.pollWechatQrApi(id);
      if (cancelled || !ok || !data) return;
      if (data.status === "confirmed") {
        setQr(null);
        onRefresh();
        return;
      }
      if (data.status === "expired") setQr(null);
    };
    const t = window.setInterval(() => void poll(), 2000);
    return () => {
      cancelled = true;
      window.clearInterval(t);
    };
  }, [qr?.id]);

  const start = async () => {
    setBusy(true);
    const { ok, data } = await api.startWechatQrApi();
    setBusy(false);
    if (!ok || !data) {
      showToast((data as any)?.detail || S.settings.setFailed, 3000);
      return;
    }
    if (data.status === "connected") {
      setQr(null);
      onRefresh();
      return;
    }
    if (data.status === "pending" && data.qrcode && data.png) {
      const ttl = Math.max(1, data.expires_in || 120);
      setQr({ id: data.qrcode, png: data.png, expiresAt: Date.now() + ttl * 1000 });
      setLeft(ttl);
    }
  };

  return (
    <div className="asst-row asst-row-wechat">
      <div className="asst-wechat-top">
        <div className="asst-row-main">
          <div className="asst-row-head">
            <span className="asst-name">{copy.title}</span>
            {ch.recommended && <span className="asst-badge rec">{S.assistant.recommended}</span>}
            {badge && <span className={"asst-badge " + badge.kind}>{badge.label}</span>}
          </div>
          <div className="asst-desc">{copy.desc}</div>
          <a className="asst-guide" href={guideUrl} target="_blank" rel="noreferrer">{S.assistant.guide}</a>
        </div>
        <div className="asst-row-actions">
          <button className="cron-act" onClick={() => void start()} disabled={busy}>
            {busy ? S.assistant.qrBusy : S.assistant.configure}
          </button>
        </div>
      </div>
      {qr && (
        <div className="asst-qr">
          <div className="asst-qr-title">{S.assistant.qrScan}</div>
          <img className="asst-qr-img" src={"data:image/png;base64," + qr.png} alt="" />
          <div className="asst-qr-ttl">{S.assistant.qrExpires(left)}</div>
        </div>
      )}
    </div>
  );
}

function ImRow({
  ch, guideUrl, onConfigure,
}: {
  ch: ChannelRow; guideUrl: string; onConfigure: () => void;
}) {
  const S = useStrings();
  const copy = channelCopy(S, ch.id);
  const badge = statusBadge(S, ch);
  return (
    <div className="asst-row">
      <div className="asst-row-main">
        <div className="asst-row-head">
          <span className="asst-name">{copy.title}</span>
          {ch.recommended && <span className="asst-badge rec">{S.assistant.recommended}</span>}
          {badge && <span className={"asst-badge " + badge.kind}>{badge.label}</span>}
        </div>
        <div className="asst-desc">{copy.desc}</div>
      </div>
      <div className="asst-row-actions">
        <a className="asst-guide" href={guideUrl} target="_blank" rel="noreferrer">{S.assistant.guide}</a>
        <button className="cron-act" onClick={onConfigure}>{S.assistant.configure}</button>
      </div>
    </div>
  );
}

function HowToDialog({
  ch, guideUrl, onClose,
}: {
  ch: ChannelRow; guideUrl: string; onClose: () => void;
}) {
  const S = useStrings();
  const copy = channelCopy(S, ch.id);
  const envBlock = ch.env.map((e) => `${e.name}=${e.example}`).join("\n");
  return (
    <Dialog
      title={copy.title}
      onClose={onClose}
      footer={
        <>
          <a className="asst-guide" href={guideUrl} target="_blank" rel="noreferrer">{S.assistant.guide}</a>
          <button className="dp-btn" onClick={onClose}>{S.common.close}</button>
        </>
      }
    >
      <p className="asst-desc">{copy.desc}</p>
      {ch.extra && <p className="asst-meta">{S.assistant.extra(ch.extra)}</p>}
      {ch.env.length > 0 && (
        <>
          <div className="asst-env-head">
            <span>{S.assistant.envTitle}</span>
            <button className="cron-act" onClick={() => void copyText(envBlock)}>{S.assistant.copyEnv}</button>
          </div>
          <pre className="asst-env">{envBlock}</pre>
        </>
      )}
      <p className="asst-meta">{S.assistant.restart}</p>
    </Dialog>
  );
}

function channelCopy(S: Strings, id: string) {
  return S.assistant.channels[id as keyof Strings["assistant"]["channels"]]
    || { title: id, desc: "" };
}

function statusBadge(S: Strings, ch: ChannelRow): { kind: string; label: string } | null {
  if (ch.connected) return { kind: "on", label: S.assistant.connected };
  if (ch.configured) return { kind: "fail", label: S.assistant.failed };
  return { kind: "off", label: S.assistant.off };
}

async function copyText(text: string) {
  await navigator.clipboard.writeText(text);
  showToast(getStrings().common.copied);
}
