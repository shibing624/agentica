import { useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router";
import * as api from "../api";
import { useStrings } from "../i18n";

/** Where to go after signing in. Only a local path is accepted: `next` comes
 *  off the query string, so an absolute URL here would be an open redirect. */
function safeNext(raw: string | null): string {
  if (!raw || !raw.startsWith("/") || raw.startsWith("//")) return "/chat";
  return raw;
}

type Status = {
  auth_enabled: boolean;
  password_set: boolean;
  authenticated: boolean;
  default_account_id: string;
  password_is_initial: boolean;
  min_password_length: number;
};

export function LoginPage() {
  const S = useStrings();
  const nav = useNavigate();
  const [params] = useSearchParams();
  const next = safeNext(params.get("next"));
  const [status, setStatus] = useState<Status | null>(null);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  // Kept as the status code plus whatever the server said, not as a finished
  // sentence: a rendered message would not follow a language change.
  const [err, setErr] = useState<{ code: number; detail: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    void (async () => {
      const { data } = await api.fetchAuthStatus();
      const st = data as Status | null;
      setStatus(st);
      // Already in (or the gate is off): nothing to ask. Reaching this page
      // with a live session means a stale link, not a sign-out.
      if (st && (st.authenticated || !st.auth_enabled)) nav(next, { replace: true });
      else {
        // Prefilled with the account every machine has, so the single-account
        // case is still "type the password and press Enter".
        setUsername(st?.default_account_id || "default");
        input.current?.focus();
      }
    })();
  }, []);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!password || busy) return;
    setBusy(true);
    setErr(null);
    const { ok, status: code, data } = await api.loginApi(username, password);
    setBusy(false);
    if (ok) { nav(next, { replace: true }); return; }
    setErr({ code, detail: String((data as any)?.detail || "") });
    setPassword("");
    input.current?.focus();
  }

  // 429 carries how long to wait, so the server's own wording wins there — a
  // generic "wrong password" after a few typos just looks broken.
  const errorText = !err ? ""
    : err.code === 429 ? (err.detail || S.login.retryLater)
    : err.code === 401 ? S.login.wrongCredentials
    : (err.detail || S.login.failed);

  if (status === null) return <div className="login-wrap" />;

  // Nobody can sign in: the account was created with a password on first start,
  // so getting here means it was explicitly cleared. Say what fixes it instead
  // of showing a form nothing can satisfy.
  if (!status.password_set) {
    return (
      <div className="login-wrap">
        <div className="login-card">
          <h1>Agentica</h1>
          <p className="login-hint">
            {S.login.noPassword} <code>agentica-gateway --set-password</code> {S.login.noPasswordTail}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="login-wrap">
      <form className="login-card" onSubmit={submit}>
        <h1>Agentica</h1>
        <p className="login-hint">
          {S.login.signInAs}<code>{status.default_account_id}</code>
          {status.password_is_initial ? S.login.printedOnStart : S.login.fullStop}
        </p>
        {/* Editable now that an admin can add accounts, and prefilled with the
            seeded one so nothing changes for a machine that has just the one.
            The field would be here regardless: a password manager will not
            offer to save a credential it cannot see a username for. */}
        <input className="pf-input" type="text" autoComplete="username"
               placeholder={S.login.username}
               value={username} onChange={(e) => setUsername(e.target.value)} />
        <input
          ref={input}
          className="pf-input"
          type="password"
          autoComplete="current-password"
          placeholder={S.login.password}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        {!!errorText && <div className="login-error">{errorText}</div>}
        <button className="dp-btn primary login-submit" type="submit" disabled={busy || !password || !username}>
          {busy ? S.login.submitting : S.login.submit}
        </button>
        <p className="login-foot">
          {status.password_is_initial ? S.login.footInitial : S.login.footForgot}
        </p>
      </form>
    </div>
  );
}
