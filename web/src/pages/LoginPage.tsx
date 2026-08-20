import { useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router";
import * as api from "../api";

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
  min_password_length: number;
};

export function LoginPage() {
  const nav = useNavigate();
  const [params] = useSearchParams();
  const next = safeNext(params.get("next"));
  const [status, setStatus] = useState<Status | null>(null);
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
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
      else input.current?.focus();
    })();
  }, []);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!password || busy) return;
    setBusy(true);
    setError("");
    const { ok, status: code, data } = await api.loginApi(password);
    setBusy(false);
    if (ok) { nav(next, { replace: true }); return; }
    // 429 carries how long to wait — worth showing verbatim, or the page just
    // looks broken after a few typos.
    setError(code === 429 ? String((data as any)?.detail || "请稍后再试")
      : code === 401 ? "密码不对"
      : String((data as any)?.detail || "登录失败"));
    setPassword("");
    input.current?.focus();
  }

  if (status === null) return <div className="login-wrap" />;

  // No password on this gateway: the printed token URL is the only way in, so
  // send the user there instead of showing a form nothing can satisfy.
  if (!status.password_set) {
    return (
      <div className="login-wrap">
        <div className="login-card">
          <h1>Agentica</h1>
          <p className="login-hint">
            这台 gateway 还没设密码，入口是启动终端里打印的那条
            <code>/chat?token=…</code> 地址，打开一次浏览器就会记住。
          </p>
          <p className="login-hint">
            想改成密码登录：在启动 gateway 的机器上执行
            <code>agentica-gateway --set-password</code>，或在网页的
            设置 › 常规 › 访问控制 里设一个。
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="login-wrap">
      <form className="login-card" onSubmit={submit}>
        <h1>Agentica</h1>
        <p className="login-hint">这个界面能改配置、读文件、执行命令，所以要先登录。</p>
        <input
          ref={input}
          className="pf-input"
          type="password"
          autoComplete="current-password"
          placeholder="密码"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        {!!error && <div className="login-error">{error}</div>}
        <button className="dp-btn primary login-submit" type="submit" disabled={busy || !password}>
          {busy ? "登录中…" : "登录"}
        </button>
        <p className="login-foot">
          忘了密码？在这台机器上执行 <code>agentica-gateway --set-password</code> 重设。
        </p>
      </form>
    </div>
  );
}
