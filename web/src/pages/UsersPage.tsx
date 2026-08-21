import { useEffect, useState } from "react";
import { Navigate } from "react-router";
import * as api from "../api";
import { Dialog, Field } from "../components/Dialog";
import { openPasswordDialog } from "../components/ChangePasswordDialog";
import { loadAuthStatus, loadUsers } from "../data";
import { getStrings, useStrings } from "../i18n";
import { USERNAME_PATTERN, normalizeUsername } from "../lib/username";
import { askConfirm, showToast, useAppState } from "../store";

/** Account table on the right of the same shell as chat.

 *  It used to live as a tab inside the settings modal, which made a
 *  machine-wide dialog own a per-account table, and packed create / reset /
 *  delete into one cramped block. The left sidebar does not change: leaving
 *  this page for a conversation is the usual next move, so the tree stays. */
export function UsersPage() {
  const s = useAppState();
  const S = useStrings();
  const [createOpen, setCreateOpen] = useState(false);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    void loadAuthStatus().then(() => setReady(true));
  }, []);
  useEffect(() => { if (ready && s.accountRole === "admin") void loadUsers(); }, [ready, s.accountRole]);

  if (!ready) {
    return (
      <div className="main"><div className="page-body"><p className="muted">{S.common.loading}</p></div></div>
    );
  }
  if (s.accountRole !== "admin") return <Navigate to="/chat" replace />;

  return (
    <>
      <div className="main">
        <div className="topbar">
          <h3 className="page-title">{S.users.title}</h3>
          <button className="settings-new-btn" onClick={() => setCreateOpen(true)}>{S.users.add}</button>
        </div>
        <div className="page-body">
          <div className="users-table-wrap">
            <table className="users-table">
              <thead>
                <tr>
                  <th>{S.users.username}</th>
                  <th>{S.users.role}</th>
                  <th>{S.users.created}</th>
                  <th className="num">{S.users.actions}</th>
                </tr>
              </thead>
              <tbody>
                {!s.users.length && (
                  <tr><td colSpan={4} className="users-empty">{S.common.loading}</td></tr>
                )}
                {s.users.map((u) => (
                  <tr key={u.user_id} className={u.user_id === s.accountId ? "is-you" : ""}>
                    <td>
                      <span className="users-id">{u.user_id}</span>
                      {u.password_is_initial && <span className="users-badge">{S.users.initialFlag}</span>}
                      {u.user_id === s.accountId && <span className="users-badge you">{S.users.you}</span>}
                    </td>
                    <td><span className="users-badge">{u.is_admin ? S.users.roleAdmin : S.users.roleUser}</span></td>
                    <td className="muted">{formatWhen(u.created_at)}</td>
                    <td className="num">
                      <button className="cron-act" onClick={() => openPasswordDialog(u.user_id)}>
                        {S.users.changePassword}
                      </button>
                      {u.user_id !== s.accountId && (
                        <button className="cron-act danger" onClick={() => removeUser(u.user_id)}>
                          {S.common.delete}
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      {createOpen && <CreateUserDialog onClose={() => setCreateOpen(false)} />}
    </>
  );
}

function CreateUserDialog({ onClose }: { onClose: () => void }) {
  const S = useStrings();
  const min = useAppState().minPasswordLength;
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [errors, setErrors] = useState<{ username?: string; password?: string }>({});
  const [busy, setBusy] = useState(false);
  const id = normalizeUsername(username);
  const idOk = USERNAME_PATTERN.test(id);

  async function submit() {
    const next: typeof errors = {};
    if (!username.trim()) next.username = S.common.requiredField;
    else if (!idOk) next.username = S.users.usernameHint;
    if (!password) next.password = S.common.requiredField;
    else if (password.length < min) next.password = S.users.passwordHint(min);
    if (next.username || next.password) { setErrors(next); return; }
    setBusy(true);
    setErrors({});
    const { ok, data } = await api.createUserApi(id, password);
    setBusy(false);
    if (!ok) {
      const detail = String((data as any)?.detail || S.common.saveFailed);
      const onPass = /password|character/i.test(detail);
      setErrors(onPass ? { password: detail } : { username: detail });
      return;
    }
    onClose();
    await loadUsers();
    showToast(getStrings().users.createdOk(id));
  }

  return (
    <Dialog
      title={S.users.add}
      onClose={onClose}
      footer={
        <>
          <button className="dp-btn" disabled={busy} onClick={onClose}>{S.common.cancel}</button>
          <button className="dp-btn primary" disabled={busy} onClick={() => void submit()}>{S.common.create}</button>
        </>
      }
    >
      <Field label={S.users.username} required hint={S.users.usernameHint} error={errors.username}>
        <input className="pf-input" autoFocus autoComplete="off" spellCheck={false}
               value={username}
               onChange={(e) => { setUsername(e.target.value); setErrors({}); }} />
      </Field>
      <Field label={S.users.initialPassword} required
             hint={S.users.passwordHint(min)} error={errors.password}>
        <input className="pf-input" type="password" autoComplete="new-password"
               value={password}
               onChange={(e) => { setPassword(e.target.value); setErrors({}); }} />
      </Field>
      {idOk && <p className="dlg-note">{S.users.defaultProject(id)}</p>}
    </Dialog>
  );
}

function removeUser(userId: string) {
  const S = getStrings();
  askConfirm({
    title: S.users.removeTitle(userId),
    msg: S.users.removeMsg(userId),
    onOk: async () => {
      const { ok, data } = await api.deleteUserApi(userId);
      if (!ok) { showToast((data as any)?.detail || S.common.deleteFailed, 3500); return; }
      await loadUsers();
      showToast(S.users.removed);
    },
  });
}

function formatWhen(iso: string) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso.replace("T", " ").slice(0, 19);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}
