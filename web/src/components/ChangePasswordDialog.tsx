import { useEffect, useState } from "react";
import * as api from "../api";
import { loadAuthStatus } from "../data";
import { getStrings, useStrings } from "../i18n";
import { getState, setState, showToast, useAppState } from "../store";
import { Dialog, Field } from "./Dialog";

/** Change a password.

 *  Two forms, because they answer different questions:
 *  - your own (the built-in administrator, or a user via the account menu):
 *    current + new + confirm. The initial-password hint is only for the
 *    seeded admin — that is the password printed at first start.
 *  - somebody else (admin changing a user): new + confirm. The admin is
 *    already signed in; asking for a current password is a reset in disguise.
 *  A token/desktop session changing its own may skip current: they already
 *  proved they can read a 0600 file on this machine. */
export function ChangePasswordDialog() {
  const s = useAppState();
  const S = useStrings();
  const dlg = s.passwordDialog;
  const self = dlg.userId === s.accountId;
  const skipOld = s.sessionVia === "token" || s.sessionVia === "desktop";
  const needOld = self && !skipOld;
  const builtinAdmin = dlg.userId === s.defaultAccountId;
  const min = s.minPasswordLength;
  const [old, setOld] = useState("");
  const [next, setNext] = useState("");
  const [repeat, setRepeat] = useState("");
  const [errors, setErrors] = useState<{ old?: string; next?: string; repeat?: string }>({});
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!dlg.open) return;
    setOld(""); setNext(""); setRepeat(""); setErrors({});
  }, [dlg.open, dlg.userId]);

  if (!dlg.open) return null;

  const close = () => setState({ passwordDialog: { open: false, userId: "" } });

  async function save() {
    const nextErr: typeof errors = {};
    if (needOld && !old) nextErr.old = S.common.requiredField;
    if (!next) nextErr.next = S.common.requiredField;
    else if (next.length < min) nextErr.next = S.settings.tooShort(min);
    if (!repeat) nextErr.repeat = S.common.requiredField;
    else if (next !== repeat) nextErr.repeat = S.settings.mismatch;
    if (nextErr.old || nextErr.next || nextErr.repeat) { setErrors(nextErr); return; }
    setBusy(true);
    setErrors({});
    const res = self
      ? await api.setPasswordApi(next, needOld ? old : undefined)
      : await api.changeUserPasswordApi(dlg.userId, next);
    setBusy(false);
    if (!res.ok) {
      const detail = String((res.data as any)?.detail || S.settings.setFailed);
      const onNew = /at least|least \d/i.test(detail);
      setErrors(onNew ? { next: detail } : { old: detail });
      return;
    }
    close();
    await loadAuthStatus();
    showToast(getStrings().settings.passwordChanged);
  }

  return (
    <Dialog
      title={S.users.changePassword}
      onClose={close}
      footer={
        <>
          <button className="dp-btn" disabled={busy} onClick={close}>{S.common.cancel}</button>
          <button className="dp-btn primary" disabled={busy} onClick={() => void save()}>{S.common.save}</button>
        </>
      }
    >
      {!self && (
        <p className="dlg-note">{S.users.changeFor(dlg.userId)}</p>
      )}
      {needOld && (
        <Field label={S.users.currentPassword} required
               hint={builtinAdmin ? S.users.currentAdminPasswordHint : undefined}
               error={errors.old}>
          <input className="pf-input" type="password" autoComplete="current-password"
                 autoFocus value={old}
                 onChange={(e) => { setOld(e.target.value); setErrors({}); }} />
        </Field>
      )}
      <Field label={S.users.newPassword} required
             hint={S.users.passwordHint(min)} error={errors.next}>
        <input className="pf-input" type="password" autoComplete="new-password"
               autoFocus={!needOld} value={next}
               onChange={(e) => { setNext(e.target.value); setErrors({}); }} />
      </Field>
      <Field label={S.users.confirmPassword} required error={errors.repeat}>
        <input className="pf-input" type="password" autoComplete="new-password"
               value={repeat}
               onChange={(e) => { setRepeat(e.target.value); setErrors({}); }} />
      </Field>
    </Dialog>
  );
}

export function openPasswordDialog(userId?: string) {
  setState({ passwordDialog: { open: true, userId: userId || getState().accountId } });
}
