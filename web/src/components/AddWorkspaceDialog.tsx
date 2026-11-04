import { useState } from "react";
import { useNavigate } from "react-router";
import { useStrings } from "../i18n";
import { getState, setState, useAppState } from "../store";
import { addWorkspace, closeAddWorkspace } from "../workspaces";
import { Dialog } from "./Dialog";
import { DirPicker } from "./DirPicker";

/** Pick a folder on the gateway's machine and make it a workspace. The folder
 *  must already exist; the server checks that and the error is shown as a toast. */
export function AddWorkspaceDialog() {
  const s = useAppState();
  const S = useStrings();
  const nav = useNavigate();
  const [busy, setBusy] = useState(false);
  const value = s.workspaceAdd.value;
  const submit = async () => {
    if (busy) return;
    setBusy(true);
    const ok = await addWorkspace(value);
    setBusy(false);
    if (ok) nav("/chat");
  };
  return (
    <Dialog
      title={S.workspace.addTitle}
      onClose={closeAddWorkspace}
      wide
      footer={
        <>
          <button className="dp-btn" onClick={closeAddWorkspace}>{S.common.cancel}</button>
          <button className="dp-btn primary" disabled={busy || !value.trim()} onClick={() => void submit()}>
            {S.workspace.addOk}
          </button>
        </>
      }
    >
      <p className="dlg-note">{S.workspace.addDesc}</p>
      <DirPicker
        value={value}
        onChange={(dir) => setState({ workspaceAdd: { ...getState().workspaceAdd, value: dir } })}
        onCommit={() => void submit()}
      />
    </Dialog>
  );
}
