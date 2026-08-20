import { closeConfirm, useAppState } from "../store";

export function Toast() {
  const s = useAppState();
  if (!s.toast.show) return null;
  return <div className="toast show">{s.toast.msg}</div>;
}

/** The single confirmation dialog. Everything destructive routes through
 *  `askConfirm` so a delete never depends on the browser's native prompt. */
export function ConfirmDialog() {
  const s = useAppState();
  if (!s.confirm.open) return null;
  const run = () => {
    const fn = s.confirm.onOk;
    closeConfirm();
    fn?.();
  };
  return (
    <div className="confirm-overlay open" onClick={closeConfirm}>
      <div className="confirm-modal" onClick={(e) => e.stopPropagation()}>
        <h3>{s.confirm.title}</h3>
        <p>{s.confirm.msg}</p>
        <div className="confirm-actions">
          <button className="dp-btn" onClick={closeConfirm}>取消</button>
          <button className="dp-btn danger" onClick={run}>{s.confirm.okLabel}</button>
        </div>
      </div>
    </div>
  );
}
