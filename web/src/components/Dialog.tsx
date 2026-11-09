import { useEffect, useRef, type ReactNode } from "react";
import { IconClose } from "../icons";

/** A labelled overlay. ConfirmDialog stays for destructive yes/no; this one
 *  is for forms (add user, change password) that need more than a sentence. */
export function Dialog({
  title, onClose, footer, children, wide,
}: {
  title: string;
  onClose: () => void;
  footer?: ReactNode;
  children: ReactNode;
  wide?: boolean;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopImmediatePropagation();
        onClose();
      }
    };
    document.addEventListener("keydown", onKey, true);
    return () => document.removeEventListener("keydown", onKey, true);
  }, [onClose]);

  const downOnBackdrop = useRef(false);
  return (
    <div className="dlg-overlay open"
         onMouseDown={(e) => { downOnBackdrop.current = e.target === e.currentTarget; }}
         onClick={(e) => { if (downOnBackdrop.current && e.target === e.currentTarget) onClose(); }}>
      <div className={"dlg" + (wide ? " wide" : "")} onClick={(e) => e.stopPropagation()}>
        <div className="dlg-head">
          <h3>{title}</h3>
          <button className="ib" onClick={onClose}><IconClose /></button>
        </div>
        <div className="dlg-body">{children}</div>
        {footer && <div className="dlg-foot">{footer}</div>}
      </div>
    </div>
  );
}

export function Field({
  label, required, hint, error, children,
}: {
  label: string;
  required?: boolean;
  hint?: string;
  error?: string;
  children: ReactNode;
}) {
  return (
    <div className={"fld" + (error ? " has-error" : "")}>
      <span className="fld-label">
        {label}{required && <span className="fld-req">*</span>}
      </span>
      {children}
      {error ? <span className="fld-err">{error}</span>
        : hint ? <span className="fld-hint">{hint}</span> : null}
    </div>
  );
}
