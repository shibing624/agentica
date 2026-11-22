import { useCallback, useEffect, useRef, useState, type ImgHTMLAttributes } from "react";
import { createPortal } from "react-dom";
import { useStrings } from "../i18n";
import { IconClose } from "../icons";

/**
 * Document-level original-image preview opened by clicking a thumbnail (dsh
 * `ImageLightbox`). Closes on Escape, backdrop press, or the close control,
 * and restores focus to the opener. Portaled above the preview, including
 * when the thumbnail lives in a modal dialog's top layer.
 */
export function ImageLightbox({ src, alt, onClose, container }: { src: string; alt: string; onClose: () => void; container: HTMLElement }) {
  const S = useStrings();
  const closeRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    const restore = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    closeRef.current?.focus();
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); onClose(); }
      if (e.key === "Tab") { e.preventDefault(); closeRef.current?.focus(); }
    };
    window.addEventListener("keydown", onKeyDown, true);
    return () => {
      window.removeEventListener("keydown", onKeyDown, true);
      restore?.focus();
    };
  }, [onClose]);

  return createPortal(
    <div className="img-lightbox" role="dialog" aria-modal="true" aria-label={S.common.imagePreview}>
      <div className="img-lightbox-mask" aria-hidden="true" onMouseDown={onClose} />
      <img className="img-lightbox-image" src={src} alt={alt} />
      <button ref={closeRef} type="button" className="img-lightbox-close" aria-label={S.common.close} onClick={onClose}>
        <IconClose />
      </button>
    </div>,
    container,
  );
}

/** An `<img>` thumbnail that opens the full image in the lightbox on click. */
export function PreviewableImage({ src, alt = "", className, ...rest }: ImgHTMLAttributes<HTMLImageElement> & { src?: string }) {
  const S = useStrings();
  const [container, setContainer] = useState<HTMLElement | null>(null);
  const close = useCallback(() => setContainer(null), []);
  if (!src) return null;
  return (
    <>
      <button type="button" className={"img-open" + (className ? " " + className : "")}
              aria-label={alt ? `${S.common.imagePreview}: ${alt}` : S.common.imagePreview}
              onClick={(e) => { e.stopPropagation(); setContainer(e.currentTarget.closest<HTMLDialogElement>("dialog[open]") ?? document.body); }}>
        <img src={src} alt={alt} {...rest} />
      </button>
      {container && <ImageLightbox src={src} alt={alt} onClose={close} container={container} />}
    </>
  );
}
