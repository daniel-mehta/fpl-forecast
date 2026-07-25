import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";

const OPEN_TOOLTIP_EVENT = "fpl-dashboard-tooltip-open";

interface InfoTooltipProps {
  label: string;
  children: string;
}

export function InfoTooltip({ label, children }: InfoTooltipProps) {
  const id = useId();
  const buttonRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLSpanElement>(null);
  const wrapperRef = useRef<HTMLSpanElement>(null);
  const [open, setOpen] = useState(false);
  const [horizontalOffset, setHorizontalOffset] = useState(0);

  function show() {
    window.dispatchEvent(new CustomEvent(OPEN_TOOLTIP_EVENT, { detail: id }));
    setOpen(true);
  }

  useEffect(() => {
    function handleOtherTooltip(event: Event) {
      if ((event as CustomEvent<string>).detail !== id) {
        setOpen(false);
      }
    }
    function handlePointerDown(event: PointerEvent) {
      if (!wrapperRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    window.addEventListener(OPEN_TOOLTIP_EVENT, handleOtherTooltip);
    document.addEventListener("pointerdown", handlePointerDown);
    return () => {
      window.removeEventListener(OPEN_TOOLTIP_EVENT, handleOtherTooltip);
      document.removeEventListener("pointerdown", handlePointerDown);
    };
  }, [id]);

  useLayoutEffect(() => {
    if (!open || !panelRef.current) {
      return;
    }
    setHorizontalOffset(0);
    const frame = requestAnimationFrame(() => {
      const rect = panelRef.current?.getBoundingClientRect();
      if (!rect) {
        return;
      }
      const gutter = 8;
      const correction =
        rect.left < gutter
          ? gutter - rect.left
          : rect.right > window.innerWidth - gutter
            ? window.innerWidth - gutter - rect.right
            : 0;
      setHorizontalOffset(correction);
    });
    return () => cancelAnimationFrame(frame);
  }, [open]);

  return (
    <span
      className="info-tooltip"
      ref={wrapperRef}
      onMouseEnter={show}
      onMouseLeave={() => setOpen(false)}
    >
      <button
        ref={buttonRef}
        type="button"
        className="info-tooltip-button"
        aria-label={`More information about ${label}`}
        aria-expanded={open}
        aria-controls={id}
        aria-describedby={open ? id : undefined}
        onFocus={show}
        onClick={show}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            event.preventDefault();
            setOpen(false);
            buttonRef.current?.focus();
          }
        }}
      >
        i
      </button>
      {open && (
        <span
          className="info-tooltip-panel"
          id={id}
          ref={panelRef}
          role="tooltip"
          style={{ transform: `translateX(calc(-50% + ${horizontalOffset}px))` }}
        >
          {children}
        </span>
      )}
    </span>
  );
}

interface InformationLabelProps extends InfoTooltipProps {
  className?: string;
}

export function InformationLabel({
  label,
  children,
  className,
}: InformationLabelProps) {
  return (
    <span className={className}>
      {label} <InfoTooltip label={label}>{children}</InfoTooltip>
    </span>
  );
}
