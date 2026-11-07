import type { ReactNode, SVGProps } from "react";
import appIcon from "./assets/app-icon.png";

export const LOGO_SRC = appIcon;

export function Logo({ className = "brand-logo" }: { className?: string }) {
  return <img className={className} src={LOGO_SRC} alt="Agentica" />;
}

function S({ children, ...rest }: SVGProps<SVGSVGElement> & { children: ReactNode }) {
  return (
    <svg viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" {...rest}>
      {children}
    </svg>
  );
}

export const IconPlus = () => (
  <S strokeWidth="1.6" strokeLinecap="round"><path d="M8 2.75v10.5M2.75 8h10.5" /></S>
);
export const IconRefresh = () => (
  <S strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
    <path d="M13.25 8a5.25 5.25 0 11-1.54-3.71" /><path d="M13.25 2.75v2.5h-2.5" />
  </S>
);
export const IconDownload = () => (
  <S strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
    <path d="M8 2.5v7.5M4.75 7l3.25 3.25L11.25 7M3 13.25h10" />
  </S>
);
export const IconTick = () => (
  <S strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="M3.5 8.5l3 3 6-6.5" /></S>
);
/** Text editor app: a page with a pen across it. */
export const IconEditorApp = () => (
  <svg viewBox="0 0 16 16" width="16" height="16" fill="none">
    <rect x="1.75" y="1.75" width="12.5" height="12.5" rx="3" fill="var(--editor-app-bg,#4f6bed)" />
    <path d="M4.75 5.5h4M4.75 8h2.5" stroke="#fff" strokeWidth="1.3" strokeLinecap="round" />
    <path d="M7.6 11.6l.35-1.7 3.6-3.6a.9.9 0 011.3 1.3l-3.6 3.6z" fill="#fff" />
  </svg>
);
export const IconSearch = () => (
  <S strokeWidth="1.4"><circle cx="7" cy="7" r="4.25" /><path d="M10.2 10.2l3.05 3.05" strokeLinecap="round" /></S>
);
export const IconClose = () => (
  <S strokeWidth="1.5" strokeLinecap="round"><path d="M4 4l8 8M12 4l-8 8" /></S>
);
export const IconExpand = () => (
  <S strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3.5" y="3.5" width="9" height="9" rx="1.5" />
    <path d="M9.5 3.5L12.5 6.5M12.5 3.5v3h-3" />
  </S>
);
export const IconClock = () => (
  <S strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="8" cy="9" r="5.5" />
    <path d="M8 6.2V9l2 1.2" />
    <path d="M2.7 3.2l1.9 1.7M13.3 3.2l-1.9 1.7M6.2 1.5h3.6" />
  </S>
);
export const IconPlug = () => (
  <S strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
    <path d="M8 14.7v-3.3" />
    <path d="M6 5.3V1.3M10 5.3V1.3" />
    <path d="M12 5.3v3.3a2.7 2.7 0 01-2.7 2.7h-2.6a2.7 2.7 0 01-2.7-2.7V5.3z" />
  </S>
);
export const IconSidebar = () => (
  <S strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
    <rect x="1.5" y="2.5" width="13" height="11" rx="2" />
    <line x1="5.5" y1="2.5" x2="5.5" y2="13.5" />
  </S>
);
/** The left sidebar glyph mirrored: the same affordance on the other edge. */
export const IconSidebarRight = () => (
  <S strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
    <rect x="1.5" y="2.5" width="13" height="11" rx="2" />
    <line x1="10.5" y1="2.5" x2="10.5" y2="13.5" />
  </S>
);
export const IconFullscreen = () => (
  <S strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M1.8 5.4V2.8c0-.55.45-1 1-1h2.7M10.5 1.8h2.7c.55 0 1 .45 1 1v2.6M14.2 10.6v2.6c0 .55-.45 1-1 1h-2.7M5.5 14.2H2.8c-.55 0-1-.45-1-1v-2.6" />
  </S>
);
export const IconExitFullscreen = () => (
  <S strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9 2.5V6c0 .55.45 1 1 1h3.5M7 13.5V10c0-.55-.45-1-1-1H2.5" />
  </S>
);
export const IconDatabase = () => (
  <S strokeWidth="1.3" strokeLinejoin="round">
    <ellipse cx="8" cy="4.2" rx="5.2" ry="2.1" />
    <path d="M2.8 4.2v7.6c0 1.16 2.33 2.1 5.2 2.1s5.2-.94 5.2-2.1V4.2" />
    <path d="M2.8 8c0 1.16 2.33 2.1 5.2 2.1s5.2-.94 5.2-2.1" />
  </S>
);
export const IconFolder = () => (
  <S strokeWidth="1.3" strokeLinejoin="round">
    <path d="M1.75 4.25c0-.55.45-1 1-1h3.1l1.1 1.3h6.05c.55 0 1 .45 1 1v6.2c0 .55-.45 1-1 1h-10.25c-.55 0-1-.45-1-1z" />
  </S>
);
export const IconPencil = () => (
  <S strokeWidth="1.3">
    <path d="M11.55 1.45a1.55 1.55 0 012.2 2.2l-8.2 8.2-3.3.9.9-3.3z" strokeLinejoin="round" />
    <path d="M10.5 2.5l3 3" />
  </S>
);
export const IconArchive = () => (
  <S strokeWidth="1.2">
    <rect x="1.5" y="2.5" width="13" height="3" rx="0.8" />
    <path d="M2.5 5.7V13a1 1 0 001 1h9a1 1 0 001-1V5.7" strokeLinecap="round" />
    <line x1="6.5" y1="8.3" x2="9.5" y2="8.3" strokeLinecap="round" />
  </S>
);
export const IconGear = () => (
  <S strokeWidth="1.3" strokeLinejoin="round" strokeLinecap="round">
    <path d="M6.6 1.75h2.8l.45 1.85c.5.16.96.4 1.38.68l1.8-.62 1.4 2.42-1.4 1.28c.05.26.07.53.07.8s-.02.54-.07.8l1.4 1.28-1.4 2.42-1.8-.62c-.42.28-.88.52-1.38.68l-.45 1.85H6.6l-.45-1.85a5.6 5.6 0 01-1.38-.68l-1.8.62-1.4-2.42 1.4-1.28A5.3 5.3 0 012.9 8c0-.27.02-.54.07-.8L1.57 5.92l1.4-2.42 1.8.62c.42-.28.88-.52 1.38-.68z" />
    <circle cx="8" cy="8" r="2.15" />
  </S>
);
export const IconProfiles = () => (
  <S strokeWidth="1.3">
    <rect x="2" y="2.5" width="12" height="11" rx="1.5" />
    <path d="M5 6h6M5 8.5h6M5 11h3.5" strokeLinecap="round" />
  </S>
);
export const IconCheck = () => (
  <S strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="8" cy="8" r="5.4" />
    <path d="M5.3 8.15l1.85 1.85 3.6-3.7" />
  </S>
);
export const IconSpinner = () => (
  <S className="icon-spin" strokeWidth="1.6" strokeLinecap="round">
    <circle cx="8" cy="8" r="5.4" opacity="0.25" />
    <path d="M8 2.6a5.4 5.4 0 014.4 2.3" />
  </S>
);
export const IconUser = () => (
  <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor">
    <circle cx="8" cy="5.2" r="2.9" />
    <path d="M2.4 14c.5-3.2 3-5 5.6-5s5.1 1.8 5.6 5z" />
  </svg>
);
export const IconLogout = () => (
  <S strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
    <path d="M6.2 2.5H3.4A1.15 1.15 0 002.25 3.65v8.7c0 .64.51 1.15 1.15 1.15H6.2" />
    <path d="M6.5 8h7.25M11.2 5.25L13.75 8 11.2 10.75" />
  </S>
);
export const IconChat = () => (
  <S strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
    <path d="M2 3.5A1.5 1.5 0 013.5 2h9A1.5 1.5 0 0114 3.5v6A1.5 1.5 0 0112.5 11H6l-3 3v-3H3.5A1.5 1.5 0 012 9.5z" />
  </S>
);
export const IconCopy = () => (
  <svg viewBox="0 0 16 16" width="13" height="13" fill="currentColor">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25v-7.5z" />
    <path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25v-7.5zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-7.5z" />
  </svg>
);
export const IconFinder = () => (
  <svg viewBox="0 0 16 16" width="13" height="13" fill="currentColor">
    <path d="M1.75 1A1.75 1.75 0 000 2.75v10.5C0 14.216.784 15 1.75 15h12.5A1.75 1.75 0 0016 13.25v-8.5A1.75 1.75 0 0014.25 3H7.5a.25.25 0 01-.2-.1l-.9-1.2C6.07 1.26 5.55 1 5 1H1.75z" />
  </svg>
);
/** dsh's compass: a ring with the needle pointing north-east. */
export const IconCompass = () => (
  <svg viewBox="0 0 16 16" width="13" height="13" fill="none" aria-hidden="true">
    <path d="M8 14A6 6 0 1 0 8 2a6 6 0 0 0 0 12Z" stroke="currentColor" />
    <path d="M10.61 5.39 8.99 8.99l-3.6 1.62 1.62-3.6 3.6-1.62Z" fill="currentColor" />
  </svg>
);
export const IconTerminal = () => (
  <svg viewBox="0 0 16 16" width="13" height="13" fill="currentColor">
    <path d="M0 2.75C0 1.784.784 1 1.75 1h12.5c.966 0 1.75.784 1.75 1.75v10.5A1.75 1.75 0 0114.25 15H1.75A1.75 1.75 0 010 13.25V2.75zm7 7.5a.75.75 0 000 1.5h4.25a.75.75 0 000-1.5H7zm-3.22-4.97a.75.75 0 00-1.06 1.06L4.44 8 2.72 9.72a.75.75 0 101.06 1.06l2-2a.75.75 0 000-1.06l-2-2z" />
  </svg>
);
export const IconArrowDown = () => (
  <S strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
    <path d="M8 3v10M4.3 9.3L8 13l3.7-3.7" />
  </S>
);
export const IconChevronDown = () => (
  <S strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3.5 6.5L8 11l4.5-4.5" />
  </S>
);
export const IconBook = () => (
  <S strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
    <path d="M8 4.2C6.6 3 4.4 2.8 2.6 3.6v8.6c1.8-.8 4-.6 5.4.6 1.4-1.2 3.6-1.4 5.4-.6V3.6C11.6 2.8 9.4 3 8 4.2z" />
    <path d="M8 4.2v8.6" />
  </S>
);
export const IconAsk = () => (
  <S strokeWidth="1.3" strokeLinecap="round">
    <ellipse cx="8" cy="8" rx="5.4" ry="3.2" />
    <circle cx="8" cy="8" r="1.35" />
  </S>
);
export const IconAuto = () => (
  <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor" stroke="none">
    <path d="M8.7 1.4L3.2 8.9h3.5L6.4 14.6 12.8 6.6H9.2z" />
  </svg>
);
export const IconAllowAll = () => (
  <S strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
    <path d="M8 2.4L14.2 13.2H1.8z" />
    <path d="M8 6.4v3.4" />
    <circle cx="8" cy="11.35" r="0.45" fill="currentColor" stroke="none" />
  </S>
);
export const IconSend = () => (
  <svg viewBox="0 0 16 16" width="16" height="16">
    <path d="M8 13V4M8 4L4.3 7.7M8 4l3.7 3.7" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);
export const IconStop = () => (
  <svg viewBox="0 0 16 16" width="16" height="16">
    <rect x="3" y="3" width="10" height="10" rx="3" fill="currentColor" />
  </svg>
);
export const IconMore = () => (
  <S strokeWidth="1.4" strokeLinecap="round">
    <circle cx="3.5" cy="8" r=".6" fill="currentColor" /><circle cx="8" cy="8" r=".6" fill="currentColor" /><circle cx="12.5" cy="8" r=".6" fill="currentColor" />
  </S>
);
export const IconTrash = () => (
  <S strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
    <path d="M2.5 4.25h11M6.2 4.25V2.9a.9.9 0 01.9-.9h1.8a.9.9 0 01.9.9v1.35M4 4.25l.55 8.4a1 1 0 001 .93h4.9a1 1 0 001-.93l.55-8.4M6.6 7v4M9.4 7v4" />
  </S>
);
export const IconChevronRight = () => (
  <S strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M6 3.5L10.5 8 6 12.5" /></S>
);
export const IconFolderPlus = () => (
  <S strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 4.5h3l1.2 1.5H13a1 1 0 011 1v4.5a1.5 1.5 0 01-1.5 1.5h-9A1.5 1.5 0 012 11.5v-6A1 1 0 013 4.5z" />
  </S>
);
export const IconThumbUp = () => (
  <S strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
    <path d="M5 7l2.4-4.6a1.3 1.3 0 012.4.8L9.4 6.3h3.3a1.3 1.3 0 011.3 1.5l-.8 4.6a1.3 1.3 0 01-1.3 1.1H5zM5 7H2.6v6.5H5" />
  </S>
);
export const IconThumbDown = () => (
  <S strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
    <path d="M11 9l-2.4 4.6a1.3 1.3 0 01-2.4-.8l.4-3.1H3.3a1.3 1.3 0 01-1.3-1.5l.8-4.6A1.3 1.3 0 014.1 2.5H11zM11 9h2.4V2.5H11" />
  </S>
);
export const IconFork = () => (
  <S strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="4.5" cy="3.5" r="1.5" /><circle cx="11.5" cy="3.5" r="1.5" /><circle cx="8" cy="12.5" r="1.5" />
    <path d="M4.5 5v1.2c0 1 .8 1.8 1.8 1.8h3.4c1 0 1.8-.8 1.8-1.8V5M8 8v3" />
  </S>
);
export const IconExternal = () => (
  <S strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9.5 2.5h4v4M13.5 2.5L7.5 8.5M12 9.5v3a1 1 0 01-1 1H3.5a1 1 0 01-1-1V5a1 1 0 011-1h3" />
  </S>
);
export const IconFileCode = () => (
  <S strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9.5 1.75H4.25a1 1 0 00-1 1v10.5a1 1 0 001 1h7.5a1 1 0 001-1V4.75z" /><path d="M9.5 1.75v3h3" />
    <path d="M6.6 8.2L5.3 9.6l1.3 1.4M9.4 8.2l1.3 1.4-1.3 1.4" />
  </S>
);
export const IconSplit = () => (
  <S strokeWidth="1.3" strokeLinejoin="round">
    <rect x="2" y="2.75" width="12" height="10.5" rx="1.5" /><path d="M8 2.75v10.5" />
  </S>
);
export const IconDiff = () => (
  <S strokeWidth="1.3" strokeLinecap="round">
    <path d="M5 2.5v5M2.5 5h5M2.5 12h5M10.5 2.5v11M13.5 10.5l-3 3-3-3" />
  </S>
);
export const IconWrap = () => (
  <S strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
    <path d="M2.5 4h11M2.5 8h9a2 2 0 010 4H8.5M10 10.5L8.5 12l1.5 1.5M2.5 12h3" />
  </S>
);
export const IconPaperclip = () => (
  <S strokeWidth="1.35" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10.75 4.75 5.6 9.9a1.4 1.4 0 002 2l5.4-5.4a2.75 2.75 0 00-3.9-3.9L3.7 8a4.1 4.1 0 005.8 5.8l4.25-4.25" />
  </S>
);
export const IconTarget = () => (
  <S strokeWidth="1.35" strokeLinecap="round">
    <circle cx="8" cy="8" r="5.75" /><circle cx="8" cy="8" r="3.1" /><circle cx="8" cy="8" r=".7" fill="currentColor" />
  </S>
);
export const IconQueue = () => (
  <S strokeWidth="1.35" strokeLinecap="round" strokeLinejoin="round">
    <path d="M2.75 4h7.5M2.75 8h7.5M2.75 12h4.5M11.5 10.25 13.25 12 11.5 13.75" />
  </S>
);
export const IconCompact = () => (
  <S strokeWidth="1.35" strokeLinecap="round">
    <path d="M8 2.25a5.75 5.75 0 105.75 5.75" /><path d="M10.6 2.85a5.8 5.8 0 012.55 2.55" strokeDasharray="1.2 1.6" />
  </S>
);
export const IconShield = () => (
  <S strokeWidth="1.35" strokeLinecap="round" strokeLinejoin="round">
    <path d="M8 1.9 3 3.75v3.9c0 3.05 2.1 5.4 5 6.45 2.9-1.05 5-3.4 5-6.45v-3.9L8 1.9z" /><path d="M8 5.25v3.2" /><path d="M8 10.6h.01" />
  </S>
);
export const IconModel = () => (
  <S strokeWidth="1.35" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3.75" y="3.75" width="8.5" height="8.5" rx="1.6" /><path d="M6.25 1.75v2M9.75 1.75v2M6.25 12.25v2M9.75 12.25v2M1.75 6.25h2M1.75 9.75h2M12.25 6.25h2M12.25 9.75h2" /><path d="M6.4 6.4h3.2v3.2H6.4z" />
  </S>
);

/* Work-row category icons (thinking, read, edit, web, subagents, plan, error). */
export const IconBrain = () => (
  <S strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
    <path d="M6 3.2a2 2 0 00-3 1.7 2 2 0 00-.6 3.4A2 2 0 004 11.5a2 2 0 003.4 1.1V3.6A1.9 1.9 0 006 3.2z" />
    <path d="M10 3.2a2 2 0 013 1.7 2 2 0 01.6 3.4A2 2 0 0112 11.5a2 2 0 01-3.4 1.1V3.6A1.9 1.9 0 0110 3.2z" />
  </S>
);
export const IconFile = () => (
  <S strokeWidth="1.3" strokeLinejoin="round"><path d="M4 2.5h5l3 3v8H4z" /><path d="M9 2.5v3h3" /></S>
);
export const IconGlobe = () => (
  <S strokeWidth="1.3"><circle cx="8" cy="8" r="5.5" /><path d="M2.5 8h11M8 2.5c1.7 1.6 2.5 3.4 2.5 5.5S9.7 11.9 8 13.5C6.3 11.9 5.5 10.1 5.5 8S6.3 4.1 8 2.5z" /></S>
);
export const IconUsers = () => (
  <S strokeWidth="1.3" strokeLinecap="round"><circle cx="6" cy="5.5" r="2.2" /><path d="M2.2 13c.4-2.2 1.9-3.4 3.8-3.4s3.4 1.2 3.8 3.4" /><path d="M10.4 3.6a2 2 0 010 3.8M11.6 9.8c1.2.4 2 1.4 2.2 3.2" /></S>
);
export const IconListCheck = () => (
  <S strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"><path d="M2.5 4.5l1.2 1.2L6 3.4M2.5 10l1.2 1.2L6 8.9M8 4.5h5.5M8 10h5.5" /></S>
);
export const IconAlert = () => (
  <S strokeWidth="1.4" strokeLinecap="round"><circle cx="8" cy="8" r="5.6" /><path d="M8 5v3.4M8 10.8v.1" /></S>
);

export const IconPin = ({ filled = false }: { filled?: boolean }) => (
  <S strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9.6 2.2l4.2 4.2-1.6.5-2.4 2.4.3 3-1.3 1.3-2.4-2.4-3.3 3.3M5.3 8.5L2.9 6.1l1.3-1.3 3 .3 2.4-2.4z"
          fill={filled ? "currentColor" : "none"} />
  </S>
);
export const IconCopyOutline = () => (
  <S strokeWidth="1.3" strokeLinejoin="round"><rect x="5.5" y="5.5" width="8" height="8" rx="1.5" /><path d="M10.5 5.5V3.5a1 1 0 00-1-1h-6a1 1 0 00-1 1v6a1 1 0 001 1h2" /></S>
);
