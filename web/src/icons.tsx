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
export const IconUpload = () => (
  <S strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
    <path d="M8 10.5V3M4.75 6.25L8 3l3.25 3.25M3 13.25h10" />
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
  <svg viewBox="0 0 16 16" width="16" height="16" fill="none" strokeWidth="1.2">
    <path d="M5.08545 8.13775L7.18455 10.2368C7.26636 10.3187 7.4003 10.3142 7.47649 10.2271L11.5148 5.61194" stroke="currentColor" />
    <path d="M6.59624 2.14853C7.50155 1.80917 8.49914 1.80919 9.40444 2.14859L13.9245 3.84317V7.11961C13.9245 11.6089 10.5565 13.5975 8.00035 14.5779C5.44423 13.5975 2.07544 11.6089 2.07544 7.11961V3.84317L6.59624 2.14853Z" stroke="currentColor" strokeLinejoin="round" />
  </svg>
);
export const IconAuto = () => (
  <svg viewBox="0 0 16 16" width="16" height="16" fill="none" strokeWidth="1.2">
    <path d="M6.4209 1.68067C7.43922 1.299 8.56177 1.29898 9.58008 1.68067L14.0996 3.375C14.2946 3.44811 14.4236 3.63455 14.4238 3.84278V6.89063C14.1115 6.71853 13.7761 6.58312 13.4238 6.48926V4.18946L9.22852 2.61621C8.43657 2.31947 7.56341 2.31939 6.77148 2.61621L2.5752 4.18946V7.11914C2.5752 11.1796 5.52369 13.056 8 14.0391C8.27653 13.9293 8.55827 13.8067 8.8418 13.6729C9.07101 13.9468 9.33228 14.1929 9.62012 14.4053C9.12409 14.6579 8.63578 14.8696 8.17871 15.0449C8.0637 15.0889 7.93628 15.0889 7.82129 15.0449C5.22011 14.0472 1.5752 11.9381 1.5752 7.11914V3.84278C1.57541 3.63469 1.70463 3.44821 1.89941 3.375L6.4209 1.68067Z" fill="currentColor" />
    <path d="M5.26392 6.60339H10.7361" stroke="currentColor" />
    <path d="M5.26392 9.86902H8.32833" stroke="currentColor" />
    <path d="M10.0317 13.2229C10.263 13.3929 10.4943 13.563 10.7256 13.733C10.7932 13.6482 10.8608 13.5634 10.9284 13.4786C12.1455 11.9522 13.3626 10.4258 14.5798 8.89935C14.6474 8.81455 14.715 8.72975 14.7826 8.64495C14.4143 8.37419 14.046 8.10344 13.6777 7.83268C13.6169 7.92252 13.5562 8.01236 13.4954 8.10219C12.4016 9.71926 11.3078 11.3363 10.214 12.9534C10.1532 13.0432 10.0924 13.1331 10.0317 13.2229Z" fill="currentColor" />
    <path d="M12.6516 12.6696C12.6516 12.925 12.6516 13.1804 12.6516 13.4359C12.6952 13.4378 12.7387 13.4398 12.7823 13.4417C13.5663 13.4768 14.3504 13.5118 15.1345 13.5469C15.178 13.5488 15.2216 13.5508 15.2651 13.5527C15.2651 13.2194 15.2651 12.8861 15.2651 12.5527C15.2216 12.5547 15.178 12.5566 15.1345 12.5586C14.3504 12.5936 13.5663 12.6287 12.7823 12.6637C12.7387 12.6657 12.6952 12.6676 12.6516 12.6696Z" fill="currentColor" />
  </svg>
);
export const IconAllowAll = () => (
  <svg viewBox="0 0 16 16" width="16" height="16" fill="none" strokeWidth="1.2">
    <path d="M6.59624 2.14853C7.50155 1.80917 8.49914 1.80919 9.40444 2.14859L13.9245 3.84317V7.11961C13.9245 11.6089 10.5565 13.5975 8.00035 14.5779C5.44423 13.5975 2.07544 11.6089 2.07544 7.11961V3.84317L6.59624 2.14853Z" stroke="currentColor" strokeLinejoin="round" />
    <path d="M8 4.39209V9.89209" stroke="currentColor" />
    <path d="M8 10.8081V11.8081" stroke="currentColor" />
  </svg>
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

/** Minimalist dark user avatar (dsh user-outline on dark circle). */
export const DefaultAvatar = () => (
  <svg viewBox="0 0 24 24" width="24" height="24" fill="none" aria-hidden="true">
    <circle cx="12" cy="12" r="12" fill="#18181b" />
    <path d="M12 11.8C13.66 11.8 15 10.46 15 8.8C15 7.14 13.66 5.8 12 5.8C10.34 5.8 9 7.14 9 8.8C9 10.46 10.34 11.8 12 11.8Z" stroke="#ffffff" strokeWidth="1.4" />
    <path d="M6.4 18.2C6.4 15.1 9.2 13.8 12 13.8C14.8 13.8 17.6 15.1 17.6 18.2" stroke="#ffffff" strokeWidth="1.4" strokeLinecap="round" />
  </svg>
);

/**
 * Coloured file/folder sheet glyphs ported from dsh's `FileTypeIcon.tsx`
 * (`@deepseek-ai/dsh-client-ui-primitives`), used on the right panel's
 * `Files` tab chip (`FilesTitle`), opened file tab chips (`TextTitle`),
 * and workspace file tree rows (`FileTree`).
 */
export type FileIconKind =
  | "folder"
  | "code"
  | "markdown"
  | "html"
  | "pdf"
  | "image"
  | "excel"
  | "word"
  | "ppt"
  | "video"
  | "other";

const EXT_ICON_KIND: Record<string, Exclude<FileIconKind, "folder">> = {
  md: "markdown",
  mdx: "markdown",
  markdown: "markdown",
  html: "html",
  htm: "html",
  pdf: "pdf",
  png: "image",
  jpg: "image",
  jpeg: "image",
  gif: "image",
  svg: "image",
  webp: "image",
  avif: "image",
  bmp: "image",
  ico: "image",
  tif: "image",
  tiff: "image",
  heic: "image",
  heif: "image",
  csv: "excel",
  tsv: "excel",
  xls: "excel",
  xlsx: "excel",
  ods: "excel",
  numbers: "excel",
  doc: "word",
  docx: "word",
  rtf: "word",
  odt: "word",
  pages: "word",
  ppt: "ppt",
  pptx: "ppt",
  odp: "ppt",
  mp4: "video",
  mov: "video",
  m4v: "video",
  webm: "video",
  mkv: "video",
  avi: "video",
};

const CODE_ICON_EXTS = new Set([
  "py", "pyi", "pyw", "js", "mjs", "cjs", "jsx", "ts", "tsx", "mts", "cts",
  "json", "jsonc", "jsonl", "ndjson", "ipynb", "yaml", "yml", "toml",
  "sh", "bash", "zsh", "fish", "css", "scss", "sass", "less", "xml", "plist",
  "rs", "go", "java", "kt", "kts", "swift", "c", "h", "cpp", "hpp", "cc", "cxx", "hh", "hxx",
  "cs", "rb", "php", "sql", "lua", "r", "jl", "dart", "scala", "clj", "cljs", "edn",
  "erl", "hrl", "ex", "exs", "hs", "fs", "pl", "pm", "proto", "graphql", "gql",
  "tf", "tfvars", "hcl", "nix", "vue", "svelte", "astro", "cmake", "mk", "makefile",
  "gradle", "groovy", "bat", "cmd", "ps1", "ini", "conf", "cfg", "properties", "env",
  "gitignore", "gitattributes", "gitconfig", "dockerfile", "diff", "patch",
  "tex", "latex", "sty", "cls", "bib", "rst", "adoc",
]);

export function classifyFileIconKind(pathOrName: string): Exclude<FileIconKind, "folder"> {
  const slash = Math.max(pathOrName.lastIndexOf("/"), pathOrName.lastIndexOf("\\"));
  const base = pathOrName.slice(slash + 1).toLowerCase();
  if (base === "readme" || base === "changelog" || base === "contributing") return "markdown";
  if (
    base === "dockerfile" || base.startsWith("dockerfile.") ||
    base === "makefile" || base === "gnumakefile" || base === "cmakelists.txt" ||
    base.startsWith(".env") || base.startsWith(".git") || base === ".bashrc" || base === ".zshrc"
  ) {
    return "code";
  }
  const dot = base.lastIndexOf(".");
  const ext = dot >= 0 ? base.slice(dot + 1) : "";
  if (Object.hasOwn(EXT_ICON_KIND, ext)) return EXT_ICON_KIND[ext]!;
  if (CODE_ICON_EXTS.has(ext)) return "code";
  return "other";
}

const DSH_FILE_BODY = "M8.48924 28H19.5108C21.6479 28 22.7165 28 23.5594 27.6509C24.6833 27.1853 25.5762 26.2924 26.0417 25.1685C26.3909 24.3256 26.3909 23.257 26.3909 21.1199V8.79443C26.3909 8.32877 26.3909 8.09593 26.3471 7.87507C26.2887 7.58058 26.173 7.30042 26.0067 7.05048C25.882 6.86303 25.7177 6.69799 25.3893 6.36792L20.0611 1.01354C19.7304 0.681235 19.5651 0.515081 19.3769 0.38885C19.126 0.220541 18.8443 0.103463 18.5481 0.0443412C18.3259 0 18.0915 0 17.6226 0H8.48924C6.35209 0 5.28351 0 4.4406 0.349145C3.31672 0.814671 2.4238 1.70759 1.95828 2.83147C1.60913 3.67438 1.60913 4.74296 1.60913 6.88011V21.1199C1.60913 23.257 1.60913 24.3256 1.95828 25.1685C2.4238 26.2924 3.31672 27.1853 4.4406 27.6509C5.28351 28 6.35209 28 8.48924 28Z";
const DSH_FILE_FOLD = "M26.3909 7.37445L19.0525 0V3.77445C19.0525 4.89271 19.0525 5.45184 19.2352 5.89289C19.4788 6.48096 19.946 6.94818 20.5341 7.19176C20.9751 7.37445 21.5342 7.37445 22.6525 7.37445H26.3909Z";
const DSH_MARK_STD = "translate(14 16) scale(1.12) translate(-14 -16)";
const DSH_MARK_LG = "translate(14 16) scale(1.22) translate(-14 -16)";
const DSH_FOLDER_MARK = "translate(14 13.0693) scale(1.12) translate(-14 -13.0693)";

export function FileTypeIcon({
  kind, path, size = 16, className = "",
}: {
  kind?: FileIconKind;
  path?: string;
  size?: number;
  className?: string;
}) {
  const resolved: FileIconKind = kind ?? classifyFileIconKind(path || "");
  const cls = `ft-icon ft-${resolved}` + (className ? ` ${className}` : "");
  if (resolved === "folder") {
    return (
      <svg width={size} height={size} className={cls} viewBox="0 0 28 28" fill="none" aria-hidden="true">
        <path
          d="M2.80078 10.2112C2.80078 9.52802 2.80078 9.18642 2.85314 8.866C2.98648 8.05 3.36942 7.29538 3.94925 6.70595C4.17694 6.4745 4.45264 6.2728 5.00404 5.86939C5.29197 5.65874 5.43594 5.55342 5.58711 5.46632C5.97089 5.24521 6.39638 5.10618 6.83667 5.05803C7.01011 5.03906 7.18849 5.03906 7.54524 5.03906H11.6543C12.4669 5.03906 12.8732 5.03906 13.2599 5.13697C13.3882 5.16945 13.5143 5.20986 13.6375 5.25795C14.0091 5.40294 14.3398 5.63902 15.0012 6.1112L16.2632 7.01224C16.5526 7.21881 16.6972 7.3221 16.8598 7.38553C16.9137 7.40657 16.9689 7.42425 17.025 7.43846C17.1942 7.4813 17.372 7.4813 17.7275 7.4813H19.8008C22.0506 7.4813 23.1755 7.4813 23.9641 8.05425C24.2188 8.23928 24.4428 8.46326 24.6278 8.71794C25.2008 9.50654 25.2008 10.6315 25.2008 12.8813V18.7283C25.2008 20.9781 25.2008 22.1031 24.6278 22.8917C24.4428 23.1463 24.2188 23.3703 23.9641 23.5554C23.1755 24.1283 22.0506 24.1283 19.8008 24.1283H8.20077C5.95094 24.1283 4.82602 24.1283 4.03743 23.5554C3.78274 23.3703 3.55877 23.1463 3.37373 22.8917C2.80078 22.1031 2.80078 20.9781 2.80078 18.7283V10.2112Z"
          fill="currentColor"
        />
        <g color="#ffffff" transform={DSH_FOLDER_MARK}>
          <path d="M6.31445 13.0693H21.6893" stroke="currentColor" strokeWidth="2.38" />
        </g>
      </svg>
    );
  }
  const muted = resolved === "other";
  let mark: ReactNode = null;
  let transform: string | undefined;
  if (resolved === "code") {
    mark = (
      <>
        <path d="M10.0053 13.126L7.0236 16.3788C6.96052 16.4476 6.96052 16.5532 7.0236 16.622L10.0053 19.8748" stroke="currentColor" strokeWidth="1.35" />
        <path d="M17.9941 13.126L20.9759 16.3788C21.039 16.4476 21.039 16.5532 20.9759 16.622L17.9941 19.8748" stroke="currentColor" strokeWidth="1.35" />
        <path d="M15.2652 12.957L12.7344 20.0433" stroke="currentColor" strokeWidth="1.35" />
      </>
    );
  } else if (resolved === "excel") {
    mark = (
      <path
        d="M14 11.5H11.4C10.5599 11.5 10.1399 11.5 9.81901 11.6635C9.53677 11.8073 9.3073 12.0368 9.16349 12.319C9 12.6399 9 13.0599 9 13.9V16.5M14 11.5H16.6C17.4401 11.5 17.8601 11.5 18.181 11.6635C18.4632 11.8073 18.6927 12.0368 18.8365 12.319C19 12.6399 19 13.0599 19 13.9V16.5M14 11.5V21.5M14 21.5H16.6C17.4401 21.5 17.8601 21.5 18.181 21.3365C18.4632 21.1927 18.6927 20.9632 18.8365 20.681C19 20.3601 19 19.9401 19 19.1V16.5M14 21.5H11.4C10.5599 21.5 10.1399 21.5 9.81901 21.3365C9.53677 21.1927 9.3073 20.9632 9.16349 20.681C9 20.3601 9 19.9401 9 19.1V16.5M19 16.5H9"
        stroke="currentColor"
        strokeWidth="1.2"
      />
    );
  } else if (resolved === "html") {
    transform = DSH_MARK_STD;
    mark = (
      <path
        fillRule="evenodd"
        clipRule="evenodd"
        d="M13.9994 9.68298C17.212 9.68298 19.8167 12.2872 19.8168 15.4997C19.8168 18.7123 17.2121 21.3171 13.9994 21.3171C10.7869 21.3169 8.18274 18.7122 8.18274 15.4997C8.1829 12.2873 10.787 9.68315 13.9994 9.68298ZM9.26213 16.0247C9.47025 17.9241 10.7936 19.4876 12.5639 20.0463C12.42 19.7977 12.2952 19.5152 12.1879 19.2116C11.885 18.3541 11.693 17.2434 11.6424 16.0247H9.26213ZM16.3565 16.0247C16.3059 17.2434 16.1145 18.3542 15.8116 19.2116C15.7044 19.5151 15.5788 19.7971 15.435 20.0456C17.2054 19.487 18.5293 17.9242 18.7374 16.0247H16.3565ZM12.6938 16.0247C12.7439 17.1459 12.9212 18.1334 13.1784 18.8616C13.332 19.2962 13.503 19.61 13.6686 19.805C13.834 19.9996 13.9473 20.0256 13.9994 20.0258C14.0514 20.0258 14.1651 20.0002 14.331 19.805C14.4966 19.61 14.6676 19.2962 14.8211 18.8616C15.0784 18.1334 15.2557 17.1459 15.3058 16.0247H12.6938ZM13.9994 10.733C13.9473 10.7331 13.834 10.7598 13.6686 10.9545C13.503 11.1494 13.3319 11.4633 13.1784 11.8978C12.903 12.6777 12.7188 13.7545 12.6849 14.9747H15.3147C15.2808 13.7545 15.0966 12.6777 14.8211 11.8978C14.6676 11.4633 14.4965 11.1494 14.331 10.9545C14.1651 10.7593 14.0514 10.733 13.9994 10.733ZM15.5888 11.0051C15.6701 11.1756 15.7444 11.3576 15.8116 11.5478C16.1343 12.4613 16.3308 13.6619 16.3647 14.9747H18.7374C18.5352 13.1307 17.2817 11.6036 15.5888 11.0051ZM12.4101 11.0051C10.7174 11.6037 9.46428 13.1308 9.26213 14.9747H11.6349C11.6688 13.6619 11.8652 12.4613 12.1879 11.5478C12.2551 11.3577 12.3288 11.1756 12.4101 11.0051Z"
        fill="currentColor"
      />
    );
  } else if (resolved === "image") {
    transform = DSH_MARK_STD;
    mark = (
      <>
        <path d="M10.4212 15.9204C10.5756 15.6558 10.9579 15.6558 11.1123 15.9204L13.6493 20.2696C13.8048 20.5362 13.6125 20.8711 13.3037 20.8711H8.22974C7.92102 20.8711 7.72868 20.5362 7.88423 20.2696L10.4212 15.9204Z" fill="currentColor" />
        <path d="M15.4981 13.186C15.6505 12.9117 16.0451 12.9117 16.1975 13.186L20.1368 20.2769C20.2849 20.5435 20.0922 20.8711 19.7872 20.8711H11.9084C11.6034 20.8711 11.4107 20.5435 11.5588 20.2769L15.4981 13.186Z" fill="currentColor" />
        <path d="M11.8603 11.3997C11.8603 12.286 11.1418 13.0045 10.2555 13.0045C9.36924 13.0045 8.65076 12.286 8.65076 11.3997C8.65076 10.5134 9.36924 9.79492 10.2555 9.79492C11.1418 9.79492 11.8603 10.5134 11.8603 11.3997Z" fill="currentColor" />
      </>
    );
  } else if (resolved === "markdown") {
    transform = DSH_MARK_LG;
    mark = (
      <path d="M8.7588 19.5V14.6H9.8998L11.9298 17.932H11.3278L13.3018 14.6H14.4428L14.4568 19.5H13.1828L13.1688 16.539H13.3858L11.9088 19.017H11.2928L9.7738 16.539H10.0398V19.5H8.7588ZM15.4375 19.5V14.6H17.7545C18.2958 14.6 18.7718 14.7003 19.1825 14.901C19.5932 15.1017 19.9128 15.384 20.1415 15.748C20.3748 16.112 20.4915 16.546 20.4915 17.05C20.4915 17.5493 20.3748 17.9833 20.1415 18.352C19.9128 18.716 19.5932 18.9983 19.1825 19.199C18.7718 19.3997 18.2958 19.5 17.7545 19.5H15.4375ZM16.8235 18.394H17.6985C17.9785 18.394 18.2212 18.3427 18.4265 18.24C18.6365 18.1327 18.7998 17.9787 18.9165 17.778C19.0332 17.5727 19.0915 17.33 19.0915 17.05C19.0915 16.7653 19.0332 16.5227 18.9165 16.322C18.7998 16.1213 18.6365 15.9697 18.4265 15.867C18.2212 15.7597 17.9785 15.706 17.6985 15.706H16.8235V18.394Z" fill="currentColor" />
    );
  } else if (resolved === "pdf") {
    transform = DSH_MARK_LG;
    mark = (
      <path d="M6.80616 19.5V14.6H9.04616C9.49416 14.6 9.87916 14.6723 10.2012 14.817C10.5278 14.9617 10.7798 15.1717 10.9572 15.447C11.1345 15.7177 11.2232 16.0397 11.2232 16.413C11.2232 16.7817 11.1345 17.1013 10.9572 17.372C10.7798 17.6427 10.5278 17.8527 10.2012 18.002C9.87916 18.1467 9.49416 18.219 9.04616 18.219H7.57616L8.19216 17.617V19.5H6.80616ZM8.19216 17.764L7.57616 17.127H8.96216C9.2515 17.127 9.46616 17.064 9.60616 16.938C9.75083 16.812 9.82316 16.637 9.82316 16.413C9.82316 16.1843 9.75083 16.007 9.60616 15.881C9.46616 15.755 9.2515 15.692 8.96216 15.692H7.57616L8.19216 15.055V17.764ZM11.8989 19.5V14.6H14.2159C14.7573 14.6 15.2333 14.7003 15.6439 14.901C16.0546 15.1017 16.3743 15.384 16.6029 15.748C16.8363 16.112 16.9529 16.546 16.9529 17.05C16.9529 17.5493 16.8363 17.9833 16.6029 18.352C16.3743 18.716 16.0546 18.9983 15.6439 19.199C15.2333 19.3997 14.7573 19.5 14.2159 19.5H11.8989ZM13.2849 18.394H14.1599C14.4399 18.394 14.6826 18.3427 14.8879 18.24C15.0979 18.1327 15.2613 17.9787 15.3779 17.778C15.4946 17.5727 15.5529 17.33 15.5529 17.05C15.5529 16.7653 15.4946 16.5227 15.3779 16.322C15.2613 16.1213 15.0979 15.9697 14.8879 15.867C14.6826 15.7597 14.4399 15.706 14.1599 15.706H13.2849V18.394ZM17.6821 19.5V14.6H21.5251V15.671H19.0681V19.5H17.6821ZM18.9701 17.82V16.749H21.2311V17.82H18.9701Z" fill="currentColor" />
    );
  } else if (resolved === "ppt") {
    transform = DSH_MARK_LG;
    mark = (
      <path d="M11.0132 20.5V13.5H14.2132C14.8532 13.5 15.4032 13.6033 15.8632 13.81C16.3299 14.0167 16.6899 14.3167 16.9432 14.71C17.1966 15.0967 17.3232 15.5567 17.3232 16.09C17.3232 16.6167 17.1966 17.0733 16.9432 17.46C16.6899 17.8467 16.3299 18.1467 15.8632 18.36C15.4032 18.5667 14.8532 18.67 14.2132 18.67H12.1132L12.9932 17.81V20.5H11.0132ZM12.9932 18.02L12.1132 17.11H14.0932C14.5066 17.11 14.8132 17.02 15.0132 16.84C15.2199 16.66 15.3232 16.41 15.3232 16.09C15.3232 15.7633 15.2199 15.51 15.0132 15.33C14.8132 15.15 14.5066 15.06 14.0932 15.06H12.1132L12.9932 14.15V18.02Z" fill="currentColor" />
    );
  } else if (resolved === "video") {
    transform = DSH_MARK_STD;
    mark = (
      <path d="M17.5 14.634C18.1667 15.0189 18.1667 15.9811 17.5 16.366L11.5 19.8301C10.8333 20.215 10 19.7339 10 18.9641L10 12.0359C10 11.2661 10.8333 10.785 11.5 11.1699L17.5 14.634Z" fill="currentColor" />
    );
  } else if (resolved === "word") {
    transform = DSH_MARK_LG;
    mark = (
      <path d="M10.5118 20.5L8.24179 13.5H10.2818L12.1918 19.56H11.1618L13.1718 13.5H14.9918L16.8918 19.56H15.9018L17.8718 13.5H19.7618L17.4918 20.5H15.3718L13.7518 15.35H14.3218L12.6318 20.5H10.5118Z" fill="currentColor" />
    );
  }
  return (
    <svg width={size} height={size} className={cls} viewBox="0 0 28 28" fill="none" aria-hidden="true">
      <path d={DSH_FILE_BODY} fill="currentColor" />
      {muted
        ? <path d={DSH_FILE_FOLD} fill="#9ca3af" />
        : <path d={DSH_FILE_FOLD} fill="#ffffff" fillOpacity=".7" />}
      {mark && <g color="#ffffff" transform={transform}>{mark}</g>}
    </svg>
  );
}
