import type { ReactNode, SVGProps } from "react";

export const LOGO_SRC =
  "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Cdefs%3E%3ClinearGradient id='bg' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' stop-color='%234A90D9'/%3E%3Cstop offset='100%25' stop-color='%23A8D8EA'/%3E%3C/linearGradient%3E%3ClinearGradient id='body' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' stop-color='%23E8F4FD'/%3E%3Cstop offset='100%25' stop-color='%23d0e8f8'/%3E%3C/linearGradient%3E%3C/defs%3E%3Ccircle cx='50' cy='50' r='48' fill='url(%23bg)'/%3E%3Cpath d='M28 38 L22 18 L38 30 Z' fill='%23E8F4FD'/%3E%3Cpath d='M72 38 L78 18 L62 30 Z' fill='%23E8F4FD'/%3E%3Cpath d='M28 38 L22 18 L38 30' fill='none' stroke='%23d0e8f8' stroke-width='1'/%3E%3Cpath d='M72 38 L78 18 L62 30' fill='none' stroke='%23d0e8f8' stroke-width='1'/%3E%3Cellipse cx='50' cy='52' rx='28' ry='28' fill='url(%23body)'/%3E%3Ccircle cx='39' cy='46' r='4.5' fill='%232C3E50'/%3E%3Ccircle cx='61' cy='46' r='4.5' fill='%232C3E50'/%3E%3Ccircle cx='40.2' cy='44.8' r='1.5' fill='white'/%3E%3Ccircle cx='62.2' cy='44.8' r='1.5' fill='white'/%3E%3Cellipse cx='50' cy='55' rx='3.5' ry='2' fill='%23F4A0B0'/%3E%3Cpath d='M46.5 55 Q50 59 53.5 55' stroke='%232C3E50' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3Cline x1='28' y1='52' x2='16' y2='48' stroke='%23ccc' stroke-width='1' stroke-linecap='round'/%3E%3Cline x1='28' y1='55' x2='16' y2='55' stroke='%23ccc' stroke-width='1' stroke-linecap='round'/%3E%3Cline x1='72' y1='52' x2='84' y2='48' stroke='%23ccc' stroke-width='1' stroke-linecap='round'/%3E%3Cline x1='72' y1='55' x2='84' y2='55' stroke='%23ccc' stroke-width='1' stroke-linecap='round'/%3E%3C/svg%3E";

export function Logo({ className = "brand-logo" }: { className?: string }) {
  return <img className={className} src={LOGO_SRC} alt="logo" />;
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
export const IconSearch = () => (
  <S strokeWidth="1.4"><circle cx="7" cy="7" r="4.25" /><path d="M10.2 10.2l3.05 3.05" strokeLinecap="round" /></S>
);
export const IconClose = () => (
  <S strokeWidth="1.5" strokeLinecap="round"><path d="M4 4l8 8M12 4l-8 8" /></S>
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
export const IconTrace = () => (
  <S strokeWidth="1.3" strokeLinecap="round">
    <path d="M2 12.5h3.2M7.2 12.5h6.8" />
    <rect x="2" y="3" width="3.2" height="7" rx="1" />
    <rect x="7.2" y="5.5" width="2.6" height="4.5" rx="1" />
    <rect x="11.4" y="4" width="2.6" height="6" rx="1" />
  </S>
);
export const IconSidebar = () => (
  <S strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
    <rect x="1.5" y="2.5" width="13" height="11" rx="2" />
    <line x1="5.5" y1="2.5" x2="5.5" y2="13.5" />
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
export const IconUser = () => (
  <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor">
    <circle cx="8" cy="5.2" r="2.9" />
    <path d="M2.4 14c.5-3.2 3-5 5.6-5s5.1 1.8 5.6 5z" />
  </svg>
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
export const IconTerminal = () => (
  <svg viewBox="0 0 16 16" width="13" height="13" fill="currentColor">
    <path d="M0 2.75C0 1.784.784 1 1.75 1h12.5c.966 0 1.75.784 1.75 1.75v10.5A1.75 1.75 0 0114.25 15H1.75A1.75 1.75 0 010 13.25V2.75zm7 7.5a.75.75 0 000 1.5h4.25a.75.75 0 000-1.5H7zm-3.22-4.97a.75.75 0 00-1.06 1.06L4.44 8 2.72 9.72a.75.75 0 101.06 1.06l2-2a.75.75 0 000-1.06l-2-2z" />
  </svg>
);
export const IconAsk = () => (
  <S strokeWidth="1.3">
    <circle cx="8" cy="8" r="6.25" />
    <path d="M6.2 6.1a1.9 1.9 0 013.6.9c0 1.3-1.8 1.5-1.8 2.9" strokeLinecap="round" />
    <circle cx="8" cy="11.7" r="0.2" fill="currentColor" stroke="none" />
  </S>
);
export const IconAuto = () => (
  <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor" stroke="none">
    <path d="M8.6 1.5L3 9.2h3.6L6.2 14.5 12.5 6.3H8.8z" />
  </svg>
);
export const IconAllowAll = () => (
  <S strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4.5 7.3V5.1a3.5 3.5 0 016.7-1.35" />
    <rect x="3" y="7.3" width="10" height="6.5" rx="1.3" />
  </S>
);
export const IconSend = () => (
  <svg viewBox="0 0 16 16" width="16" height="16">
    <path d="M8 13V4M8 4L4.3 7.7M8 4l3.7 3.7" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);
export const IconStop = () => (
  <svg viewBox="0 0 16 16" width="16" height="16">
    <rect x="4.5" y="4.5" width="7" height="7" rx="1.8" fill="currentColor" />
  </svg>
);
export const IconFolderPlus = () => (
  <S strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 4.5h3l1.2 1.5H13a1 1 0 011 1v4.5a1.5 1.5 0 01-1.5 1.5h-9A1.5 1.5 0 012 11.5v-6A1 1 0 013 4.5z" />
  </S>
);
