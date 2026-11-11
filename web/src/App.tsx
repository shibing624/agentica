import { lazy, Suspense, useEffect } from "react";
import { BrowserRouter, Navigate, Route, Routes, useLocation, useNavigate } from "react-router";
import { setUnauthorizedHandler } from "./api";
import { ChatPage } from "./pages/ChatPage";
import { LoginPage } from "./pages/LoginPage";
const AutomationsPage = lazy(() => import("./pages/AutomationsPage").then((m) => ({ default: m.AutomationsPage })));
const TracesPage = lazy(() => import("./pages/TracesPage").then((m) => ({ default: m.TracesPage })));
import { AppLayout } from "./components/AppShell";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { ConfirmDialog, Toast } from "./components/Feedback";
import { applyLang } from "./i18n";
import { primeSettings } from "./panels/SettingsModal";
import { applyTheme, getState } from "./store";

/** Legacy standalone routes (`/users`, `/assistant`, `/plugins`) open their
 *  matching tab inside the unified Settings modal over the chat workspace. */
function SettingsRoute({ tab }: { tab: string }) {
  useEffect(() => {
    void primeSettings(tab);
  }, [tab]);
  return <Navigate to="/chat" replace />;
}

/** Any 401 means the same thing — the session expired, or the gateway was
 *  restarted without one — and the same answer: the login page, remembering
 *  where the user was. Registered here because `api.ts` must not know about
 *  the router. */
function AuthGate() {
  const nav = useNavigate();
  const loc = useLocation();
  useEffect(() => {
    setUnauthorizedHandler(() => {
      if (loc.pathname === "/login") return;
      nav(`/login?next=${encodeURIComponent(loc.pathname + loc.search)}`, { replace: true });
    });
  }, [nav, loc.pathname, loc.search]);
  return null;
}

export function App() {
  // Both on <html>, so a deep link straight to /traces is themed and tagged too.
  useEffect(() => {
    applyTheme(getState().theme);
    applyLang(getState().lang);
  }, []);
  return (
    <BrowserRouter>
      <AuthGate />
      <ErrorBoundary>
        <Suspense fallback={null}>
          <Routes>
            <Route path="/" element={<Navigate to="/chat" replace />} />
            <Route element={<AppLayout />}>
              <Route path="/chat" element={<ChatPage />} />
              <Route path="/traces" element={<TracesPage />} />
              <Route path="/automations" element={<AutomationsPage />} />
              <Route path="/users" element={<SettingsRoute tab="users" />} />
              <Route path="/assistant" element={<SettingsRoute tab="assistant" />} />
              <Route path="/plugins" element={<SettingsRoute tab="plugins" />} />
            </Route>
            <Route path="/login" element={<LoginPage />} />
            <Route path="*" element={<Navigate to="/chat" replace />} />
          </Routes>
        </Suspense>
      </ErrorBoundary>
      <Toast />
      <ConfirmDialog />
    </BrowserRouter>
  );
}
