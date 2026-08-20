import { useEffect } from "react";
import { BrowserRouter, Navigate, Route, Routes, useLocation, useNavigate } from "react-router";
import { setUnauthorizedHandler } from "./api";
import { ChatPage } from "./pages/ChatPage";
import { LoginPage } from "./pages/LoginPage";
import { TracesPage } from "./pages/TracesPage";
import { ConfirmDialog, Toast } from "./components/Feedback";
import { applyTheme, getState } from "./store";

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
  // On <html>, so a deep link straight to /traces is themed too.
  useEffect(() => { applyTheme(getState().theme); }, []);
  return (
    <BrowserRouter>
      <AuthGate />
      <Routes>
        <Route path="/" element={<Navigate to="/chat" replace />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/traces" element={<TracesPage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="*" element={<Navigate to="/chat" replace />} />
      </Routes>
      <Toast />
      <ConfirmDialog />
    </BrowserRouter>
  );
}
