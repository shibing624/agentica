import { useEffect } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router";
import { ChatPage } from "./pages/ChatPage";
import { TracesPage } from "./pages/TracesPage";
import { ConfirmDialog, Toast } from "./components/Feedback";
import { applyTheme, getState } from "./store";

export function App() {
  // On <html>, so a deep link straight to /traces is themed too.
  useEffect(() => { applyTheme(getState().theme); }, []);
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/chat" replace />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/traces" element={<TracesPage />} />
        <Route path="*" element={<Navigate to="/chat" replace />} />
      </Routes>
      <Toast />
      <ConfirmDialog />
    </BrowserRouter>
  );
}
