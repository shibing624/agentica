/**
 * Each chat has its own model. The gateway resolves it (the chat's own pick,
 * else its project's default, else config.yaml's) and remembers a pick in the
 * session sidecar, so reopening the chat, another tab, or the desktop app all
 * see the same model. Picking one here never touches another chat.
 */
import * as api from "./api";
import { getStrings } from "./i18n";
import { getState, setState, showToast } from "./store";

type View = NonNullable<ReturnType<typeof getState>["sessionModel"]>;

function view(id: string, m: api.SessionModel): View {
  return {
    id, profile: m.profile, source: m.source,
    provider: m.model_provider, model: m.model_name, effort: m.reasoning_effort,
  };
}

function workDirOf(id: string): string {
  const st = getState();
  return st.sessions[id]?.dir || st.pendingNewChatDir || st.serverDir || "";
}

let seq = 0;

/** Load the open chat's model (call on every chat switch). */
export async function loadSessionModel(id: string | null) {
  const nonce = ++seq;
  if (!id) { setState({ sessionModel: null }); return; }
  const { ok, data } = await api.fetchSessionModel(id, workDirOf(id));
  if (nonce !== seq || getState().curSess !== id) return;
  setState({ sessionModel: ok && data ? view(id, data) : null });
}

/** Pick a profile for the open chat; a new-chat screen holds it until the
 *  chat exists (see {@link applyPendingModel}). */
export async function pickSessionModel(profile: string) {
  const st = getState();
  const S = getStrings();
  setState({ modelDDOpen: false });
  const id = st.curSess;
  if (!id) { setState({ pendingModel: profile }); return; }
  if (st.sessionModel?.id === id && st.sessionModel.source === "session" && st.sessionModel.profile === profile) return;
  const nonce = ++seq;
  const { ok, data } = await api.setSessionModelApi(id, profile, workDirOf(id));
  if (!ok || !data) { showToast((data as any)?.detail || S.settings.switchFailed, 3000); return; }
  if (nonce === seq && getState().curSess === id) setState({ sessionModel: view(id, data) });
  showToast(S.settings.switchedTo(profile));
}

/** The first message of a new chat: record the model picked before it existed. */
export async function applyPendingModel(id: string) {
  const profile = getState().pendingModel;
  if (!profile) return;
  setState({ pendingModel: "" });
  const nonce = ++seq;
  const { ok, data } = await api.setSessionModelApi(id, profile, workDirOf(id));
  if (ok && data && nonce === seq && getState().curSess === id) setState({ sessionModel: view(id, data) });
}
