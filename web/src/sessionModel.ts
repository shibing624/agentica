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
/** Last known model per chat, so switching back shows it at once instead of
 *  the gateway default until the request returns. */
const known = new Map<string, View>();
/** Chats whose model was asked for and answered, even if the answer failed. */
const settled = new Set<string>();

export function cachedSessionModel(id: string | null): View | null {
  return id ? known.get(id) ?? null : null;
}

/** Whether the composer can name this chat's model without guessing. */
export function sessionModelKnown(id: string | null): boolean {
  return !id || known.has(id) || settled.has(id);
}

function remember(id: string, m: api.SessionModel): View {
  const v = view(id, m);
  known.set(id, v);
  return v;
}

/** Load the open chat's model (call on every chat switch). */
export async function loadSessionModel(id: string | null) {
  const nonce = ++seq;
  if (!id) { setState({ sessionModel: null }); return; }
  const cached = known.get(id);
  if (cached && getState().sessionModel !== cached) setState({ sessionModel: cached });
  const { ok, data } = await api.fetchSessionModel(id, workDirOf(id));
  settled.add(id);
  if (nonce !== seq || getState().curSess !== id) return;
  if (ok && data) setState({ sessionModel: remember(id, data) });
  else setState({ sessionModel: cached ?? null });
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
  const picked = remember(id, data);
  if (nonce === seq && getState().curSess === id) setState({ sessionModel: picked });
  showToast(S.settings.switchedTo(profile));
}

/** The first message of a new chat: record the model picked before it existed. */
export async function applyPendingModel(id: string) {
  const profile = getState().pendingModel;
  if (!profile) return;
  setState({ pendingModel: "" });
  const nonce = ++seq;
  const { ok, data } = await api.setSessionModelApi(id, profile, workDirOf(id));
  if (!ok || !data) return;
  const picked = remember(id, data);
  if (nonce === seq && getState().curSess === id) setState({ sessionModel: picked });
}
