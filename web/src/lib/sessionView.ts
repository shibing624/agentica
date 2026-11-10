/** Where the user left each session: which page (chat or trace) and where on
 *  it. Reopening a session — from the sidebar, the Dock, or a fresh start —
 *  returns there instead of always landing at the bottom of the chat. */

export type SessionView = {
  page?: "chat" | "traces";
  /** Distance from the bottom, not from the top: older pages are prepended,
   *  so this stays put while they load and across a reload that starts from
   *  the newest page again. */
  chat?: { fromBottom: number; stick: boolean };
  trace?: { call?: number; scrollTop?: number };
};

type ViewStorage = Pick<Storage, "getItem" | "setItem">;

/** Oldest entries are dropped past this; a view is only worth keeping for
 *  sessions the user still moves between. */
export const MAX_VIEWS = 200;
const FLUSH_MS = 300;

export function createViewStore(storage: ViewStorage, key: () => string) {
  let loaded: { key: string; views: Map<string, SessionView> } | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;

  const views = () => {
    const k = key();
    if (loaded?.key !== k) {
      const raw = JSON.parse(storage.getItem(k) || "[]") as [string, SessionView][];
      loaded = { key: k, views: new Map(raw) };
    }
    return loaded.views;
  };

  const flush = () => {
    if (timer !== null) clearTimeout(timer);
    timer = null;
    if (!loaded) return;
    try {
      storage.setItem(loaded.key, JSON.stringify([...loaded.views]));
    } catch (e) {
      // Like the session cache: losing a scroll position must not break the page.
      console.warn("session view not saved:", e);
    }
  };

  return {
    get(id: string): SessionView | undefined {
      return views().get(id);
    },
    update(id: string, patch: SessionView) {
      const map = views();
      const prev = map.get(id);
      const next: SessionView = { ...prev, ...patch };
      if (patch.trace) next.trace = { ...prev?.trace, ...patch.trace };
      map.delete(id);
      map.set(id, next);
      while (map.size > MAX_VIEWS) map.delete(map.keys().next().value as string);
      if (timer === null) timer = setTimeout(flush, FLUSH_MS);
    },
    flush,
  };
}
