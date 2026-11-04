# Agentica Web SPA

Vite + React + TypeScript UI for the Gateway. Source is this directory, and
`npm run build` writes **`web/dist/`** (gitignored) — nothing is written into a
Python source tree.

Who picks the dist up:

- **The `agentica-gateway` wheel/sdist.** `gateway/hatch_build.py` copies
  `web/dist` to `agentica_gateway/ui` at build time, and fails the build when
  it is missing, so a release can never ship a gateway without its UI. pip
  users therefore never need Node.
- **A source checkout.** An editable gateway has no copy in the package; it
  serves `web/dist` straight from the checkout (`agentica_gateway/ui_dist.py`).
- **Anything else.** `AGENTICA_WEB_DIST=/path/to/dist` overrides both.

A checkout without a build still starts the API; `:8881/chat` returns 503
until you build (or until you use Vite below).

## Develop (two processes)

From the repo root, once:

```bash
pip install -e . -e ./gateway          # agentica + agentica-gateway, editable
```

Then:

```bash
agentica-gateway                       # :8881, this checkout's code
```

and in another terminal:

```bash
cd web
npm install
npm run dev          # http://localhost:5173  (proxies /api to :8881)
```

Open `http://localhost:5173/chat`, not `:8881/chat`, while Vite is running.
`:8881/chat` serves `web/dist`, which only changes when you build:

```bash
npm run build        # tsc --noEmit, then writes web/dist
```

Refresh the page; no copy step and no gateway restart.
