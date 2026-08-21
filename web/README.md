# Agentica Web SPA

Vite + React + TypeScript UI for the Gateway. Source is this directory.
`npm run build` writes `agentica/gateway/ui/` (gitignored). Release / CI
run that before `python -m build`, so `agentica[gateway]` on PyPI already
contains the dist. Runtime does not need Node.

A checkout without a build still starts the API; `:8881/chat` returns 503
until you build (or until you use Vite below).

## Develop (two processes)

`agentica-gateway` on PATH is whichever checkout is `pip install -e`'d. To be
sure you are running *this* tree, start it as a module from the repo root:

```bash
cd /path/to/agentica
PYTHONPATH=. python -m agentica.gateway.main
```

Then, in another terminal:

```bash
cd web
npm install
npm run dev          # http://localhost:5173  (proxies /api to :8881)
```

Open `http://localhost:5173/chat`, not `:8881/chat`, while Vite is running.
`:8881/chat` serves `agentica/gateway/ui/`, which only changes when you build:

```bash
npm run build        # tsc --noEmit, then writes ../agentica/gateway/ui
```

Do not commit that directory. A release wheel is built from a tree that
already ran `npm run build`; `MANIFEST.in` pulls `gateway/ui/` into the
sdist even though git ignores it.
