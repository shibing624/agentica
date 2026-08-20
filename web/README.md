# Agentica Web SPA

Vite + React + TypeScript UI for the Gateway. Production builds land in
`agentica/gateway/ui/` and ship inside `agentica[gateway]`. Runtime does not
need Node.

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
`:8881/chat` serves the built output in `agentica/gateway/ui/`, which only
changes when you build:

```bash
npm run build        # tsc --noEmit, then writes ../agentica/gateway/ui
```
