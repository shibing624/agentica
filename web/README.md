# Agentica Web SPA

Vite + React + TypeScript UI for the Gateway. Production builds land in
`agentica/gateway/ui/` and ship inside `agentica[gateway]`. Runtime does not
need Node.

## Develop (two processes)

`agentica-gateway` on PATH is the **editable install of the main checkout**,
not this worktree. Starting it from here still serves the old petite-vue
`/static` page. Run the Gateway as a module from this tree:

```bash
cd /path/to/agentica/.worktrees/web-v2
PYTHONPATH=. python -m agentica.gateway.main
```

Then, in another terminal:

```bash
cd web
npm install
npm run dev          # http://localhost:5173  (proxies /api to :8881)
```

Open `http://localhost:5173/chat`, not `:8881/chat`, while Vite is running.
`:8881/chat` is the built dist (run `npm run build` to refresh it).

```bash
npm run build        # writes ../agentica/gateway/ui
```
