"use strict";
/**
 * Why the desktop app remembers a port.
 *
 * The SPA keeps the session tree, the active session and the theme in
 * `localStorage`, and `localStorage` is scoped to the origin. The origin is
 * `http://127.0.0.1:<port>`. So `--port 0` — correct as an allocator, since a
 * fixed number would collide with the user's own `agentica-gateway` on 8881 —
 * silently hands the user an empty sidebar on every single launch: same data
 * root, same sessions on disk, different origin, no memory of which one they
 * were in.
 *
 * The fix is not a fixed port. It is: ask for the port we got last time, and
 * fall back to 0 when something else has taken it. First launch, and any launch
 * after a collision, still costs the user their local preferences once — that is
 * the price of not owning a port number, and it is paid rarely instead of
 * always.
 *
 * The memory lives in Electron's `userData`, not under `~/.agentica`: it is
 * shell state, meaningless to the gateway and to the CLI.
 */

const fs = require("node:fs");
const net = require("node:net");
const path = require("node:path");
const { parsePort } = require("./util");

/** The port to prefer this launch, or null when there is no memory. */
function readPreferredPort(file) {
  try {
    return parsePort(fs.readFileSync(file, "utf8"));
  } catch {
    return null;
  }
}

/** Remember the port the gateway actually bound. */
function rememberPort(file, port) {
  try {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, `${port}\n`);
  } catch {
    // Best-effort: losing the memory costs one launch's localStorage, not
    // correctness, and there is nothing useful to say about it here.
  }
}

/**
 * Whether binding host:port would succeed. Only EADDRINUSE / EACCES veto it —
 * any other error (no IPv6 stack, for instance) says nothing about whether the
 * port is free.
 */
function portIsFree(port, host) {
  return new Promise((resolve) => {
    const probe = net.createServer();
    probe.unref();
    probe.once("error", (err) => {
      resolve(err.code !== "EADDRINUSE" && err.code !== "EACCES");
    });
    probe.listen({ port, host, exclusive: true }, () => {
      probe.close(() => resolve(true));
    });
  });
}

/**
 * The port to ask the gateway for: last launch's when still free, else 0.
 *
 * Both loopback stacks are probed. The gateway binds 127.0.0.1 and the window
 * asks for 127.0.0.1, but a foreign listener on `::1` at the same port is a
 * real hazard for anything that resolves `localhost` — and the check is free.
 */
async function choosePort(preferred) {
  if (preferred === null) return 0;
  const free = await Promise.all([
    portIsFree(preferred, "127.0.0.1"),
    portIsFree(preferred, "::1"),
  ]);
  return free.every(Boolean) ? preferred : 0;
}

module.exports = { readPreferredPort, rememberPort, choosePort, portIsFree };
