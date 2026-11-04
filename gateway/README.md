# agentica-gateway

The server half of [agentica](https://github.com/shibing624/agentica): a FastAPI
gateway that serves the Web UI, drives IM channels (Feishu, WeChat, WeCom,
DingTalk, Telegram, Discord, Slack, QQ) and runs scheduled jobs, all on top of
the `agentica` agent runtime.

```bash
pip install agentica-gateway                 # server + Web UI + cron
pip install "agentica-gateway[feishu]"       # + one IM channel
pip install "agentica-gateway[all]"          # every IM channel
agentica-gateway                             # http://127.0.0.1:8881/chat
```

`pip install "agentica[gateway]"` installs the same thing. With uv, install the
gateway itself so its command is exposed: `uv tool install agentica-gateway`.

`agentica-gateway` is released together with `agentica` and pins the exact same
version. The Web UI is built from `web/` and bundled into the wheel, so no Node
is needed at runtime.

Docs: <https://shibing624.github.io/agentica/advanced/gateway/>
