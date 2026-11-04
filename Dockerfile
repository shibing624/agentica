# syntax=docker/dockerfile:1

# Two stages: Node builds the Web UI into web/dist, then the Python stage builds
# the agentica and agentica-gateway wheels (the gateway wheel carries the UI,
# gateway/hatch_build.py) and installs them. The runtime image has no Node.
FROM node:22-bookworm-slim AS web

WORKDIR /src/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim-bookworm AS runtime

WORKDIR /src

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8881 \
    AGENTICA_HOME=/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /data /workspace

COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY agentica ./agentica
COPY gateway ./gateway
COPY --from=web /src/web/dist ./web/dist

# Wheels, not `pip install .`: the gateway's exact `agentica==X.Y.Z` pin must
# resolve to this checkout's core, not to whatever PyPI has. Feishu stays in,
# as it was in the old `agentica[gateway]` image.
RUN pip wheel --no-cache-dir --no-deps -w /tmp/wheels . ./gateway \
    && pip install --no-cache-dir /tmp/wheels/agentica-*.whl "$(ls /tmp/wheels/agentica_gateway-*.whl)[feishu]" \
    && rm -rf /tmp/wheels /src /root/.cache/pip

# Bind-mounted /workspace must stay writable; compose publishes only on loopback.
ENV HOME=/data
WORKDIR /workspace

EXPOSE 8881

HEALTHCHECK --interval=10s --timeout=5s --retries=5 --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8881/api/health', timeout=2).read()"

CMD ["agentica-gateway"]
