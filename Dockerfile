# syntax=docker/dockerfile:1

# Web UI is compiled here, then copied into the Python image and into the
# wheel path (agentica/gateway/ui). Runtime still does not need Node.
FROM node:22-bookworm-slim AS web

WORKDIR /src/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
# vite.config.ts writes ../agentica/gateway/ui
RUN mkdir -p /src/agentica/gateway && npm run build

FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8881 \
    AGENTICA_HOME=/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /data /workspace

COPY pyproject.toml README.md LICENSE /app/
COPY agentica /app/agentica
COPY --from=web /src/agentica/gateway/ui /app/agentica/gateway/ui

RUN pip install --no-cache-dir ".[gateway]" \
    && rm -rf /root/.cache/pip

# Bind-mounted /workspace must stay writable; compose publishes only on loopback.
ENV HOME=/data
WORKDIR /workspace

EXPOSE 8881

HEALTHCHECK --interval=10s --timeout=5s --retries=5 --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8881/api/health', timeout=2).read()"

CMD ["agentica-gateway"]
