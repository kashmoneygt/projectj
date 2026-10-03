# hosted jym image, built by deploy/azure.sh
# run locally from jym/ (needs a writable data/ folder):
#   docker build -f deploy/Dockerfile -t jym .
#   docker run --rm --cpus 2 -p 8080:8080 -e JYM_INSECURE=1 -e LLAMA_THREADS=2 -v "$PWD/data:/home/data" jym

FROM node:22-slim AS ui
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY ui/ ./
RUN npm run build

FROM ghcr.io/ggml-org/llama.cpp@sha256:6257697a7f5d034b8fb499ddb07af3e250506352f94102054252a23f3b85e0af AS llama

# recorded runs replay exactly with Ubuntu 24.04's Python and math library
FROM ubuntu:24.04@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9.30 /uv /usr/local/bin/uv
COPY --from=llama /app /opt/llama
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_NO_CACHE=1 UV_PYTHON_DOWNLOADS=never PYTHONUNBUFFERED=1
WORKDIR /jym
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project --python /usr/bin/python3
COPY src ./src
COPY library ./library
COPY models.toml ./
COPY deploy/jym.toml ./deploy/
# editable install, since jym serves ui/ and library/ from this folder
# build the aircraft files now (the jym user can't write to /jym)
RUN uv sync --locked --no-dev --python /usr/bin/python3 \
    && .venv/bin/python -c "from jym.environments.flight import cessna; cessna.aircraft_path()"
COPY --from=ui /ui/dist ./ui/dist
# deploy/azure.sh mounts a file share at /home/data to keep runs and models
RUN useradd --uid 10001 --create-home jym && mkdir -p /home/data && chown jym /home/data
USER jym
ENV PORT=8080 JYM_RUNS=/home/data/runs JYM_MODELS=/home/data/models \
    LLAMA_SERVER=/opt/llama/llama-server LD_LIBRARY_PATH=/opt/llama
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/health', timeout=4)"
ENTRYPOINT ["/jym/.venv/bin/jym", "host", "--settings", "/jym/deploy/jym.toml"]
