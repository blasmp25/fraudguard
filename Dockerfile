# ---------- Stage 1: install dependencies with uv ----------
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependencies first: this layer is cached until uv.lock changes
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-default-groups --no-install-project

# Then the project itself
COPY src ./src
RUN uv sync --frozen --no-default-groups

# ---------- Stage 2: minimal runtime image ----------
FROM python:3.11-slim AS runtime

# LightGBM needs the OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Never run as root inside the container
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=builder /app/.venv ./.venv
COPY src ./src
COPY configs ./configs
COPY models/champion ./models/champion
RUN mkdir -p data && chown app:app data

ENV PATH="/app/.venv/bin:$PATH" \
    FRAUDGUARD_CONFIG=/app/configs/docker.yaml \
    PYTHONUNBUFFERED=1

USER app
EXPOSE 8000
CMD ["uvicorn", "fraudguard.api.main:build_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
