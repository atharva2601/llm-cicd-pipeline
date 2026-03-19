# ─────────────────────────────────────────────────────────────────
# Dockerfile — LLM CI/CD Evaluation Pipeline
#
# Provides a reproducible container environment so tests run
# identically on every developer's machine and in CI.
#
# Build:
#   docker build -t llm-eval .
#
# Run smoke tests:
#   docker run --env-file .env llm-eval make smoke
#
# Run all tests:
#   docker run --env-file .env llm-eval make all-tests
#
# Interactive shell:
#   docker run -it --env-file .env llm-eval /bin/bash
#
# Notes:
#   - The .env file is NEVER baked into the image (it's bind-mounted
#     at runtime via --env-file).
#   - The /app/reports directory is declared as a VOLUME so CI
#     runners can extract evaluation reports after the run.
# ─────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadata
LABEL org.opencontainers.image.title="LLM Evaluation Pipeline"
LABEL org.opencontainers.image.description="DeepEval + Gemini Flash CI/CD evaluation"

# ── System packages ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    make \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (cached layer) ───────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────
COPY . .

# Remove .env if accidentally included (defence in depth)
RUN rm -f .env

# ── Reports volume (extracted by CI after run) ────────────────────
VOLUME ["/app/reports"]

# ── Default entrypoint: run smoke tests ──────────────────────────
# Override at runtime: docker run ... llm-eval make full
CMD ["make", "smoke"]
