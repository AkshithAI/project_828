# ─────────────────────────────────────────────────────────────────────────────
# Project 828 — Multi-stage Docker build
# Stage 1: slim builder with CPU-only PyTorch (tests + dev)
# Stage 2: production image (no test deps)
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /app

# System deps for building wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ make && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (CPU-only torch for smaller image)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pytest

# Copy source and tests
COPY . .

# Run the test suite as a build-time gate — image only builds if tests pass
RUN make test

# ── Stage 2: Production ────────────────────────────────────────────────────
FROM python:3.13-slim AS production

WORKDIR /app

# Install runtime-only Python deps (CPU torch)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy only source code (no tests, no cache, no checkpoints)
COPY src/ src/
COPY Makefile pytest.ini ./
COPY tests/ tests/

# Default entrypoint
CMD ["python3", "-c", "print('Project 828 container ready. Use: python3 -m pytest tests/ for tests.')"]
