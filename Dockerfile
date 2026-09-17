# =============================================================================
# Piston Engine Digital Twin — Multi-stage Production Dockerfile
# Offline-capable: all dependencies baked into the image.
# =============================================================================

# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy dependency specification first for layer caching
COPY pyproject.toml ./
RUN uv sync --no-dev

# Copy source code and application files
COPY src/ src/
COPY config/ config/
COPY models/ models/
COPY alembic/ alembic/
COPY alembic.ini ./

# --- Stage 2: Runtime ---
FROM python:3.11-slim AS runtime

WORKDIR /app

# Create non-root user and group
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Copy app environment from builder
COPY --from=builder /app /app

# Set up runtime directories and ownership
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appuser /app

USER appuser

# Environment defaults
ENV APP_ENV=production \
    LOG_LEVEL=INFO \
    CONFIG_PATH=config/default.yaml \
    DATABASE_URL=sqlite+aiosqlite:///./data/digital_twin.db \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    PYTHONPATH=/app

EXPOSE 8000

# Health check against Module 20 API system health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/api/v1/system/health'); r.raise_for_status()"

# Entrypoint running ASGI server
CMD ["/app/.venv/bin/uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
