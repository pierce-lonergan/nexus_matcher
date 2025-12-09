# docker/Dockerfile | Env: PROD
# Multi-stage Dockerfile for NexusMatcher
# SENSITIVE: No | REQUIRED_BY: docker-compose, CI/CD

# =============================================================================
# BUILD STAGE
# =============================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir build && \
    python -m build --wheel && \
    pip wheel --no-cache-dir --wheel-dir=/app/wheels -e ".[full]"

# =============================================================================
# RUNTIME STAGE
# =============================================================================
FROM python:3.11-slim as runtime

# Create non-root user
RUN groupadd --gid 1000 nexus && \
    useradd --uid 1000 --gid nexus --shell /bin/bash --create-home nexus

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels and install
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/dist /dist

RUN pip install --no-cache-dir /dist/*.whl && \
    rm -rf /wheels /dist

# Create directories
RUN mkdir -p /app/data /app/models /app/logs && \
    chown -R nexus:nexus /app

# Switch to non-root user
USER nexus

# Environment
ENV NEXUS_ENV=production \
    NEXUS_DATA_DIR=/app/data \
    NEXUS_MODELS_DIR=/app/models \
    NEXUS_LOGS_DIR=/app/logs \
    NEXUS_SERVER_HOST=0.0.0.0 \
    NEXUS_SERVER_PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "nexus_matcher.presentation.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
