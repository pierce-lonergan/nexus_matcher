# docker/Dockerfile | Env: PROD
# Multi-stage Dockerfile for NexusMatcher
# SENSITIVE: No | REQUIRED_BY: docker-compose, CI/CD

# =============================================================================
# BUILD STAGE
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# LICENSE is required at BUILD time, not just at runtime: pyproject.toml declares
# `license = {file = "LICENSE"}`, and hatchling resolves that while generating
# metadata. Without it `python -m build` dies with
#   OSError: License file does not exist: LICENSE
# which is exactly how this image was failing.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --no-cache-dir build && \
    python -m build --wheel

# =============================================================================
# RUNTIME STAGE
# =============================================================================
FROM python:3.11-slim AS runtime

# Create non-root user
RUN groupadd --gid 1000 nexus && \
    useradd --uid 1000 --gid nexus --shell /bin/bash --create-home nexus

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the built wheel and install it WITH the extras this image actually runs.
# The previous version installed the bare wheel, which pulls only the five core
# dependencies - so `uvicorn` (the CMD below) and the qdrant/redis clients that
# docker-compose.yml configures were never present and the container could not
# start. Extras must be requested explicitly; they are not implied by the wheel.
COPY --from=builder /app/dist /dist

RUN WHEEL="$(ls /dist/*.whl)" && \
    pip install --no-cache-dir "${WHEEL}[full]" && \
    rm -rf /dist

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

# Health check.
#
# /health/ready, NOT /health. `/health` answers HTTP 200 with `status: "degraded"` when
# the dictionary failed to load, so `curl -f` passes and the container reports healthy
# while every POST /api/v1/match answers 503 - which is how a bad deploy gets through a
# rollout gate. /health/ready returns 503 for that state and names the red component in
# the body.
#
# A container with no NEXUS_API_DICTIONARY is therefore unhealthy, deliberately: it
# serves the matching routes and cannot answer them. A health-and-introspection
# deployment sets NEXUS_API_MATCHING_OPTIONAL=true to say so. That variable is NOT set
# here - defaulting the image to the exemption would put every deployment back where it
# started.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health/ready || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "nexus_matcher.presentation.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
