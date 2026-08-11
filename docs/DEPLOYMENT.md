# NexusMatcher Deployment Guide

Docker, Kubernetes and cloud deployment patterns for the service.

> **Read this before following the rest of the guide.** Several things this document
> assumes are not implemented today:
>
> - **A deployment that sets no dictionary answers 503 on every match.** The FastAPI app
>   serves `POST /api/v1/match`, `/api/v1/match/batch` and `/api/v1/feedback` alongside
>   `/`, `/health`, `/health/live`, `/health/ready`, `/health/startup`, `/docs`, `/redoc`
>   and `/openapi.json` — but the matching routes need `NEXUS_API_DICTIONARY` and
>   `NEXUS_API_GOVERNANCE`, which are §2 below and are **not** the `NEXUS_*` settings
>   classes. An earlier revision of this box said the service did not match schemas at
>   all; that is retracted. See [API_REFERENCE.md](API_REFERENCE.md).
> - **YAML config files shown below do not configure matching.** `NexusMatcher.from_config()`
>   ignores its `config_path`, and the `NEXUS_*` settings classes are read only by the
>   logging setup. Treat the config examples here as a target design.
> - **`/metrics` is not routed.** A `PrometheusMetrics` backend class exists in
>   `nexus_matcher.shared.metrics`, but no endpoint exposes it, so the Prometheus scrape
>   config and Grafana dashboard below have nothing to read yet.
> - **`/health/ready` does not probe dependencies.** It reports `api`, `config` and
>   `matcher`, and only the last of those is a real check — the first two are set
>   unconditionally at startup. It tells you the process started and whether a dictionary
>   loaded, not that Qdrant or Redis are reachable. `vector_store` and `cache` used to
>   appear in that map, each set `True` inside a `try:` block whose body was a comment, so
>   neither could ever be `False`; they were removed rather than left as a claim a rollout
>   gate reads. See §5 "Probe targets".
> - **The multi-layer cache is not consulted by the matching pipeline.** Cache tuning
>   parameters below have no effect on matching today.

---

## Table of Contents

1. [Deployment Options](#1-deployment-options)
2. [Environment Configuration](#2-environment-configuration)
3. [Local Development](#3-local-development)
4. [Docker Deployment](#4-docker-deployment)
5. [Kubernetes Deployment](#5-kubernetes-deployment)
6. [Cloud Deployments](#6-cloud-deployments)
7. [Performance Tuning](#7-performance-tuning)
8. [Monitoring & Observability](#8-monitoring--observability)
9. [Security Hardening](#9-security-hardening)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Deployment Options

### Deployment Matrix

| Option | Best For | Complexity | Scalability |
|--------|----------|------------|-------------|
| **Local** | Development, testing | Low | Single instance |
| **Docker Compose** | Small teams, PoC | Low | Manual scaling |
| **Kubernetes** | Production, enterprise | High | Auto-scaling |
| **AWS ECS/Fargate** | AWS-native apps | Medium | Auto-scaling |
| **GCP Cloud Run** | GCP-native apps | Medium | Auto-scaling |
| **Azure Container Apps** | Azure-native apps | Medium | Auto-scaling |

### Resource Requirements

| Tier | vCPU | RAM | Storage | Dictionary Size |
|------|------|-----|---------|-----------------|
| **Development** | 2 | 4GB | 10GB | <10K entries |
| **Small** | 4 | 8GB | 50GB | <50K entries |
| **Medium** | 8 | 16GB | 100GB | <200K entries |
| **Large** | 16 | 32GB | 500GB | <1M entries |

---

## 2. Environment Configuration

### Required Environment Variables

```bash
# Core Settings
NEXUS_ENV=production              # production, staging, development
NEXUS_LOG_LEVEL=INFO              # DEBUG, INFO, WARNING, ERROR
NEXUS_SECRET_KEY=<generate-secure-key>

# Embedding Model
NEXUS_EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
NEXUS_EMBEDDING_DEVICE=cpu        # cpu, cuda, mps
NEXUS_EMBEDDING_BATCH_SIZE=32
NEXUS_EMBEDDING_USE_INT8=true     # Recommended for production

# Vector Store
NEXUS_VECTOR_BACKEND=qdrant       # qdrant, memory
NEXUS_VECTOR_QDRANT_HOST=localhost
NEXUS_VECTOR_QDRANT_PORT=6333
NEXUS_VECTOR_QDRANT_COLLECTION=dictionary_embeddings
NEXUS_VECTOR_QDRANT_API_KEY=<optional-api-key>

# Caching
NEXUS_CACHE_L1_ENABLED=true
NEXUS_CACHE_L1_MAX_SIZE=5000
NEXUS_CACHE_L2_ENABLED=true
NEXUS_CACHE_L2_REDIS_URL=redis://localhost:6379/0
NEXUS_CACHE_L3_ENABLED=true
NEXUS_CACHE_L3_MAX_SIZE=10000

# API Settings -- ten of the thirteen variables create_app() actually reads
NEXUS_API_DICTIONARY=/srv/glossary.csv          # REQUIRED for matching (.csv / .xlsx)
NEXUS_API_GOVERNANCE=/srv/protection_classes.json  # REQUIRED for matching
NEXUS_API_MATCHING_CONFIG=/srv/matching.json    # optional MatchingConfig JSON/TOML
NEXUS_API_FEEDBACK_PATH=/var/lib/nexus/feedback.jsonl  # POST /api/v1/feedback 503s without it
NEXUS_API_DEADLINE_SECONDS=25.0  # answer 504 rather than let a caller hang
NEXUS_API_MAX_WORKERS=4          # threads running matches
NEXUS_API_MAX_QUEUED=32          # work admitted beyond the workers; over this, shed with 503
NEXUS_API_MAX_FIELDS=100         # cap for POST /api/v1/match
NEXUS_API_MAX_BATCH_FIELDS=250   # cap for POST /api/v1/match/batch
NEXUS_API_MAX_BODY_BYTES=        # raw request-body cap; UNSET (the default) derives it

# Three more, documented where they are used: NEXUS_API_CORS_ORIGINS and
# NEXUS_API_CORS_ALLOW_CREDENTIALS in §9, NEXUS_API_MATCHING_OPTIONAL in §5.
```

Thirteen is the whole set, measured by handing `create_app()` an environment mapping that
records every key it looks up and driving a full startup through it — not by grepping,
because three of the thirteen are read inside the lifespan handler and one is read only
when another is non-empty.

**Both `NEXUS_API_DICTIONARY` and `NEXUS_API_GOVERNANCE` are required for matching**, and
the second one is the trap: with a dictionary and no vocabulary, a server would answer 200
and return `"governance": null` on every field — indistinguishable, to the caller, from a
glossary that carries no protection classes at all. The bootstrap therefore refuses to bring
the matcher up when the glossary plainly has a protection-code column and nothing was
configured to read it: the process starts, `/health/ready` returns 503, and every match
returns 503 naming the column. It cannot see the case where your codes live under a column
name it does not recognise.

#### `NEXUS_API_MAX_BODY_BYTES`, and the one way it stops the process starting

Leave it unset unless you have a reason. Unset, the cap is **derived** from
`NEXUS_API_MAX_BATCH_FIELDS` and the per-field `max_length`s the request schema publishes:
250 fields × (9,856 characters × 4 bytes for UTF-8 + 64 bytes of JSON framing) + 1,024 =
**9,873,024 bytes, 9.42 MiB**. Deriving it is what keeps the two caps in step — raise the
field cap and the byte cap rises with it, so an operator cannot end up with a 413 that
names a limit the schema says is fine.

Setting it is therefore an escape hatch, and it is validated **against the field cap rather
than on its own**, in `MatchServiceLimits.__post_init__`. A value below the floor above is
refused at construction:

```text
$ NEXUS_API_MAX_BODY_BYTES=1048576 uvicorn nexus_matcher.presentation.api.app:app
Traceback (most recent call last):
  ...
ValueError: NEXUS_API_MAX_BODY_BYTES=1048576 is below the 9873024 bytes a
max_batch_fields=250 request may legitimately carry, so this server would answer 413 to
bodies every other limit it publishes accepts. Unset it to derive the cap, or lower
NEXUS_API_MAX_BATCH_FIELDS to match.
```

The process does not start — and `nexus-matcher api` refuses with the same message, because
both entry points build the same limits object. That is deliberate and it is the cheaper
failure: the alternative is a server that comes up healthy, passes every probe, and answers
413 to batches its own `/openapi.json` declares valid — which the caller reads as a bug in
their client, because their client was generated from that document. The refusal names both
ways out, and the second one really does work: `NEXUS_API_MAX_BODY_BYTES=1048576` alongside
`NEXUS_API_MAX_BATCH_FIELDS=20` starts, because 20 fields put the floor at 790,784 bytes.

One 413 the cap cannot avoid: a client that escapes every character as `\uXXXX` spends six
bytes per character on the wire, so a body inside its declared field bounds can still cross
a derived cap. Raise this variable for that — the validation only refuses values *below* the
floor, never above it.

This block was previously `NEXUS_API_HOST`, `NEXUS_API_PORT`, `NEXUS_API_WORKERS`,
`NEXUS_API_TIMEOUT` and `NEXUS_API_CORS_ORIGINS`. **No code read any of those five**
(verified by grep over `src/`), and an operator who configured a deployment from that list
got a server that 503s every match. Host, port and worker count are flags on
`nexus-matcher api`, not environment variables. `NEXUS_API_CORS_ORIGINS` is the exception:
it is now read, in comma-separated form, and it is documented in §9 with the rest of the
security surface. The rate-limit block that used to sit here was deleted rather than
implemented — see §9 for why.

### Configuration File

```yaml
# config/production.yaml
environment: production

embedding:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  device: cpu
  batch_size: 32
  use_int8: true
  cache_embeddings: true

vector_store:
  backend: qdrant
  qdrant:
    host: ${NEXUS_VECTOR_QDRANT_HOST:-localhost}
    port: ${NEXUS_VECTOR_QDRANT_PORT:-6333}
    collection_name: dictionary_embeddings
    prefer_grpc: true

caching:
  l1:
    enabled: true
    max_size: 5000
    ttl: 3600
  l2:
    enabled: true
    redis_url: ${NEXUS_CACHE_L2_REDIS_URL:-redis://localhost:6379/0}
    ttl: 86400
  l3:
    enabled: true
    max_size: 10000

retrieval:
  top_k: 100
  rerank_top_k: 10
  use_hybrid: true
  use_maxsim: true
  maxsim_precompute: true

scoring:
  thresholds:
    auto_approve: 0.75
    review: 0.50
  weights:
    semantic: 0.60
    lexical: 0.15
    type: 0.15
    pattern: 0.10

api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  timeout: 30
  cors:
    enabled: true
    origins: ["*"]

logging:
  level: INFO
  format: json
  handlers:
    - type: stdout
    - type: file
      path: /var/log/nexus_matcher/app.log
      rotation: daily
      retention: 30
```

---

## 3. Local Development

### Quick Start

```bash
# Clone repository
git clone https://github.com/your-org/nexus_matcher.git
cd nexus_matcher

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[full,dev]"

# Start development server
nexus-matcher api --reload
```

### Development with Docker Compose

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: docker/Dockerfile.dev
    volumes:
      - .:/app
      - ~/.cache/huggingface:/root/.cache/huggingface
    ports:
      - "8000:8000"
    environment:
      - NEXUS_ENV=development
      - NEXUS_VECTOR_BACKEND=memory
    command: uvicorn nexus_matcher.presentation.api.app:app --host 0.0.0.0 --port 8000 --reload

  # Optional: Local Qdrant for testing
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_dev_data:/qdrant/storage

volumes:
  qdrant_dev_data:
```

```bash
# Start development stack
docker compose -f docker-compose.dev.yml up
```

---

## 4. Docker Deployment

### Dockerfile

```dockerfile
# docker/Dockerfile
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy application
COPY src/ src/
COPY config/ config/
COPY pyproject.toml .

# Install application
RUN pip install --no-cache-dir .

# Create non-root user
RUN useradd -m -u 1000 nexus
USER nexus

# Health check -- /health/ready, not /health. See "Probe targets" in §5.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health/ready || exit 1

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "nexus_matcher.presentation.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose Production

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - NEXUS_ENV=production
      - NEXUS_VECTOR_BACKEND=qdrant
      - NEXUS_VECTOR_QDRANT_HOST=qdrant
      - NEXUS_CACHE_L2_ENABLED=true
      - NEXUS_CACHE_L2_REDIS_URL=redis://redis:6379/0
    depends_on:
      qdrant:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G

  qdrant:
    image: qdrant/qdrant:v1.7.0
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # Optional: Nginx reverse proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - app
    restart: unless-stopped

volumes:
  qdrant_data:
  redis_data:

networks:
  default:
    driver: bridge
```

### Nginx Configuration

```nginx
# nginx/nginx.conf
upstream nexus_matcher {
    server app:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name _;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name _;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;

    location / {
        proxy_pass http://nexus_matcher;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    location /health {
        proxy_pass http://nexus_matcher/health;
        access_log off;
    }
}
```

### Deployment Commands

```bash
# Build and start
docker compose build
docker compose up -d

# Check status
docker compose ps
docker compose logs -f app

# Scale application
docker compose up -d --scale app=3

# Update deployment
docker compose pull
docker compose up -d --no-deps app

# Graceful shutdown
docker compose down
```

---

## 5. Kubernetes Deployment

### Namespace and ConfigMap

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: nexus-matcher
  labels:
    app.kubernetes.io/name: nexus-matcher
---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nexus-matcher-config
  namespace: nexus-matcher
data:
  NEXUS_ENV: "production"
  NEXUS_LOG_LEVEL: "INFO"
  NEXUS_EMBEDDING_MODEL_NAME: "sentence-transformers/all-MiniLM-L6-v2"
  NEXUS_EMBEDDING_DEVICE: "cpu"
  NEXUS_EMBEDDING_USE_INT8: "true"
  NEXUS_VECTOR_BACKEND: "qdrant"
  NEXUS_VECTOR_QDRANT_HOST: "qdrant-service"
  NEXUS_VECTOR_QDRANT_PORT: "6333"
  NEXUS_CACHE_L1_ENABLED: "true"
  NEXUS_CACHE_L2_ENABLED: "true"
```

### Secret

```yaml
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: nexus-matcher-secrets
  namespace: nexus-matcher
type: Opaque
stringData:
  NEXUS_SECRET_KEY: "your-secret-key-here"
  NEXUS_CACHE_L2_REDIS_URL: "redis://:password@redis-service:6379/0"
```

### Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nexus-matcher
  namespace: nexus-matcher
  labels:
    app: nexus-matcher
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nexus-matcher
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: nexus-matcher
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: nexus-matcher
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: nexus-matcher
        image: your-registry/nexus-matcher:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP
        envFrom:
        - configMapRef:
            name: nexus-matcher-config
        - secretRef:
            name: nexus-matcher-secrets
        resources:
          requests:
            cpu: "1000m"
            memory: "2Gi"
          limits:
            cpu: "4000m"
            memory: "8Gi"
        # /health/ready and /health/live, NOT /health. See "Probe targets" below.
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 3
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 10"]
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - nexus-matcher
              topologyKey: kubernetes.io/hostname
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: ScheduleAnyway
        labelSelector:
          matchLabels:
            app: nexus-matcher
```

### Probe targets

**Do not point a probe at `/health`.** Every probe this repository shipped did — the two
manifests above, `docker/Dockerfile`'s `HEALTHCHECK` and `docker/docker-compose.yml` —
and `/health` answers **HTTP 200 with `status: "degraded"`** when the dictionary failed to
load. `curl -f` passes on it, `httpGet` passes on it, and the pod goes into service
answering 503 to every `POST /api/v1/match`. That is a rollout gate that cannot fail.

| Probe | Path | Answers 503 when |
|---|---|---|
| `readinessProbe` | `/health/ready` | any component that gates readiness is red — today, `matcher` |
| `livenessProbe` | `/health/live` | never; it says the process is running, which is what liveness means |
| `startupProbe` | `/health/startup` | startup has not finished |
| — | `/health` | never — it is a human-readable summary, not a gate |

The readiness 503 carries the whole component map in `error.details.components`, so
`kubectl describe` plus one `curl` tells you which component is red rather than only that
one is:

```json
{"error": {"code": "NEXUS-8000",
           "message": "The service is not ready: matcher not healthy.",
           "details": {"status_code": 503, "components": {"api": true, "config": true, "matcher": false}}}}
```

**A deployment that sets no `NEXUS_API_DICTIONARY` is now NOT READY.** This is a
behaviour change and it is deliberate: readiness previously aggregated only components
that had registered, and a missing dictionary registered nothing, so `all(())` reported
ready — the state a Helm value that fails to resolve produces. If this deployment really
serves only health and introspection, say so:

```yaml
- name: NEXUS_API_MATCHING_OPTIONAL
  value: "true"
```

That exemption is opt-*out*, not opt-in, on purpose: a knob whose default is the unsafe
value protects only the operators who remembered to set it, who are not the ones with a
misconfigured rollout. Do not bake it into a base image or a shared ConfigMap.

### Service and Ingress

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: nexus-matcher-service
  namespace: nexus-matcher
spec:
  type: ClusterIP
  selector:
    app: nexus-matcher
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: nexus-matcher-ingress
  namespace: nexus-matcher
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - nexus-matcher.your-domain.com
    secretName: nexus-matcher-tls
  rules:
  - host: nexus-matcher.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: nexus-matcher-service
            port:
              number: 80
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nexus-matcher-hpa
  namespace: nexus-matcher
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nexus-matcher
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
```

### Helm Chart Values

```yaml
# helm/values.yaml
replicaCount: 3

image:
  repository: your-registry/nexus-matcher
  tag: latest
  pullPolicy: Always

resources:
  requests:
    cpu: 1000m
    memory: 2Gi
  limits:
    cpu: 4000m
    memory: 8Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
  - host: nexus-matcher.your-domain.com
    paths:
    - path: /
      pathType: Prefix
  tls:
  - secretName: nexus-matcher-tls
    hosts:
    - nexus-matcher.your-domain.com

qdrant:
  enabled: true
  replicaCount: 3
  persistence:
    size: 100Gi

redis:
  enabled: true
  architecture: replication
  auth:
    enabled: true
```

---

## 6. Cloud Deployments

### AWS ECS/Fargate

```json
// task-definition.json
{
  "family": "nexus-matcher",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/nexusMatcherTaskRole",
  "containerDefinitions": [
    {
      "name": "nexus-matcher",
      "image": "ACCOUNT.dkr.ecr.REGION.amazonaws.com/nexus-matcher:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "NEXUS_ENV", "value": "production"},
        {"name": "NEXUS_VECTOR_BACKEND", "value": "qdrant"}
      ],
      "secrets": [
        {
          "name": "NEXUS_SECRET_KEY",
          "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT:secret:nexus-matcher-SECRET"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/nexus-matcher",
          "awslogs-region": "REGION",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health/ready || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

### GCP Cloud Run

```yaml
# cloudrun.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: nexus-matcher
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 80
      timeoutSeconds: 30
      containers:
      - image: gcr.io/PROJECT/nexus-matcher:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            cpu: "4"
            memory: "8Gi"
        env:
        - name: NEXUS_ENV
          value: "production"
        - name: NEXUS_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: nexus-matcher-secrets
              key: secret-key
```

---

## 7. Performance Tuning

### Application Tuning

```yaml
# config/performance.yaml
embedding:
  batch_size: 64          # Larger batches for throughput
  use_int8: true          # measured 1.26x-2.93x depending on batch size, on a
                          # machine without VNNI; 1.27x at batch 32. See
                          # docs/BENCHMARK_REGISTRY.md#suite-002-real
  max_seq_length: 128     # Shorter for speed

retrieval:
  top_k: 50               # Fewer candidates
  rerank_top_k: 5         # Fewer final results
  use_maxsim: true
  maxsim_precompute: true # Critical!

caching:
  l1:
    max_size: 10000       # More cache
    ttl: 7200             # Longer TTL
  l3:
    max_size: 50000       # Large semantic cache

api:
  workers: 8              # More workers
  worker_connections: 1000
```

### Qdrant Tuning

```yaml
# qdrant/config.yaml
storage:
  optimizers:
    default_segment_number: 4
    indexing_threshold_kb: 20000
    
  hnsw:
    m: 16                 # More connections
    ef_construct: 200     # Better index
    
service:
  grpc_port: 6334
  enable_tls: false
  
performance:
  search_workers: 8
```

### Redis Tuning

```bash
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
tcp-keepalive 300
tcp-backlog 511
```

---

## 8. Monitoring & Observability

### Prometheus Metrics — NOT YET EXPOSED

No route serves `/metrics`. The metric names below are those defined by
`nexus_matcher.shared.metrics`; wiring an endpoint that exposes them is outstanding
work. The scrape configuration and dashboard in this section will collect nothing until
that exists.

```python
# PLANNED: application would expose metrics at /metrics
# Key metrics:
# - nexus_requests_total
# - nexus_request_duration_seconds
# - nexus_cache_hits_total
# - nexus_cache_misses_total
# - nexus_embedding_latency_seconds
# - nexus_rerank_latency_seconds
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "NexusMatcher",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {"expr": "rate(nexus_requests_total[5m])"}
        ]
      },
      {
        "title": "Latency P95",
        "targets": [
          {"expr": "histogram_quantile(0.95, rate(nexus_request_duration_seconds_bucket[5m]))"}
        ]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {"expr": "rate(nexus_cache_hits_total[5m]) / (rate(nexus_cache_hits_total[5m]) + rate(nexus_cache_misses_total[5m]))"}
        ]
      }
    ]
  }
}
```

### Alerting Rules

```yaml
# prometheus/alerts.yaml
groups:
- name: nexus-matcher
  rules:
  - alert: HighErrorRate
    expr: rate(nexus_requests_total{status="error"}[5m]) > 0.05
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(nexus_request_duration_seconds_bucket[5m])) > 0.5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High latency detected (P95 > 500ms)"
      
  - alert: LowCacheHitRate
    expr: rate(nexus_cache_hits_total[5m]) / (rate(nexus_cache_hits_total[5m]) + rate(nexus_cache_misses_total[5m])) < 0.3
    for: 15m
    labels:
      severity: warning
    annotations:
      summary: "Cache hit rate below 30%"
```

---

## 9. Security Hardening

### API Security

**This service authenticates nobody.** There is no API key, no OAuth and no rate limiting
in the application, and there is no environment variable that turns any of them on. Put it
behind your own gateway and authenticate there; the network policy below is the other half
of that, not a substitute for it.

This section used to instruct operators to set four variables. **No code read any of
them**, so hardening by the book produced a server whose CORS policy was still
`allow_origins=["*"]` with credentials enabled, and whose request rate was still
unlimited. Both claims are retracted:

| Was documented here | Status |
|---|---|
| `NEXUS_API_CORS_ORIGINS` in JSON-list syntax | **now read** — comma-separated, see below |
| `NEXUS_RATE_LIMIT_ENABLED` | **deleted.** Not implemented, and not planned |
| `NEXUS_RATE_LIMIT_REQUESTS` | **deleted.** Not implemented, and not planned |
| `NEXUS_RATE_LIMIT_WINDOW` | **deleted.** Not implemented, and not planned |

Rate limiting was deleted rather than implemented because the load protection that matters
already exists and is real: `NEXUS_API_MAX_WORKERS` + `NEXUS_API_MAX_QUEUED` bound the
in-flight work and shed the excess with 503 immediately, and `NEXUS_API_DEADLINE_SECONDS`
answers 504 rather than letting a caller hang. A second overlapping mechanism is surface
for no gain. Per-client quotas belong on the gateway, which is the only component that
knows who the clients are.

#### Cross-origin access (browsers)

Closed by default — with no `NEXUS_API_CORS_ORIGINS`, no CORS middleware is mounted at
all and a browser on another origin can read nothing. Server-to-server callers, including
the Java pipeline this endpoint was built for, never send an `Origin` and are unaffected.

```bash
# Comma-separated origins. NOT the JSON-list syntax this document used to show; the
# application parses one spelling, so the other silently meant nothing.
NEXUS_API_CORS_ORIGINS=https://console.your-domain.example,https://ops.your-domain.example

# Cookies cross-origin. Default false, and REFUSED AT STARTUP together with `*`:
# Starlette reflects the requesting origin rather than sending `*` when credentials are
# on, so "allow every origin" silently becomes "allow this origin, with cookies".
NEXUS_API_CORS_ALLOW_CREDENTIALS=false
```

Allowed methods are `GET`, `POST` and `OPTIONS`. There is no setting for that list: it is
the set of verbs the service serves.

Until this was fixed, a preflight from any origin came back with that origin reflected and
`access-control-allow-credentials: true`, so any page anywhere could read a match response
and could `POST /api/v1/feedback` — which fsyncs a reviewer verdict into the audit trail
this library treats as evidence.

### Network Security

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: nexus-matcher-network-policy
  namespace: nexus-matcher
spec:
  podSelector:
    matchLabels:
      app: nexus-matcher
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: qdrant
    ports:
    - protocol: TCP
      port: 6333
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

### Secret Management

```bash
# Use external secret managers
# AWS Secrets Manager
# GCP Secret Manager
# HashiCorp Vault
# Azure Key Vault
```

---

## 10. Troubleshooting

### Common Issues

**Issue**: High memory usage
```
Cause: Embedding model or cache size
Solution:
- Enable INT8 quantization (75% model size reduction)
- Reduce L1/L3 cache sizes
- Increase pod memory limits
```

**Issue**: Slow response times
```
Cause: Cold cache or unoptimized queries
Solution:
- Warm up cache on startup
- Enable MaxSim pre-computation
- Check Qdrant index health
```

**Issue**: Connection refused to Qdrant/Redis
```
Cause: Network or service issue
Solution:
- Check service endpoints
- Verify network policies
- Check pod logs for connection errors
```

### Debug Commands

```bash
# Check pod logs
kubectl logs -f deployment/nexus-matcher -n nexus-matcher

# Check pod resources
kubectl top pods -n nexus-matcher

# Check Qdrant health
curl http://qdrant-service:6333/health

# Check Redis
redis-cli -h redis-service ping

# Test API
curl http://nexus-matcher-service/health
```

---

*Deployment Guide Version 1.0.0*
*Last Updated: December 2025*
