# Deployment Guide

This guide covers deploying AutoCognitix to production environments, including Railway, Docker, and manual deployment options.

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Railway Deployment](#railway-deployment)
- [Docker Production Setup](#docker-production-setup)
- [Manual Deployment](#manual-deployment)
- [Environment Variables](#environment-variables)
- [Database Setup](#database-setup)
- [SSL/TLS Configuration](#ssltls-configuration)
- [Monitoring Setup](#monitoring-setup)
- [Scaling](#scaling)
- [Backup and Recovery](#backup-and-recovery)

---

## Architecture Overview

```
                                    ┌─────────────────────────────────────────┐
                                    │           Railway Project               │
                                    └─────────────────────────────────────────┘
                                                      │
                    ┌─────────────────────────────────┼─────────────────────────────────┐
                    │                                 │                                 │
                    ▼                                 ▼                                 ▼
          ┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
          │    Frontend     │              │     Backend     │              │    Databases    │
          │   (React/Vite)  │◄────────────►│    (FastAPI)    │◄────────────►│                 │
          │   Nixpacks      │              │    Dockerfile   │              │  PostgreSQL     │
          └─────────────────┘              └─────────────────┘              │  Redis          │
                                                   │                        └─────────────────┘
                                                   │
                                    ┌──────────────┼──────────────┐
                                    │              │              │
                                    ▼              ▼              ▼
                              ┌──────────┐  ┌──────────┐  ┌──────────┐
                              │ Neo4j    │  │ Qdrant   │  │ LLM API  │
                              │ Aura     │  │ Cloud    │  │ Anthropic│
                              │ (cloud)  │  │ (cloud)  │  │ /OpenAI  │
                              └──────────┘  └──────────┘  └──────────┘
                                    External Services
```

---

## Railway Deployment

Railway provides the simplest deployment path with automatic builds and scaling.

### Prerequisites

1. Railway account (https://railway.app)
2. Railway CLI installed: `npm install -g @railway/cli`
3. External service accounts:
   - Neo4j Aura (https://cloud.neo4j.com) - Free tier available
   - Qdrant Cloud (https://cloud.qdrant.io) - Free tier available
   - Anthropic or OpenAI API key

### Step 1: Initialize Railway Project

```bash
# Login to Railway
railway login

# Initialize project
railway init
```

### Step 2: Add Database Services

In Railway Dashboard:
1. Click **"New"** > **"Database"** > **"PostgreSQL"**
2. Click **"New"** > **"Database"** > **"Redis"**

### Step 3: Configure External Services

#### Neo4j Aura Setup
1. Go to https://cloud.neo4j.com
2. Create a free instance
3. Save the connection URI and password
4. Note the URI format: `neo4j+s://xxx.databases.neo4j.io`

#### Qdrant Cloud Setup
1. Go to https://cloud.qdrant.io
2. Create a free cluster
3. Save the URL and API key
4. Note the URL format: `https://xxx.cloud.qdrant.io:6333`

### Step 4: Configure Environment Variables

In Railway Dashboard > Service > Variables:

```bash
# Auto-configured by Railway (don't set manually)
# DATABASE_URL
# REDIS_URL
# PORT

# Required - External services
NEO4J_URI=neo4j+s://xxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
QDRANT_URL=https://xxx.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key

# Required - LLM
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_anthropic_key

# Required - Security
JWT_SECRET_KEY=generate_with_openssl_rand_hex_32
SECRET_KEY=generate_with_openssl_rand_hex_32

# Required - Application
ENVIRONMENT=production
DEBUG=false

# Optional - Monitoring
SENTRY_DSN=your_sentry_dsn
```

### Step 5: Deploy Backend

The backend uses a Dockerfile. Create/verify `backend/railway.toml`:

```toml
[build]
builder = "dockerfile"
dockerfilePath = "Dockerfile"

[deploy]
healthcheckPath = "/api/v1/health"
healthcheckTimeout = 300
numReplicas = 1
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3
```

Deploy:
```bash
cd backend
railway up
```

### Step 6: Deploy Frontend

The frontend uses Nixpacks. Create/verify `frontend/railway.toml`:

```toml
[build]
builder = "nixpacks"

[deploy]
healthcheckPath = "/"
numReplicas = 1
```

Set frontend environment variables:
```bash
VITE_API_URL=https://your-backend.railway.app
```

Deploy:
```bash
cd frontend
railway up
```

### Step 7: Run Migrations

```bash
railway run --service backend alembic upgrade head
```

### Step 8: Seed Initial Data

```bash
railway run --service backend python -m scripts.seed_database
```

---

## Docker Production Setup

For self-hosted deployments using Docker.

### Production docker-compose.yml

```yaml
version: '3.8'

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    environment:
      - DATABASE_URL=postgresql+asyncpg://autocognitix:${POSTGRES_PASSWORD}@postgres:5432/autocognitix
      - NEO4J_URI=${NEO4J_URI}
      - NEO4J_USER=${NEO4J_USER}
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
      - QDRANT_URL=${QDRANT_URL}
      - QDRANT_API_KEY=${QDRANT_API_KEY}
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=${SECRET_KEY}
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - ENVIRONMENT=production
      - DEBUG=false
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
      target: production
    environment:
      - VITE_API_URL=https://api.yourdomain.com
    ports:
      - "3000:3000"
    depends_on:
      - backend
    restart: unless-stopped

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: autocognitix
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: autocognitix
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U autocognitix"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 512M

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - ./nginx/certbot:/var/www/certbot:ro
    depends_on:
      - backend
      - frontend
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### Production Dockerfile (Backend)

```dockerfile
# backend/Dockerfile
FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Production command with Gunicorn
CMD ["gunicorn", "app.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", \
     "--bind", "0.0.0.0:8000", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
```

### Nginx Configuration

```nginx
# nginx/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server backend:8000;
    }

    upstream frontend {
        server frontend:3000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    server {
        listen 80;
        server_name yourdomain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name yourdomain.com;

        ssl_certificate /etc/nginx/ssl/fullchain.pem;
        ssl_certificate_key /etc/nginx/ssl/privkey.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
        ssl_prefer_server_ciphers off;

        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;

        # API requests
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Frontend
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        # Health check
        location /health {
            proxy_pass http://backend/api/v1/health;
        }
    }
}
```

---

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://user:pass@host:5432/db` |
| `NEO4J_URI` | Neo4j connection URI | `neo4j+s://xxx.databases.neo4j.io` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `your_password` |
| `QDRANT_URL` | Qdrant URL | `https://xxx.cloud.qdrant.io:6333` |
| `QDRANT_API_KEY` | Qdrant API key | `your_key` |
| `REDIS_URL` | Redis connection string | `redis://host:6379/0` |
| `SECRET_KEY` | Application secret | `openssl rand -hex 32` |
| `JWT_SECRET_KEY` | JWT signing key | `openssl rand -hex 32` |
| `ANTHROPIC_API_KEY` | Anthropic API key (if using Claude) | `sk-ant-...` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Environment name | `production` |
| `DEBUG` | Debug mode | `false` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `RATE_LIMIT_PER_MINUTE` | API rate limit | `60` |
| `SENTRY_DSN` | Sentry error tracking | - |

---

## Database Setup

### PostgreSQL Initialization

```bash
# Run migrations
alembic upgrade head

# Seed data
python -m scripts.seed_database
```

### Neo4j Initialization

The seed script creates:
- DTC nodes with relationships
- Symptom nodes
- Component nodes
- Repair nodes

### Qdrant Initialization

The application automatically:
1. Creates the `dtc_codes` collection on startup
2. Indexes DTC embeddings

To manually recreate:
```python
from app.db.qdrant_client import qdrant_client
import asyncio

asyncio.run(qdrant_client.initialize())
```

---

## SSL/TLS Configuration

### Using Let's Encrypt (Certbot)

```bash
# Install certbot
apt-get install certbot python3-certbot-nginx

# Obtain certificate
certbot --nginx -d yourdomain.com -d api.yourdomain.com

# Auto-renewal
certbot renew --dry-run
```

### Manual Certificate Setup

Place certificates in:
- `/etc/nginx/ssl/fullchain.pem`
- `/etc/nginx/ssl/privkey.pem`

---

## Monitoring Setup

### Prometheus Metrics

The backend exposes metrics at `/api/v1/metrics`:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'autocognitix'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/api/v1/metrics'
```

### Available Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Total HTTP requests |
| `http_request_duration_seconds` | Histogram | Request latency |
| `diagnosis_requests_total` | Counter | Diagnosis API calls |
| `diagnosis_confidence_score` | Histogram | Confidence score distribution |
| `database_query_duration_seconds` | Histogram | Database query latency |

### Sentry Error Tracking

Configure in environment:
```bash
SENTRY_DSN=https://xxx@sentry.io/xxx
```

### Log Aggregation

JSON structured logs are emitted. Configure log forwarding to your preferred system:

```python
# Example log output
{
  "timestamp": "2024-02-03T10:30:00.000Z",
  "level": "INFO",
  "message": "Diagnosis completed",
  "diagnosis_id": "550e8400-e29b-41d4-a716-446655440000",
  "confidence_score": 0.85,
  "duration_ms": 1250
}
```

---

## Scaling

### Horizontal Scaling

#### Backend
```yaml
# docker-compose scale
docker-compose up -d --scale backend=3

# Railway
railway scale backend --replicas 3
```

#### Load Balancing
Use Nginx upstream configuration or Railway's built-in load balancing.

### Vertical Scaling

Adjust resource limits:
```yaml
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2'
```

### Database Scaling

- **PostgreSQL**: Use connection pooling (PgBouncer)
- **Redis**: Configure Redis Cluster for high availability
- **Qdrant**: Scale to multiple nodes for large vector databases
- **Neo4j**: Use Neo4j Aura Professional for higher throughput

---

## Backup and Recovery

### PostgreSQL Backup

```bash
# Backup
pg_dump -h localhost -U autocognitix autocognitix > backup.sql

# Restore
psql -h localhost -U autocognitix autocognitix < backup.sql

# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h $DB_HOST -U $DB_USER $DB_NAME | gzip > backup_$DATE.sql.gz
# Upload to S3 or other storage
aws s3 cp backup_$DATE.sql.gz s3://your-bucket/backups/
```

### Neo4j Backup

For Neo4j Aura, backups are automatic. For self-hosted:
```bash
neo4j-admin dump --database=neo4j --to=backup.dump
```

### Redis Backup

```bash
# Trigger RDB snapshot
redis-cli BGSAVE

# Copy dump.rdb to backup location
cp /var/lib/redis/dump.rdb /backups/redis_$(date +%Y%m%d).rdb
```

### Qdrant Backup

```bash
# Using Qdrant API
curl -X POST "http://localhost:6333/collections/dtc_codes/snapshots"

# Download snapshot
curl -O "http://localhost:6333/collections/dtc_codes/snapshots/{snapshot_name}"
```

---

## Troubleshooting

### Common Issues

#### Database Connection Failed
```bash
# Check PostgreSQL
docker-compose logs postgres
docker-compose exec postgres pg_isready

# Check connectivity
docker-compose exec backend python -c "
from app.db.postgres.session import async_session_factory
import asyncio
async def test():
    async with async_session_factory() as session:
        result = await session.execute('SELECT 1')
        print('Connected:', result.scalar())
asyncio.run(test())
"
```

#### Memory Issues
```bash
# Check container memory usage
docker stats

# Reduce embedding batch size
EMBEDDING_BATCH_SIZE=4

# Reduce worker count
CMD ["gunicorn", "...", "--workers", "2"]
```

#### Slow Responses
```bash
# Check database indexes
docker-compose exec postgres psql -U autocognitix -c "\di"

# Check Qdrant performance
curl "http://localhost:6333/collections/dtc_codes"

# Enable query logging
LOG_LEVEL=DEBUG
```

### Health Checks

```bash
# Backend health
curl http://localhost:8000/api/v1/health

# Detailed health (includes database status)
curl http://localhost:8000/api/v1/health/detailed

# Frontend health
curl http://localhost:3000
```

---

## Security Checklist

- [ ] Strong passwords for all services
- [ ] JWT secrets rotated
- [ ] SSL/TLS enabled
- [ ] Rate limiting configured
- [ ] CORS properly restricted
- [ ] Sentry or error tracking enabled
- [ ] Regular security updates
- [ ] Backup verification tested
- [ ] Firewall rules configured
- [ ] No debug mode in production
