# AutoCognitix Monitoring Guide

Comprehensive monitoring and observability setup for the AutoCognitix platform.

## Table of Contents

1. [Overview](#overview)
2. [Structured Logging](#structured-logging)
3. [Metrics Collection](#metrics-collection)
4. [Health Endpoints](#health-endpoints)
5. [Alerting Rules](#alerting-rules)
6. [Grafana Dashboards](#grafana-dashboards)
7. [Troubleshooting](#troubleshooting)

---

## Overview

AutoCognitix implements a comprehensive monitoring stack:

- **Structured Logging**: JSON-formatted logs with request correlation
- **Prometheus Metrics**: Request latency, error rates, database performance
- **Health Probes**: Kubernetes-compatible liveness and readiness checks
- **Sentry Integration**: Error tracking and performance monitoring

### Architecture

```
Application
    |
    +-- Logging Middleware (Request ID, User ID)
    |       |
    |       +-- JSON Logs --> Log Aggregator (ELK/Loki)
    |
    +-- Metrics Middleware (Prometheus)
    |       |
    |       +-- /metrics --> Prometheus --> Grafana
    |
    +-- Health Endpoints
    |       |
    |       +-- /health/live --> Container Orchestrator
    |       +-- /health/ready --> Load Balancer
    |
    +-- Sentry SDK --> Sentry.io
```

---

## Structured Logging

### Log Format

All logs are output in JSON format in production:

```json
{
  "timestamp": "2024-02-05T10:30:45.123456Z",
  "level": "INFO",
  "logger": "request",
  "service": "AutoCognitix",
  "environment": "production",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user_123",
  "correlation_id": "corr_abc123",
  "module": "main",
  "function": "health_check",
  "line": 150,
  "message": "Request completed",
  "event": "request_complete",
  "method": "GET",
  "path": "/health",
  "status_code": 200,
  "duration_ms": 12.5
}
```

### Log Levels

| Level    | Usage                                    | Examples                           |
|----------|------------------------------------------|------------------------------------|
| DEBUG    | Detailed debugging information           | Database query details, cache hits |
| INFO     | General operational information          | Request start/complete, startup    |
| WARNING  | Potential issues, degraded performance   | Slow queries, retry attempts       |
| ERROR    | Actual errors requiring attention        | Failed requests, database errors   |
| CRITICAL | System failures requiring immediate fix  | Service unavailable, data loss     |

### Request Correlation

Every request is assigned a unique Request ID that propagates through all logs:

```python
# Request ID is automatically injected via middleware
logger.info("Processing diagnosis", extra={"dtc_code": "P0300"})
# Output includes request_id from context
```

Headers supported:
- `X-Request-ID`: Unique request identifier (generated if not provided)
- `X-Correlation-ID`: Cross-service correlation ID

### Performance Logging

Use the `PerformanceLogger` for tracking operation timing:

```python
from app.core.logging import PerformanceLogger

# As context manager
with PerformanceLogger("database_query", table="dtc_codes"):
    result = await db.execute(query)

# As decorator
@PerformanceLogger.track("embedding_generation")
async def generate_embedding(text: str):
    ...
```

### Specialized Logging Functions

```python
from app.core.logging import (
    log_database_operation,
    log_external_api_call,
)

# Database operations
log_database_operation(
    operation="select",
    table="dtc_codes",
    duration_ms=15.2,
    rows_affected=100,
)

# External API calls
log_external_api_call(
    service="nhtsa",
    endpoint="/vehicles/decode-vin",
    method="GET",
    status_code=200,
    duration_ms=350.5,
)
```

---

## Metrics Collection

### Available Metrics

#### HTTP Request Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `autocognitix_http_requests_total` | Counter | method, endpoint, status_code | Total HTTP request count |
| `autocognitix_http_request_duration_seconds` | Histogram | method, endpoint | Request latency distribution |
| `autocognitix_http_requests_in_progress` | Gauge | method, endpoint | Currently processing requests |
| `autocognitix_http_request_size_bytes` | Summary | method, endpoint | Request body size |
| `autocognitix_http_response_size_bytes` | Summary | method, endpoint | Response body size |

#### Database Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `autocognitix_db_queries_total` | Counter | database, operation, table | Total database queries |
| `autocognitix_db_query_duration_seconds` | Histogram | database, operation | Query latency |
| `autocognitix_db_query_errors_total` | Counter | database, operation, error_type | Query errors |
| `autocognitix_db_connections` | Gauge | database, state | Connection pool status |
| `autocognitix_db_rows_affected_total` | Counter | database, operation | Rows affected |

#### Embedding & Vector Search Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `autocognitix_embedding_generations_total` | Counter | model, status | Embedding generations |
| `autocognitix_embedding_generation_duration_seconds` | Histogram | model | Generation latency |
| `autocognitix_vector_searches_total` | Counter | collection, status | Vector search operations |
| `autocognitix_vector_search_duration_seconds` | Histogram | collection | Search latency |

#### Business Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `autocognitix_diagnosis_requests_total` | Counter | status, language | Diagnosis requests |
| `autocognitix_diagnosis_duration_seconds` | Histogram | - | Diagnosis latency |
| `autocognitix_dtc_lookups_total` | Counter | found | DTC code lookups |
| `autocognitix_vehicle_decodes_total` | Counter | status | VIN decode requests |

#### LLM Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `autocognitix_llm_requests_total` | Counter | provider, model, status | LLM API requests |
| `autocognitix_llm_duration_seconds` | Histogram | provider, model | LLM request latency |
| `autocognitix_llm_tokens_total` | Counter | provider, model, type | Token usage |

#### System Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `autocognitix_system_cpu_percent` | Gauge | - | System CPU usage |
| `autocognitix_system_memory_percent` | Gauge | - | System memory usage |
| `autocognitix_process_cpu_percent` | Gauge | - | Process CPU usage |
| `autocognitix_process_memory_bytes` | Gauge | type | Process memory (rss, vms) |

#### Error Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `autocognitix_errors_total` | Counter | error_type, endpoint | Total errors |
| `autocognitix_exceptions_total` | Counter | exception_type, endpoint | Unhandled exceptions |

### Using Metrics in Code

```python
from app.core.metrics import (
    track_database_query,
    track_embedding_generation,
    track_vector_search,
    track_diagnosis_request,
    track_external_api_call,
    track_llm_request,
)

# Database query tracking
with track_database_query("postgres", "select", "dtc_codes"):
    result = await db.execute(query)

# Embedding generation tracking
with track_embedding_generation("hubert", batch_size=10):
    embeddings = await model.encode(texts)

# Vector search tracking
with track_vector_search("dtc_embeddings") as ctx:
    results = await qdrant.search(...)
    ctx["results_count"] = len(results)

# LLM request tracking
with track_llm_request("anthropic", "claude-3") as ctx:
    response = await llm.generate(...)
    ctx["input_tokens"] = response.usage.input_tokens
    ctx["output_tokens"] = response.usage.output_tokens
```

---

## Health Endpoints

### Endpoint Summary

| Endpoint | Purpose | Response Time | Used By |
|----------|---------|---------------|---------|
| `/health` | Basic alive check | < 10ms | Quick checks |
| `/health/live` | Liveness probe | < 10ms | Kubernetes |
| `/health/ready` | Readiness probe | < 500ms | Load balancer |
| `/health/detailed` | Full status | < 5s | Dashboards |
| `/health/db` | Database stats | < 5s | Debugging |

### Liveness Probe (`/health/live`)

Returns 200 if the application process is running.

**Usage in Kubernetes:**
```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 15
  timeoutSeconds: 5
  failureThreshold: 3
```

**Response:**
```json
{
  "status": "alive",
  "checked_at": "2024-02-05T10:30:45.123456Z"
}
```

### Readiness Probe (`/health/ready`)

Returns 200 if the application can accept traffic (database connected).

**Usage in Kubernetes:**
```yaml
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

**Response (healthy):**
```json
{
  "status": "ready",
  "checks": {
    "postgres": true
  },
  "checked_at": "2024-02-05T10:30:45.123456Z"
}
```

**Response (unhealthy - 503):**
```json
{
  "status": "not_ready",
  "checks": {
    "postgres": false
  },
  "message": "Critical services unavailable"
}
```

### Detailed Health (`/health/detailed`)

Returns comprehensive health status for all services.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "environment": "production",
  "uptime_seconds": 3600.5,
  "services": {
    "postgres": {
      "name": "PostgreSQL",
      "status": "healthy",
      "latency_ms": 15.2,
      "details": {
        "table_counts": {
          "dtc_codes": 63,
          "vehicle_makes": 50,
          "users": 10
        },
        "pool_status": {
          "pool_size": 5,
          "max_overflow": 10
        }
      }
    },
    "neo4j": {
      "name": "Neo4j",
      "status": "healthy",
      "latency_ms": 45.3,
      "details": {
        "node_counts": {
          "DTCCode": 63,
          "Symptom": 150,
          "Component": 80
        },
        "relationship_count": 500
      }
    },
    "qdrant": {
      "name": "Qdrant",
      "status": "healthy",
      "latency_ms": 25.1,
      "details": {
        "collections_count": 2,
        "collections": {
          "dtc_embeddings": {
            "vectors_count": 63,
            "status": "green"
          }
        }
      }
    },
    "redis": {
      "name": "Redis",
      "status": "healthy",
      "latency_ms": 5.2,
      "details": {
        "version": "7.0.0",
        "connected_clients": 5,
        "used_memory_human": "10M"
      }
    }
  },
  "checked_at": "2024-02-05T10:30:45.123456Z"
}
```

---

## Alerting Rules

### Prometheus Alert Rules

Create a file `prometheus-rules.yaml`:

```yaml
groups:
  - name: autocognitix-alerts
    rules:
      # High Error Rate
      - alert: HighErrorRate
        expr: |
          sum(rate(autocognitix_http_requests_total{status_code=~"5.."}[5m])) /
          sum(rate(autocognitix_http_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"

      # Slow Response Time
      - alert: SlowResponseTime
        expr: |
          histogram_quantile(0.95,
            sum(rate(autocognitix_http_request_duration_seconds_bucket[5m])) by (le)
          ) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow response time detected"
          description: "P95 response time is {{ $value | humanizeDuration }}"

      # Database Connection Failures
      - alert: DatabaseConnectionFailure
        expr: |
          increase(autocognitix_db_query_errors_total{error_type="ConnectionError"}[5m]) > 5
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Database connection failures"
          description: "{{ $value }} connection failures in the last 5 minutes"

      # High Memory Usage
      - alert: HighMemoryUsage
        expr: autocognitix_system_memory_percent > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}%"

      # Critical Memory Usage
      - alert: CriticalMemoryUsage
        expr: autocognitix_system_memory_percent > 95
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Critical memory usage"
          description: "Memory usage is {{ $value }}%"

      # High CPU Usage
      - alert: HighCPUUsage
        expr: autocognitix_system_cpu_percent > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage"
          description: "CPU usage is {{ $value }}%"

      # Slow Database Queries
      - alert: SlowDatabaseQueries
        expr: |
          histogram_quantile(0.95,
            sum(rate(autocognitix_db_query_duration_seconds_bucket[5m])) by (le, database)
          ) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow database queries"
          description: "P95 query time for {{ $labels.database }} is {{ $value | humanizeDuration }}"

      # Slow Embedding Generation
      - alert: SlowEmbeddingGeneration
        expr: |
          histogram_quantile(0.95,
            sum(rate(autocognitix_embedding_generation_duration_seconds_bucket[5m])) by (le)
          ) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow embedding generation"
          description: "P95 embedding generation time is {{ $value | humanizeDuration }}"

      # LLM Request Failures
      - alert: LLMRequestFailures
        expr: |
          sum(rate(autocognitix_llm_requests_total{status="error"}[5m])) /
          sum(rate(autocognitix_llm_requests_total[5m])) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High LLM request failure rate"
          description: "LLM failure rate is {{ $value | humanizePercentage }}"

      # Health Check Failures
      - alert: ServiceUnhealthy
        expr: |
          probe_success{job="autocognitix-health"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Service health check failing"
          description: "Health check has been failing for 2 minutes"

      # No Requests (Service Down)
      - alert: NoRequests
        expr: |
          sum(rate(autocognitix_http_requests_total[5m])) == 0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "No incoming requests"
          description: "No requests received in the last 10 minutes"
```

### Alert Severity Levels

| Severity | Response Time | Examples |
|----------|---------------|----------|
| critical | Immediate (< 5 min) | Error rate > 5%, Database down, Memory > 95% |
| warning | Within 1 hour | Slow queries, High memory, LLM failures |
| info | Next business day | Unusual patterns, Capacity planning |

### Recommended Alert Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Error Rate | > 1% | > 5% |
| P95 Response Time | > 1s | > 2s |
| Memory Usage | > 80% | > 95% |
| CPU Usage | > 80% | > 95% |
| Database Query Time (P95) | > 500ms | > 1s |
| LLM Failure Rate | > 5% | > 10% |

---

## Grafana Dashboards

### Dashboard JSON Template

Import this dashboard into Grafana:

```json
{
  "title": "AutoCognitix Overview",
  "panels": [
    {
      "title": "Request Rate",
      "type": "graph",
      "targets": [
        {
          "expr": "sum(rate(autocognitix_http_requests_total[5m])) by (status_code)",
          "legendFormat": "{{status_code}}"
        }
      ]
    },
    {
      "title": "Response Time (P95)",
      "type": "gauge",
      "targets": [
        {
          "expr": "histogram_quantile(0.95, sum(rate(autocognitix_http_request_duration_seconds_bucket[5m])) by (le))"
        }
      ]
    },
    {
      "title": "Error Rate",
      "type": "stat",
      "targets": [
        {
          "expr": "sum(rate(autocognitix_http_requests_total{status_code=~\"5..\"}[5m])) / sum(rate(autocognitix_http_requests_total[5m])) * 100"
        }
      ]
    },
    {
      "title": "Database Query Time",
      "type": "graph",
      "targets": [
        {
          "expr": "histogram_quantile(0.95, sum(rate(autocognitix_db_query_duration_seconds_bucket[5m])) by (le, database))",
          "legendFormat": "{{database}}"
        }
      ]
    },
    {
      "title": "Memory Usage",
      "type": "gauge",
      "targets": [
        {
          "expr": "autocognitix_system_memory_percent"
        }
      ]
    },
    {
      "title": "Active Requests",
      "type": "stat",
      "targets": [
        {
          "expr": "sum(autocognitix_http_requests_in_progress)"
        }
      ]
    }
  ]
}
```

### Key Dashboard Panels

1. **Request Overview**
   - Request rate by status code
   - Error rate percentage
   - Active requests gauge

2. **Performance**
   - Response time percentiles (P50, P95, P99)
   - Database query latency
   - Embedding generation time

3. **Resource Usage**
   - CPU usage (system and process)
   - Memory usage (system and process)
   - Database connection pool

4. **Business Metrics**
   - Diagnosis requests per minute
   - DTC lookups per minute
   - LLM token usage

---

## Troubleshooting

### Common Issues

#### High Error Rate

1. Check recent error logs:
   ```bash
   kubectl logs -l app=autocognitix --tail=100 | jq 'select(.level == "ERROR")'
   ```

2. Check database connectivity:
   ```bash
   curl http://localhost:8000/health/detailed | jq '.services'
   ```

3. Review metrics:
   ```bash
   curl http://localhost:8000/metrics | grep error
   ```

#### Slow Response Times

1. Check P95 latency by endpoint:
   ```promql
   histogram_quantile(0.95,
     sum(rate(autocognitix_http_request_duration_seconds_bucket[5m])) by (le, endpoint)
   )
   ```

2. Identify slow database queries:
   ```promql
   topk(5,
     histogram_quantile(0.95,
       sum(rate(autocognitix_db_query_duration_seconds_bucket[5m])) by (le, table)
     )
   )
   ```

3. Check external API latency:
   ```promql
   histogram_quantile(0.95,
     sum(rate(autocognitix_external_api_duration_seconds_bucket[5m])) by (le, service)
   )
   ```

#### Memory Issues

1. Check process memory:
   ```promql
   autocognitix_process_memory_bytes{type="rss"}
   ```

2. Monitor memory growth:
   ```promql
   rate(autocognitix_process_memory_bytes{type="rss"}[1h])
   ```

3. Check for memory leaks in specific operations

#### Database Connection Issues

1. Check connection pool status:
   ```bash
   curl http://localhost:8000/health/db | jq '.postgres.pool_status'
   ```

2. Monitor connection errors:
   ```promql
   increase(autocognitix_db_query_errors_total{error_type="ConnectionError"}[5m])
   ```

### Log Analysis

Filter logs by request ID:
```bash
kubectl logs -l app=autocognitix | jq 'select(.request_id == "abc123")'
```

Find slow requests:
```bash
kubectl logs -l app=autocognitix | jq 'select(.duration_ms > 1000)'
```

Find errors for specific endpoint:
```bash
kubectl logs -l app=autocognitix | jq 'select(.level == "ERROR" and .path == "/api/v1/diagnosis/analyze")'
```

---

## Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Minimum log level | INFO |
| `LOG_FORMAT` | Log format (json/text) | json |
| `SENTRY_DSN` | Sentry DSN for error tracking | - |
| `ENVIRONMENT` | Environment name | development |

### Sentry Configuration

Enable Sentry for error tracking:

```bash
SENTRY_DSN=https://xxx@sentry.io/123
ENVIRONMENT=production
```

Sentry will automatically capture:
- Unhandled exceptions
- ERROR level logs and above
- Performance traces (10% sample rate in production)
