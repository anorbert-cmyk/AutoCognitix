# Security Fixes Coordination Document

**Lead Coordinator:** Claude (Security Lead)
**Date Created:** 2026-02-08
**Status:** In Planning

---

## Executive Summary

This document coordinates three critical security fixes for the AutoCognitix platform:
1. Remove hardcoded credentials from all scripts (27 files identified)
2. Implement JWT token validation with Redis-backed blacklist
3. Add CSRF protection and security headers middleware

**Total Affected Files:** 30+ files
**Estimated Complexity:** High (multiple DB systems involved)
**Risk Level:** Critical security vulnerabilities (credential exposure, token hijacking)

---

## Issue Tracking Table

| ID | Issue | Description | Assignee | Status | Priority | Risk |
|----|-------|-------------|----------|--------|----------|------|
| SEC-001 | Hardcoded Credentials in Scripts | 27 Python scripts contain hardcoded Neo4j, PostgreSQL, and Qdrant credentials | Teammate 1 | Pending | CRITICAL | High |
| SEC-002 | JWT Blacklist In-Memory Only | Token blacklist uses Python set instead of Redis, lost on restart | Teammate 2 | Pending | CRITICAL | High |
| SEC-003 | Missing CSRF Protection | No CSRF token validation middleware | Teammate 3 | Pending | HIGH | Medium |
| SEC-004 | Missing Security Headers | No X-Frame-Options, X-Content-Type-Options, HSTS, CSP headers | Teammate 3 | Pending | HIGH | Medium |
| SEC-005 | JWT Validation Gap | No rate limiting on token refresh endpoint | Teammate 2 | Pending | MEDIUM | Medium |
| SEC-006 | Missing Logout Flow | Tokens not reliably invalidated on logout | Teammate 2 | Pending | MEDIUM | Low |

---

## Dependency Map

```
SEC-001 (Credentials)
  ├─ Must complete BEFORE any deployment
  ├─ Blocks: SEC-002 (Redis can't be tested with hardcoded creds)
  └─ No external dependencies

SEC-002 (JWT/Redis Blacklist)
  ├─ Depends on: Redis configuration in backend/app/core/config.py
  ├─ Must complete BEFORE: SEC-005
  ├─ Files affected:
  │   ├── backend/app/core/security.py (primary)
  │   ├── backend/app/db/redis_cache.py (integration)
  │   ├── backend/app/api/v1/endpoints/auth.py (usage)
  │   └── Tests: tests/test_security.py (NEW)
  └─ Depends on: SEC-001 (for clean Redis credentials)

SEC-003 & SEC-004 (CSRF + Headers)
  ├─ Independent of other fixes
  ├─ Files affected:
  │   ├── backend/app/core/security.py (CSRF utilities)
  │   ├── backend/app/main.py (middleware registration)
  │   ├── backend/app/api/v1/endpoints/auth.py (CSRF token generation)
  │   └── Tests: tests/test_csrf_headers.py (NEW)
  └─ Must complete BEFORE: Frontend changes

SEC-005 & SEC-006 (Token Management)
  ├─ Depends on: SEC-002 (Redis blacklist)
  ├─ Files affected:
  │   ├── backend/app/core/security.py (blacklist checks)
  │   ├── backend/app/api/v1/endpoints/auth.py (logout endpoint)
  │   └── Tests: tests/test_auth_flow.py (enhancement)
  └─ Related to: JWT token expiration handling

Integration Points:
  SEC-001 ──────────────────┐
                            ├──→ SEC-002 ────→ SEC-005 & SEC-006
  SEC-003 & SEC-004 ────────┘
```

---

## File Inventory

### Teammate 1: Credentials Removal

**Primary Files to Modify:** 27 scripts
**Location:** `/backend/scripts/*.py`

Scripts requiring hardcoded credential removal:

```python
# High Priority (API/Auth related):
- seed_neo4j_aura.py          # Lines 15-18: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD hardcoded
- seed_database.py             # Loads from settings but may have inline examples
- import_back4app_vehicles.py  # Back4App API credentials
- import_nhtsa_recalls.py      # NHTSA API configuration
- import_nhtsa_complaints.py   # NHTSA API configuration
- import_dtcdb.py              # Database credentials
- import_obdb.py               # OBDb database credentials
- import_obdb_github.py        # GitHub API token (if any)

# Medium Priority (Data pipeline):
- export_data.py               # Database connection strings
- export_openapi.py            # Configuration
- backup_data.py               # Database credentials
- index_qdrant.py              # Qdrant cloud credentials
- index_qdrant_robust.py       # Qdrant API key
- index_all_to_qdrant.py       # Qdrant configuration
- index_qdrant_hubert.py       # Qdrant + HuBERT credentials
- index_qdrant_full.py         # Qdrant configuration
- init_qdrant.py               # Qdrant setup
- data_sync.py                 # All DB credentials
- expand_neo4j_graph.py        # Neo4j credentials
- create_symptom_database.py   # Database configuration
- import_data.py               # Multi-DB configuration

# Lower Priority (Utilities):
- health_check.py              # May have inline connection strings
- download_all_obdb.py         # GitHub credentials
- setup_neo4j_indexes.py       # Neo4j connection
- continue_translations.py     # Database connection
- fix_translations.py          # Database connection
```

**Pattern to Fix:**
```python
# WRONG (Hardcoded):
uri = "neo4j+s://xxx.databases.neo4j.io"
password = "mypassword123"

# CORRECT (Environment variables):
from app.core.config import settings
uri = settings.NEO4J_URI
password = settings.NEO4J_PASSWORD
```

**Testing Requirements:**
- [ ] All scripts run with environment variables only
- [ ] No credentials in git history (use `git log -p` to verify)
- [ ] Scripts fail gracefully when env vars missing
- [ ] No credentials in error messages/logs

---

### Teammate 2: JWT Validation + Redis Blacklist

**Primary File:** `backend/app/core/security.py`
**Secondary Files:**
- `backend/app/db/redis_cache.py`
- `backend/app/api/v1/endpoints/auth.py`
- `backend/app/main.py`

#### Current Implementation Analysis

**Status quo (security.py lines 24-26):**
```python
# In-memory token blacklist (in production, use Redis)
# This stores JTI (JWT ID) of invalidated tokens
_token_blacklist: Set[str] = set()
```

**Problem:** Token blacklist is lost on application restart.

#### Required Changes

1. **Replace in-memory set with Redis (security.py)**
   - Remove lines 24-26
   - Import Redis client from `app.db.redis_cache`
   - Implement async blacklist operations:
     ```python
     async def blacklist_token(token: str) -> bool:
         # Get token JTI, store in Redis with TTL = token expiration time

     async def is_token_blacklisted(jti: str) -> bool:
         # Check Redis for JTI
     ```
   - TTL calculation: Use `exp` claim from JWT to auto-expire Redis keys

2. **Update decode_token() function (security.py lines 152-176)**
   - Make function async: `async def decode_token(token: str) -> dict | None`
   - Check Redis blacklist: `if jti and await is_token_blacklisted(jti)`
   - Update all callers in auth.py endpoints

3. **Update logout endpoint (auth.py)**
   - Current endpoint should call `await blacklist_token(token)`
   - Verify token is actually blacklisted

4. **Add token refresh rate limiting (auth.py)**
   - Rate limit: max 5 token refreshes per hour per user
   - Use Redis counter: `refresh_count:{user_id}:{hour}`

**Testing Requirements:**
- [ ] Blacklist token before expiration → decode returns None
- [ ] Blacklist persists after app restart
- [ ] Token expiration auto-removes from Redis
- [ ] Refresh rate limiting works correctly
- [ ] Multiple refresh attempts within hour rejected
- [ ] Performance: Redis blacklist check < 10ms

**Test Files to Create:**
- `tests/unit/core/test_security_redis_blacklist.py`
- `tests/integration/test_jwt_token_lifecycle.py`

---

### Teammate 3: CSRF Protection + Security Headers

**Files to Create:**
- `backend/app/core/csrf.py` (NEW)

**Files to Modify:**
- `backend/app/main.py` (middleware registration)
- `backend/app/api/v1/endpoints/auth.py` (CSRF token generation)
- `backend/app/core/security.py` (CSRF utilities)

#### CSRF Implementation

**Requirement:** Double-submit cookie CSRF pattern

1. **Create CSRF middleware (NEW: backend/app/core/csrf.py)**
   - Generate CSRF tokens: 32 bytes, URL-safe random
   - Store in Redis: `csrf_token:{token_hash}` → TTL 1 hour
   - Validate: Check header `X-CSRF-Token` against cookie
   - Exempt endpoints: GET, OPTIONS, HEAD
   - Exempt paths: `/health`, `/metrics`, `/api/v1/docs`, `/api/v1/redoc`

2. **Register middleware in main.py (after line 162)**
   ```python
   from app.core.csrf import CSRFMiddleware
   application.add_middleware(
       CSRFMiddleware,
       exclude_paths=["/health", "/metrics", "/api/v1/docs", "/api/v1/redoc"],
       safe_methods=["GET", "OPTIONS", "HEAD"],
   )
   ```

3. **Add CSRF token endpoint in auth.py (NEW)**
   ```python
   @router.get("/csrf-token", tags=["Authentication"])
   async def get_csrf_token() -> dict:
       """Get CSRF token for form submissions"""
       return {"csrf_token": generate_csrf_token()}
   ```

4. **Update login form flow in frontend**
   - Call `/api/v1/auth/csrf-token` before login
   - Send token in `X-CSRF-Token` header on POST

#### Security Headers Implementation

**Requirement:** Add 5 critical headers

**Modify main.py (NEW middleware - before CORS):**
```python
from app.core.security import SecurityHeadersMiddleware
application.add_middleware(SecurityHeadersMiddleware)
```

**Headers to add (security.py NEW function):**
```python
X-Frame-Options: DENY                           # Prevent clickjacking
X-Content-Type-Options: nosniff                 # Prevent MIME sniffing
X-XSS-Protection: 1; mode=block                 # Legacy XSS protection
Strict-Transport-Security: max-age=31536000     # Require HTTPS for 1 year
Content-Security-Policy: default-src 'self'    # Restrict resource loading
```

**Testing Requirements:**
- [ ] CSRF token generated on request
- [ ] CSRF token validated for POST/PUT/DELETE
- [ ] GET requests bypass CSRF check
- [ ] Expired CSRF tokens rejected (> 1 hour)
- [ ] Security headers present in all responses
- [ ] CSP blocks inline scripts
- [ ] HSTS header forces HTTPS redirects
- [ ] Clickjacking protection blocks iframe embedding

**Test Files to Create:**
- `tests/unit/core/test_csrf_protection.py`
- `tests/unit/core/test_security_headers.py`
- `tests/integration/test_csrf_flow.py`

---

## Testing Strategy

### Unit Tests

#### Teammate 1 (Credentials)
```bash
# No unit tests possible - this is config validation
# Manual verification: grep -r "password\|api_key\|token" scripts/ | grep -v "^[[:space:]]*#" | grep -v os.environ
```

#### Teammate 2 (JWT/Redis)
```bash
pytest tests/unit/core/test_security_redis_blacklist.py -v
pytest tests/unit/core/test_jwt_validation.py -v
```

#### Teammate 3 (CSRF/Headers)
```bash
pytest tests/unit/core/test_csrf_protection.py -v
pytest tests/unit/core/test_security_headers.py -v
```

### Integration Tests

```bash
# Complete auth flow with new security
pytest tests/integration/test_jwt_token_lifecycle.py -v

# CSRF token + form submission flow
pytest tests/integration/test_csrf_flow.py -v

# All three fixes together
pytest tests/integration/test_security_suite.py -v
```

### End-to-End Tests (manual)

1. **SEC-001 Verification:**
   ```bash
   # Run script with missing env var
   unset NEO4J_PASSWORD
   python scripts/seed_neo4j_aura.py  # Should fail with clear error

   # Search for hardcoded credentials
   git diff HEAD~1 -- scripts/ | grep -i "password\|api_key\|token"
   ```

2. **SEC-002 Verification:**
   ```bash
   # Test logout + reuse token
   1. Login → get access token
   2. POST /api/v1/auth/logout
   3. Reuse token → should get 401 Unauthorized
   4. Restart app
   5. Reuse token → should still get 401
   ```

3. **SEC-003 Verification:**
   ```bash
   # Test CSRF flow
   1. GET /api/v1/auth/csrf-token → get token X
   2. POST /api/v1/auth/login with X-CSRF-Token: X → success
   3. POST /api/v1/auth/login without header → 403 Forbidden
   4. POST /api/v1/auth/login with wrong token → 403 Forbidden
   ```

4. **SEC-004 Verification:**
   ```bash
   # Check all response headers
   curl -I http://localhost:8000/api/v1/health
   # Verify: X-Frame-Options, X-Content-Type-Options, HSTS, CSP
   ```

---

## Performance Requirements

| Fix | Metric | Threshold | Current | Target |
|-----|--------|-----------|---------|--------|
| SEC-002 | Redis blacklist check latency | < 10ms | N/A | ✓ |
| SEC-002 | Token decode time with blacklist | < 15ms | N/A | ✓ |
| SEC-003 | CSRF token generation | < 5ms | N/A | ✓ |
| SEC-003 | CSRF validation latency | < 5ms | N/A | ✓ |
| SEC-004 | Security header injection | < 1ms | N/A | ✓ |

**Monitoring:**
- Add request metrics for each security check
- Log slow operations (> threshold)
- Use Prometheus to track security operation latencies

---

## Security Validation Checklist

- [ ] No hardcoded credentials in any script
- [ ] No credentials in git history
- [ ] No credentials in error messages
- [ ] JWT tokens rejected after logout
- [ ] JWT blacklist persists across restarts
- [ ] CSRF tokens generated and validated
- [ ] CSRF tokens expire after 1 hour
- [ ] Security headers present in all responses
- [ ] Content-Security-Policy blocks inline scripts
- [ ] HSTS header forces HTTPS
- [ ] X-Frame-Options prevents clickjacking

---

## Deployment Plan

### Pre-deployment
1. **Teammate 1** completes SEC-001 and creates PR
2. **Lead** reviews for any remaining credentials
3. **Teammate 2** starts SEC-002 once SEC-001 merged
4. **Teammate 3** starts SEC-003/004 in parallel
5. All PCs pass CI/CD checks

### Deployment Steps
```bash
# Step 1: Deploy backend with new security middleware
# (assumes no breaking changes yet)
git merge sec-001-credentials
railway deploy

# Step 2: Deploy with Redis blacklist (requires Redis operational)
git merge sec-002-jwt-redis
alembic upgrade head  # If any schema changes
railway deploy

# Step 3: Deploy with CSRF + headers (can be done anytime)
git merge sec-003-004-csrf-headers
railway deploy

# Step 4: Frontend changes (CSRF token retrieval)
# After step 3 is live
cd frontend && npm run build
railway deploy
```

### Rollback Plan
If deployment fails:
1. Revert the problematic commit
2. Restart application
3. Clear Redis cache: `redis-cli FLUSHDB` (for CSRF tokens only, preserve other data)
4. If JWT blacklist corrupted: `redis-cli DEL "jti:*"`

---

## Monitoring & Alerting

### Metrics to Track

```python
# In Prometheus/metrics endpoint
security_tokens_blacklisted_total        # Counter of blacklisted tokens
security_csrf_tokens_generated_total     # Counter of CSRF tokens issued
security_csrf_validations_failed_total   # Counter of CSRF validation failures
security_redis_blacklist_latency_ms      # Histogram of Redis lookup times
security_password_reset_tokens_issued    # Counter of password reset tokens
```

### Alerts to Configure

| Alert | Condition | Action |
|-------|-----------|--------|
| **HighCSRFFailureRate** | > 5% of requests fail CSRF | Review traffic patterns |
| **RedisBlacklistSlow** | p95 latency > 20ms | Investigate Redis performance |
| **TokenBlacklistGrowth** | > 100k JTIs in Redis | Review logout patterns |
| **CredentialLeak** | Password/token in logs | Immediate incident response |

---

## Documentation Updates Needed

After all fixes are complete, update:
1. `backend/README.md` - Add CSRF flow documentation
2. `docs/API.md` - Document `/auth/csrf-token` endpoint
3. `docs/DEPLOYMENT.md` - Add security headers verification step
4. `docs/SECURITY.md` - Document token lifecycle, CSRF protection

---

## Communication Plan

### Weekly Status Updates
- Teammate 1 (Monday): Credential removal progress
- Teammate 2 (Tuesday): JWT/Redis implementation status
- Teammate 3 (Wednesday): CSRF/Headers implementation status
- Lead (Friday): Integration test results and blockers

### If Critical Issues Found
- Lead convenes emergency sync
- All teammates share findings in `tasks/security_audit_blockers.md`
- Jointly decide rollback vs. fix-forward strategy

---

## Success Criteria

✅ **SEC-001 Complete When:**
- All 27 scripts use environment variables only
- No credentials in git history
- Scripts fail gracefully with missing env vars
- CI/CD pipeline passes

✅ **SEC-002 Complete When:**
- Redis blacklist persists after restart
- Token refresh rate limiting enforced
- All unit + integration tests pass
- Performance metrics within threshold

✅ **SEC-003/004 Complete When:**
- CSRF tokens generated for every request
- CSRF validation works for form submissions
- All 5 security headers present
- No inline scripts execute (CSP enforced)

✅ **Overall Complete When:**
- All 6 issues marked as "Done"
- E2E tests pass in staging
- Security audit passes (no new vulnerabilities)
- Deployment to production successful

---

## References

**Files Involved:** 30+
**Estimated Hours:** 40-60 hours total (16-20 per teammate)
**Risk Assessment:** High (security critical)
**Review Process:** Code review + security review before merge

**Related Documentation:**
- `backend/app/core/config.py` - Settings structure
- `backend/app/core/security.py` - Current implementation
- `backend/app/main.py` - Middleware registration
- `.env.example` - Environment variable template

---

**Document Version:** 1.0
**Last Updated:** 2026-02-08
**Next Review:** After SEC-001 completion
