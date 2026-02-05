# AutoCognitix Integration Guide

This guide covers how to integrate with the AutoCognitix API for vehicle diagnostics.

## Table of Contents

- [Getting Started](#getting-started)
- [Authentication Flow](#authentication-flow)
- [Making Requests](#making-requests)
- [Rate Limiting](#rate-limiting)
- [Error Handling](#error-handling)
- [Common Integration Patterns](#common-integration-patterns)
- [Language Support](#language-support)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

- API access credentials (contact support@autocognitix.hu)
- HTTPS client library for your platform
- Basic understanding of REST APIs and JSON

### Base URLs

| Environment | URL |
|-------------|-----|
| Production | `https://api.autocognitix.hu/api/v1` |
| Staging | `https://staging-api.autocognitix.hu/api/v1` |
| Local Development | `http://localhost:8000/api/v1` |

### Quick Start

1. Register a user account
2. Login to obtain JWT tokens
3. Use access token for API requests
4. Refresh token when access token expires

```bash
# 1. Register
curl -X POST "https://api.autocognitix.hu/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"email":"your@email.com","password":"yourpassword"}'

# 2. Login
curl -X POST "https://api.autocognitix.hu/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=your@email.com&password=yourpassword"

# 3. Use API (example: analyze vehicle)
curl -X POST "https://api.autocognitix.hu/api/v1/diagnosis/analyze" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "vehicle_make": "Volkswagen",
    "vehicle_model": "Golf",
    "vehicle_year": 2018,
    "dtc_codes": ["P0101"],
    "symptoms": "A motor nehezen indul."
  }'
```

---

## Authentication Flow

### Token Lifecycle

```
┌─────────────────┐
│  User Login     │
│  POST /login    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Receive Tokens  │
│ access_token    │◄───────────────────────────┐
│ refresh_token   │                            │
└────────┬────────┘                            │
         │                                      │
         ▼                                      │
┌─────────────────┐                            │
│ Make API Calls  │                            │
│ Bearer Token    │                            │
└────────┬────────┘                            │
         │                                      │
         ▼                                      │
┌─────────────────┐     Yes    ┌──────────────┐
│ Token Expired?  │───────────►│ POST /refresh│
└────────┬────────┘            └──────────────┘
         │ No
         ▼
┌─────────────────┐
│ Continue Using  │
│ API             │
└─────────────────┘
```

### Token Storage Recommendations

| Platform | Recommended Storage |
|----------|---------------------|
| Web Browser | HttpOnly cookies or secure storage |
| Mobile App | Secure enclave / Keychain |
| Backend Service | Environment variables or secrets manager |

### Handling Token Expiration

**Access Token** expires in 30 minutes. When you receive a `401 Unauthorized` response:

1. Try refreshing the token using the refresh endpoint
2. If refresh fails, redirect user to login

**Refresh Token** expires in 7 days. When refresh fails:

1. Clear stored tokens
2. Redirect user to login page

### Code Example: Token Refresh

```python
import requests
from datetime import datetime, timedelta

class AutoCognitixClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None

    def login(self, email: str, password: str):
        response = requests.post(
            f"{self.base_url}/auth/login",
            data={"username": email, "password": password}
        )
        response.raise_for_status()
        tokens = response.json()

        self.access_token = tokens["access_token"]
        self.refresh_token = tokens["refresh_token"]
        # Access token expires in 30 minutes
        self.token_expires_at = datetime.now() + timedelta(minutes=25)

    def _refresh_if_needed(self):
        """Refresh token if it's about to expire."""
        if self.token_expires_at and datetime.now() >= self.token_expires_at:
            response = requests.post(
                f"{self.base_url}/auth/refresh",
                json={"refresh_token": self.refresh_token}
            )
            if response.status_code == 200:
                tokens = response.json()
                self.access_token = tokens["access_token"]
                self.refresh_token = tokens["refresh_token"]
                self.token_expires_at = datetime.now() + timedelta(minutes=25)
            else:
                raise Exception("Session expired, please login again")

    def _get_headers(self):
        self._refresh_if_needed()
        return {"Authorization": f"Bearer {self.access_token}"}

    def analyze(self, vehicle_data: dict) -> dict:
        response = requests.post(
            f"{self.base_url}/diagnosis/analyze",
            headers=self._get_headers(),
            json=vehicle_data
        )
        response.raise_for_status()
        return response.json()
```

---

## Making Requests

### Request Headers

| Header | Required | Description |
|--------|----------|-------------|
| `Authorization` | Yes (for most endpoints) | `Bearer <access_token>` |
| `Content-Type` | Yes (for POST/PUT) | `application/json` |
| `Accept` | Optional | `application/json` |
| `Accept-Language` | Optional | `hu` for Hungarian, `en` for English |

### Request Body Format

All POST/PUT requests should send JSON:

```json
{
  "field_name": "value",
  "nested_object": {
    "key": "value"
  },
  "array_field": ["item1", "item2"]
}
```

### Response Format

All responses are JSON:

```json
{
  "id": "...",
  "data": "...",
  "created_at": "2024-02-03T10:30:00Z"
}
```

### Pagination

List endpoints support pagination:

```
GET /api/v1/diagnosis/history/list?skip=0&limit=10
```

| Parameter | Type | Default | Max |
|-----------|------|---------|-----|
| skip | integer | 0 | - |
| limit | integer | 10 | 100 |

---

## Rate Limiting

### Limits

| Limit | Value |
|-------|-------|
| Per minute | 60 requests |
| Per hour | 1000 requests |

### Response Headers

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1707134400
```

### Handling Rate Limits

When you receive `429 Too Many Requests`:

```python
import time

def make_request_with_retry(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = func()
            return response
        except requests.HTTPError as e:
            if e.response.status_code == 429:
                # Get retry time from header or use exponential backoff
                retry_after = int(e.response.headers.get('Retry-After', 60))
                wait_time = min(retry_after, (2 ** attempt) * 10)
                print(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    raise Exception("Max retries exceeded")
```

### Best Practices for Rate Limits

1. **Cache responses** when possible
2. **Batch requests** instead of making many individual calls
3. **Implement exponential backoff** for retries
4. **Monitor rate limit headers** proactively

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message"
}
```

### Validation Errors (422)

```json
{
  "detail": [
    {
      "loc": ["body", "dtc_codes", 0],
      "msg": "Invalid DTC code format",
      "type": "value_error"
    }
  ]
}
```

### Error Codes and Actions

| Code | Meaning | Action |
|------|---------|--------|
| 400 | Bad Request | Check request body/parameters |
| 401 | Unauthorized | Refresh token or re-login |
| 403 | Forbidden | Check permissions |
| 404 | Not Found | Resource doesn't exist |
| 422 | Validation Error | Fix input data |
| 429 | Rate Limited | Wait and retry |
| 500 | Server Error | Retry later, contact support |
| 502 | Bad Gateway | External service issue (NHTSA) |

### Error Handling Example

```python
import requests

def handle_api_response(response):
    """Handle API response with proper error handling."""
    try:
        response.raise_for_status()
        return response.json()

    except requests.HTTPError as e:
        status = response.status_code

        if status == 400:
            error = response.json()
            raise ValueError(f"Bad request: {error['detail']}")

        elif status == 401:
            raise AuthenticationError("Token expired or invalid")

        elif status == 404:
            raise NotFoundError(f"Resource not found")

        elif status == 422:
            errors = response.json()['detail']
            messages = [f"{e['loc']}: {e['msg']}" for e in errors]
            raise ValidationError(f"Validation failed: {', '.join(messages)}")

        elif status == 429:
            retry_after = response.headers.get('Retry-After', 60)
            raise RateLimitError(f"Rate limited. Retry after {retry_after}s")

        elif status >= 500:
            raise ServerError(f"Server error: {status}")

        else:
            raise APIError(f"Unexpected error: {status}")
```

---

## Common Integration Patterns

### Pattern 1: Basic Vehicle Diagnosis

```python
def diagnose_vehicle(client, make, model, year, dtc_codes, symptoms):
    """
    Perform basic vehicle diagnosis.

    Args:
        client: AutoCognitixClient instance
        make: Vehicle make (e.g., "Volkswagen")
        model: Vehicle model (e.g., "Golf")
        year: Vehicle year (e.g., 2018)
        dtc_codes: List of DTC codes (e.g., ["P0101", "P0171"])
        symptoms: Symptom description in Hungarian

    Returns:
        Diagnosis result with probable causes and repairs
    """
    result = client.analyze({
        "vehicle_make": make,
        "vehicle_model": model,
        "vehicle_year": year,
        "dtc_codes": dtc_codes,
        "symptoms": symptoms
    })

    return {
        "confidence": result["confidence_score"],
        "causes": result["probable_causes"],
        "repairs": result["recommended_repairs"]
    }
```

### Pattern 2: VIN-Based Diagnosis

```python
def diagnose_with_vin(client, vin, dtc_codes, symptoms):
    """
    Diagnose vehicle using VIN for automatic vehicle identification.
    """
    # First decode VIN
    vehicle = client.decode_vin(vin)

    # Then perform diagnosis with vehicle info
    result = client.analyze({
        "vehicle_make": vehicle["make"],
        "vehicle_model": vehicle["model"],
        "vehicle_year": vehicle["year"],
        "vin": vin,
        "dtc_codes": dtc_codes,
        "symptoms": symptoms
    })

    return result
```

### Pattern 3: Quick DTC Lookup

```python
def quick_dtc_info(client, dtc_codes):
    """
    Get quick information about DTC codes without full analysis.
    Useful for previews or basic information display.
    """
    return client.quick_analyze(dtc_codes)
```

### Pattern 4: Recall Check Integration

```python
def check_vehicle_safety(client, make, model, year):
    """
    Check for recalls and complaints for a vehicle.
    Important for safety-critical applications.
    """
    recalls = client.get_recalls(make, model, year)
    complaints = client.get_complaints(make, model, year)

    safety_issues = []

    for recall in recalls:
        safety_issues.append({
            "type": "recall",
            "severity": "high",
            "description": recall["subject"],
            "remedy": recall["remedy"]
        })

    for complaint in complaints:
        if complaint.get("crash") or complaint.get("fire"):
            safety_issues.append({
                "type": "complaint",
                "severity": "critical" if complaint.get("crash") else "high",
                "description": complaint["summary"]
            })

    return safety_issues
```

### Pattern 5: Async Integration (JavaScript)

```javascript
class AutoCognitixService {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
    this.tokens = null;
  }

  async login(email, password) {
    const response = await fetch(`${this.baseUrl}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({ username: email, password }),
    });

    if (!response.ok) {
      throw new Error('Login failed');
    }

    this.tokens = await response.json();
    return this.tokens;
  }

  async analyzeVehicle(data) {
    const response = await fetch(`${this.baseUrl}/diagnosis/analyze`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.tokens.access_token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (response.status === 401) {
      // Try to refresh token
      await this.refreshToken();
      return this.analyzeVehicle(data);
    }

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Analysis failed');
    }

    return response.json();
  }

  async refreshToken() {
    const response = await fetch(`${this.baseUrl}/auth/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: this.tokens.refresh_token }),
    });

    if (!response.ok) {
      throw new Error('Session expired');
    }

    this.tokens = await response.json();
  }
}

// Usage
const service = new AutoCognitixService('https://api.autocognitix.hu/api/v1');
await service.login('user@example.com', 'password');

const diagnosis = await service.analyzeVehicle({
  vehicle_make: 'Volkswagen',
  vehicle_model: 'Golf',
  vehicle_year: 2018,
  dtc_codes: ['P0101'],
  symptoms: 'A motor nehezen indul.',
});
```

---

## Language Support

### Hungarian Language Processing

The API is optimized for Hungarian text in the `symptoms` field:

```json
{
  "symptoms": "A motor nehezen indul hidegben, egyenetlenul jar alapjaraton, es a fogyasztas megnott."
}
```

### Bilingual Responses

DTC descriptions include both English and Hungarian:

```json
{
  "description_en": "Mass Air Flow Circuit Range/Performance",
  "description_hu": "Levegotomeg-mero aramkor tartomany/teljesitmeny hiba"
}
```

### Character Encoding

- Always use UTF-8 encoding
- Hungarian special characters (a, e, i, o, u, o, u, u, o) are fully supported
- Accented characters can be submitted with or without accents

---

## Best Practices

### 1. Implement Proper Error Handling

Always handle all possible error codes and provide meaningful feedback to users.

### 2. Cache When Possible

- Cache DTC category lists (rarely change)
- Cache vehicle makes/models (change infrequently)
- Cache VIN decode results (vehicle info doesn't change)

### 3. Validate Input Client-Side

Validate DTC code format before sending to API:

```python
import re

def validate_dtc_code(code: str) -> bool:
    """Validate DTC code format: P0101, B1234, etc."""
    pattern = r'^[PBCU][0-9]{4}$'
    return bool(re.match(pattern, code.upper()))
```

### 4. Use Appropriate Endpoints

| Need | Endpoint |
|------|----------|
| Quick DTC lookup | `/diagnosis/quick-analyze` |
| Full diagnosis | `/diagnosis/analyze` |
| DTC search | `/dtc/search` |
| DTC details | `/dtc/{code}` |

### 5. Handle Long-Running Requests

The `/diagnosis/analyze` endpoint may take several seconds. Implement appropriate timeouts and loading states:

```python
# Set appropriate timeout
response = requests.post(
    f"{BASE_URL}/diagnosis/analyze",
    json=data,
    timeout=30  # 30 second timeout
)
```

### 6. Log API Interactions

Maintain logs for debugging:

```python
import logging

logger = logging.getLogger('autocognitix')

def analyze_with_logging(client, data):
    logger.info(f"Starting analysis for {data['vehicle_make']} {data['vehicle_model']}")
    try:
        result = client.analyze(data)
        logger.info(f"Analysis complete. Confidence: {result['confidence_score']}")
        return result
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise
```

---

## Troubleshooting

### Common Issues

#### 1. "Invalid DTC code format" Error

**Problem:** DTC code doesn't match expected format.

**Solution:** Ensure codes are 5 characters: letter (P/B/C/U) + 4 digits.

```python
# Correct
dtc_codes = ["P0101", "B1234"]

# Incorrect
dtc_codes = ["P101", "0101", "P-0101"]
```

#### 2. "VIN contains invalid characters" Error

**Problem:** VIN contains I, O, or Q.

**Solution:** VINs never contain these letters. Check for typos or OCR errors.

#### 3. Token Refresh Fails

**Problem:** Refresh token is expired or invalid.

**Solution:** User must login again. Clear stored tokens.

#### 4. Rate Limit Exceeded

**Problem:** Too many requests.

**Solution:** Implement caching and request batching. Wait before retrying.

#### 5. NHTSA API Errors (502)

**Problem:** External NHTSA service is unavailable.

**Solution:** Implement retry logic with exponential backoff. NHTSA may have temporary outages.

### Debug Mode

For development, enable detailed logging:

```python
import logging
import http.client

# Enable HTTP request/response logging
http.client.HTTPConnection.debuglevel = 1
logging.basicConfig(level=logging.DEBUG)
```

### Support

For issues not covered here:

- Email: support@autocognitix.hu
- Documentation: https://docs.autocognitix.hu
- API Status: https://status.autocognitix.hu

---

## Changelog

### Version 0.1.0 (Current)

- Initial API release
- Authentication endpoints
- Diagnosis endpoints with AI analysis
- DTC code search and lookup
- Vehicle information endpoints
- NHTSA integration (recalls, complaints)
- Hungarian language support
