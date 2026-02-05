# AutoCognitix API Reference

**Version:** 0.1.0
**Base URL:** `/api/v1`
**Last Updated:** 2026-02-05

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Rate Limiting](#rate-limiting)
- [Error Handling](#error-handling)
- [Endpoints](#endpoints)
  - [Health](#health-endpoints)
  - [Authentication](#authentication-endpoints)
  - [Diagnosis](#diagnosis-endpoints)
  - [DTC Codes](#dtc-codes-endpoints)
  - [Vehicles](#vehicle-endpoints)
  - [Metrics](#metrics-endpoints)
- [Code Examples](#code-examples)
- [DTC Code Format](#dtc-code-format)

---

## Overview

AutoCognitix is an AI-powered vehicle diagnostic platform with Hungarian language support.

### API Documentation

| Format | URL |
|--------|-----|
| Swagger UI | `/api/v1/docs` |
| ReDoc | `/api/v1/redoc` |
| OpenAPI JSON | `/api/v1/openapi.json` |

### Key Features

- AI-powered diagnostics using RAG (Retrieval Augmented Generation)
- Full Hungarian language support with huBERT embeddings
- Comprehensive OBD-II DTC code database
- VIN decoding via NHTSA API
- NHTSA recalls and complaints integration

---

## Authentication

### Overview

Most API endpoints require JWT authentication. Tokens are obtained via the login endpoint.

### Token Types

| Type | Expiry | Purpose |
|------|--------|---------|
| Access Token | 30 minutes | API access |
| Refresh Token | 7 days | Obtain new access token |

### Using Tokens

Include the access token in the `Authorization` header:

```
Authorization: Bearer <access_token>
```

### Token Lifecycle

1. User logs in via `POST /api/v1/auth/login`
2. Receives access and refresh tokens
3. Uses access token for API requests
4. When access token expires, uses refresh token to get new tokens
5. If refresh token expires, user must log in again

---

## Rate Limiting

| Limit | Value |
|-------|-------|
| Per minute | 60 requests |
| Per hour | 1000 requests |

### Rate Limit Headers

Responses include rate limit information:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 58
X-RateLimit-Reset: 1707134400
```

### Rate Limit Exceeded Response

```json
{
  "detail": "Rate limit exceeded. Please wait before making more requests."
}
```

---

## Error Handling

### HTTP Status Codes

| Code | Description | Common Causes |
|------|-------------|---------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created |
| 400 | Bad Request | Invalid input, malformed data |
| 401 | Unauthorized | Missing/invalid token |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource doesn't exist |
| 422 | Unprocessable Entity | Validation error |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 502 | Bad Gateway | External API error (NHTSA) |

### Error Response Format

**Standard Error:**
```json
{
  "detail": "Error message describing the problem"
}
```

**Validation Error (422):**
```json
{
  "detail": [
    {
      "loc": ["body", "field_name"],
      "msg": "Error description",
      "type": "error_type"
    }
  ]
}
```

---

## Endpoints

### Health Endpoints

#### GET `/health`

Basic health check for container orchestration.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "service": "autocognitix-backend",
  "environment": "production"
}
```

---

#### GET `/api/v1/health/detailed`

Detailed health check for all services.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "environment": "production",
  "services": {
    "postgres": {
      "name": "PostgreSQL",
      "status": "healthy",
      "latency_ms": 12.5,
      "details": {
        "table_counts": {
          "dtc_codes": 63,
          "users": 10
        }
      }
    },
    "neo4j": {
      "name": "Neo4j",
      "status": "healthy",
      "latency_ms": 45.2,
      "details": {
        "node_counts": {
          "DTCCode": 63,
          "Symptom": 150
        }
      }
    },
    "qdrant": {
      "name": "Qdrant",
      "status": "healthy",
      "latency_ms": 8.1
    },
    "redis": {
      "name": "Redis",
      "status": "healthy",
      "latency_ms": 2.3
    }
  },
  "checked_at": "2024-02-03T10:30:00Z"
}
```

---

#### GET `/api/v1/health/ready`

Kubernetes/Railway readiness probe.

**Response (200):**
```json
{
  "status": "ready",
  "checked_at": "2024-02-03T10:30:00Z"
}
```

**Response (503):**
```json
{
  "detail": "Service not ready: PostgreSQL - Connection refused"
}
```

---

#### GET `/api/v1/health/live`

Kubernetes/Railway liveness probe.

**Response:**
```json
{
  "status": "alive",
  "checked_at": "2024-02-03T10:30:00Z"
}
```

---

### Authentication Endpoints

#### POST `/api/v1/auth/register`

Register a new user account.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword123",
  "full_name": "John Doe"
}
```

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| email | string | Yes | Valid email format |
| password | string | Yes | 8-100 characters |
| full_name | string | No | Max 100 characters |

**Response (201 Created):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "full_name": "John Doe",
  "is_active": true,
  "role": "user"
}
```

**Errors:**
- `400` - Email already registered
- `422` - Validation error

---

#### POST `/api/v1/auth/login`

Authenticate user and obtain tokens.

**Request:** `application/x-www-form-urlencoded`

| Field | Type | Required |
|-------|------|----------|
| username | string | Yes (email) |
| password | string | Yes |

**Response (200 OK):**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer"
}
```

**Errors:**
- `401` - Invalid credentials

---

#### POST `/api/v1/auth/refresh`

Refresh access token.

**Request Body:**
```json
{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer"
}
```

**Errors:**
- `401` - Invalid or expired refresh token

---

#### GET `/api/v1/auth/me`

Get current authenticated user.

**Headers:** `Authorization: Bearer <access_token>`

**Response (200 OK):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "full_name": "John Doe",
  "is_active": true,
  "role": "user"
}
```

**Errors:**
- `401` - Invalid or missing access token

---

### Diagnosis Endpoints

#### POST `/api/v1/diagnosis/analyze`

**Main diagnostic endpoint** - AI-powered vehicle analysis.

**Request Body:**
```json
{
  "vehicle_make": "Volkswagen",
  "vehicle_model": "Golf",
  "vehicle_year": 2018,
  "vehicle_engine": "2.0 TSI",
  "vin": "WVWZZZ3CZWE123456",
  "dtc_codes": ["P0101", "P0171"],
  "symptoms": "A motor nehezen indul hidegben, egyenetlenul jar alapjaraton, es a fogyasztas megnott.",
  "additional_context": "A problema telen rosszabb."
}
```

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| vehicle_make | string | Yes | 1-100 characters |
| vehicle_model | string | Yes | 1-100 characters |
| vehicle_year | integer | Yes | 1900-2030 |
| vehicle_engine | string | No | Max 100 characters |
| vin | string | No | Exactly 17 characters |
| dtc_codes | array | Yes | 1-20 items |
| symptoms | string | Yes | 10-2000 characters (Hungarian) |
| additional_context | string | No | Max 1000 characters |

**Response (201 Created):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "vehicle_make": "Volkswagen",
  "vehicle_model": "Golf",
  "vehicle_year": 2018,
  "dtc_codes": ["P0101", "P0171"],
  "symptoms": "A motor nehezen indul hidegben...",
  "probable_causes": [
    {
      "title": "MAF szenzor hiba",
      "description": "A levegotomeg-mero szenzor hibas vagy szennyezett. Ez okozza a motor teljesitmenyveszteseget es az egyenetlen alapjaratot.",
      "confidence": 0.85,
      "related_dtc_codes": ["P0101"],
      "components": ["MAF szenzor", "Levegoszuro"]
    },
    {
      "title": "Vakuumszivarga",
      "description": "Szivarga a szivocsoben vagy a vakuumvezetekekben.",
      "confidence": 0.65,
      "related_dtc_codes": ["P0171"],
      "components": ["Szivocso", "Vakuumvezetekek"]
    }
  ],
  "recommended_repairs": [
    {
      "title": "MAF szenzor tisztitasa/csereje",
      "description": "Ellenorizze es tisztitsa meg a MAF szenzort specialis tisztitoval. Ha a tisztitas nem segit, csereje szukseges.",
      "estimated_cost_min": 5000,
      "estimated_cost_max": 45000,
      "estimated_cost_currency": "HUF",
      "difficulty": "intermediate",
      "parts_needed": ["MAF szenzor tisztito", "MAF szenzor (csere eseten)"],
      "estimated_time_minutes": 30
    }
  ],
  "confidence_score": 0.82,
  "sources": [
    {
      "type": "database",
      "title": "OBD-II DTC Database",
      "url": null,
      "relevance_score": 0.95
    }
  ],
  "created_at": "2024-02-03T10:30:00Z"
}
```

**Errors:**
- `400` - Invalid DTC codes or VIN
- `500` - Internal service error

---

#### GET `/api/v1/diagnosis/{diagnosis_id}`

Retrieve a specific diagnosis by ID.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| diagnosis_id | UUID | Unique diagnosis identifier |

**Response (200 OK):** Same as analyze response

**Errors:**
- `404` - Diagnosis not found

---

#### GET `/api/v1/diagnosis/history/list`

Get diagnosis history for the current user.

**Query Parameters:**

| Parameter | Type | Default | Constraints |
|-----------|------|---------|-------------|
| skip | integer | 0 | >= 0 |
| limit | integer | 10 | 1-100 |

**Response (200 OK):**
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "vehicle_make": "Volkswagen",
    "vehicle_model": "Golf",
    "vehicle_year": 2018,
    "dtc_codes": ["P0101"],
    "confidence_score": 0.85,
    "created_at": "2024-02-03T10:30:00Z"
  }
]
```

---

#### POST `/api/v1/diagnosis/quick-analyze`

Quick DTC code lookup without full AI analysis.

**Query Parameters:**

| Parameter | Type | Required | Constraints |
|-----------|------|----------|-------------|
| dtc_codes | array | Yes | 1-10 items |

**Example:** `POST /api/v1/diagnosis/quick-analyze?dtc_codes=P0101&dtc_codes=P0171`

**Response (200 OK):**
```json
{
  "dtc_codes": [
    {
      "code": "P0101",
      "description": "Levegotomeg-mero aramkor tartomany/teljesitmeny hiba",
      "severity": "medium",
      "symptoms": ["Motor teljesitmenyvesztese", "Egyenetlen alapjarat"],
      "possible_causes": ["Szennyezett MAF szenzor", "Levegoszuro eltomodes"]
    }
  ],
  "message": "Reszletes diagnozishoz hasznalja a /analyze vegpontot jarmuadatokkal."
}
```

**Errors:**
- `400` - Invalid DTC code format

---

### DTC Codes Endpoints

#### GET `/api/v1/dtc/search`

Search for DTC codes by code or description.

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| q | string | Yes | - | Search query (min 1 char) |
| category | string | No | - | Filter by category |
| make | string | No | - | Filter by vehicle make |
| limit | integer | No | 20 | Max results (1-100) |

**Categories:** `powertrain`, `body`, `chassis`, `network`

**Response (200 OK):**
```json
[
  {
    "code": "P0101",
    "description_en": "Mass Air Flow Circuit Range/Performance",
    "description_hu": "Levegotomeg-mero aramkor tartomany/teljesitmeny hiba",
    "category": "powertrain",
    "severity": "medium",
    "is_generic": true,
    "relevance_score": 0.95
  }
]
```

---

#### GET `/api/v1/dtc/{code}`

Get detailed information about a specific DTC code.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| code | string | DTC code (e.g., P0101) |

**Response (200 OK):**
```json
{
  "code": "P0101",
  "description_en": "Mass Air Flow Circuit Range/Performance",
  "description_hu": "Levegotomeg-mero aramkor tartomany/teljesitmeny hiba",
  "category": "powertrain",
  "severity": "medium",
  "is_generic": true,
  "system": "Fuel and Air Metering",
  "symptoms": [
    "Motor teljesitmenyvesztese",
    "Egyenetlen alapjarat",
    "Nehez inditas",
    "Megnovelt uzemanyag-fogyasztas",
    "Fekete fust a kipufogobol"
  ],
  "possible_causes": [
    "Szennyezett MAF szenzor",
    "Levegoszuro eltomodes",
    "Vakuumszivarga a szivorendszerben",
    "MAF szenzor meghibasodasa",
    "Vezetek vagy csatlakozo problema"
  ],
  "diagnostic_steps": [
    "1. Vizualisan ellenorizze a MAF szenzort es a levegoszurot",
    "2. Ellenorizze a MAF szenzor csatlakozojat es vezetkeit",
    "3. Tisztitsa meg a MAF szenzort specialis tisztitoval",
    "4. Ellenorizze a szivorendszert vakuumszivarga szempontjabol",
    "5. Tesztelje a MAF szenzor jelet oszcilloszkoppal"
  ],
  "related_codes": ["P0100", "P0102", "P0103", "P0171", "P0174"],
  "common_vehicles": [
    "Volkswagen Golf (2010-2020)",
    "Audi A3 (2012-2020)",
    "Ford Focus (2011-2018)",
    "Toyota Corolla (2014-2019)"
  ]
}
```

**Errors:**
- `400` - Invalid DTC code format
- `404` - DTC code not found

---

#### GET `/api/v1/dtc/{code}/related`

Get DTC codes related to the specified code.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| code | string | DTC code |

**Response (200 OK):**
```json
[
  {
    "code": "P0100",
    "description_en": "Mass Air Flow Circuit Malfunction",
    "description_hu": "Levegotomeg-mero aramkor meghibasodas",
    "category": "powertrain",
    "severity": "medium",
    "is_generic": true
  }
]
```

---

#### GET `/api/v1/dtc/categories/list`

Get list of DTC categories.

**Response (200 OK):**
```json
[
  {
    "code": "P",
    "name": "Powertrain",
    "name_hu": "Hajtaslanc",
    "description": "Engine, transmission, and emission systems",
    "description_hu": "Motor, valto es emissziós rendszerek"
  },
  {
    "code": "B",
    "name": "Body",
    "name_hu": "Karosszeria",
    "description": "Body systems including airbags, A/C, lighting",
    "description_hu": "Karosszeria rendszerek: legzsakok, klima, vilagitas"
  },
  {
    "code": "C",
    "name": "Chassis",
    "name_hu": "Alvaz",
    "description": "Chassis systems including ABS, steering, suspension",
    "description_hu": "Alvaz rendszerek: ABS, kormanyzas, felfuggesztes"
  },
  {
    "code": "U",
    "name": "Network",
    "name_hu": "Halozat",
    "description": "Communication network and module systems",
    "description_hu": "Kommunikacios halozat es vezerlo modulok"
  }
]
```

---

### Vehicle Endpoints

#### POST `/api/v1/vehicles/decode-vin`

Decode a VIN (Vehicle Identification Number).

**Request Body:**
```json
{
  "vin": "WVWZZZ3CZWE123456"
}
```

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| vin | string | Yes | Exactly 17 characters |

**Response (200 OK):**
```json
{
  "vin": "WVWZZZ3CZWE123456",
  "make": "Volkswagen",
  "model": "Golf",
  "year": 2018,
  "trim": "Hatchback",
  "engine": "2.0L 4-cyl Gasoline",
  "transmission": "Automatic",
  "drive_type": "FWD",
  "body_type": "Hatchback",
  "fuel_type": "Gasoline",
  "region": "Europe",
  "country_of_origin": "Germany"
}
```

**Errors:**
- `400` - Invalid VIN format or invalid characters (I, O, Q not allowed)
- `502` - NHTSA API error

---

#### GET `/api/v1/vehicles/makes`

Get list of vehicle makes (manufacturers).

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| search | string | No | Search term for filtering |

**Response (200 OK):**
```json
[
  {
    "id": "volkswagen",
    "name": "Volkswagen",
    "country": "Germany",
    "logo_url": null
  },
  {
    "id": "audi",
    "name": "Audi",
    "country": "Germany",
    "logo_url": null
  }
]
```

---

#### GET `/api/v1/vehicles/models/{make_id}`

Get list of models for a specific make.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| make_id | string | ID of the vehicle make |

**Query Parameters:**

| Parameter | Type | Required | Constraints |
|-----------|------|----------|-------------|
| year | integer | No | 1900-2030 |

**Response (200 OK):**
```json
[
  {
    "id": "golf",
    "name": "Golf",
    "make_id": "volkswagen",
    "year_start": 1974,
    "year_end": null,
    "body_types": []
  }
]
```

---

#### GET `/api/v1/vehicles/years`

Get list of available vehicle years.

**Response (200 OK):**
```json
{
  "years": [2027, 2026, 2025, 2024, 2023, "...", 1980]
}
```

---

#### GET `/api/v1/vehicles/{make}/{model}/{year}/recalls`

Get recall information from NHTSA.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| make | string | Vehicle manufacturer |
| model | string | Vehicle model |
| year | integer | Model year (1900-2030) |

**Response (200 OK):**
```json
[
  {
    "campaign_number": "24V123000",
    "manufacturer": "Volkswagen",
    "subject": "Fuel Pump May Fail",
    "summary": "The fuel pump may fail causing the engine to stall...",
    "consequence": "An engine stall while driving increases the risk of a crash.",
    "remedy": "Dealers will replace the fuel pump free of charge.",
    "report_received_date": "2024-01-15",
    "component": "FUEL SYSTEM"
  }
]
```

**Errors:**
- `502` - NHTSA API error

---

#### GET `/api/v1/vehicles/{make}/{model}/{year}/complaints`

Get complaint information from NHTSA.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| make | string | Vehicle manufacturer |
| model | string | Vehicle model |
| year | integer | Model year (1900-2030) |

**Response (200 OK):**
```json
[
  {
    "odiNumber": "12345678",
    "manufacturer": "Volkswagen",
    "crash": false,
    "fire": false,
    "numberOfInjuries": 0,
    "numberOfDeaths": 0,
    "dateOfIncident": "2023-06-15",
    "dateComplaintFiled": "2023-06-20",
    "summary": "The vehicle experienced sudden power loss while driving...",
    "components": "ENGINE"
  }
]
```

**Errors:**
- `502` - NHTSA API error

---

### Metrics Endpoints

#### GET `/api/v1/metrics`

Prometheus metrics endpoint.

**Response:** Prometheus text format

```
# HELP autocognitix_requests_total Total request count
# TYPE autocognitix_requests_total counter
autocognitix_requests_total{method="GET",endpoint="/api/v1/health",status="200"} 150

# HELP autocognitix_request_latency_seconds Request latency in seconds
# TYPE autocognitix_request_latency_seconds histogram
autocognitix_request_latency_seconds_bucket{method="GET",endpoint="/api/v1/health",le="0.01"} 145
```

---

#### GET `/api/v1/metrics/summary`

Human-readable metrics summary.

**Response (200 OK):**
```json
{
  "dtc_codes_total": 63,
  "environment": "production",
  "project_name": "AutoCognitix",
  "metrics_endpoint": "/metrics"
}
```

---

## Code Examples

### cURL

**Login:**
```bash
curl -X POST "https://api.autocognitix.hu/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=yourpassword"
```

**Analyze Vehicle:**
```bash
curl -X POST "https://api.autocognitix.hu/api/v1/diagnosis/analyze" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "vehicle_make": "Volkswagen",
    "vehicle_model": "Golf",
    "vehicle_year": 2018,
    "dtc_codes": ["P0101", "P0171"],
    "symptoms": "A motor nehezen indul hidegben, egyenetlenul jar alapjaraton."
  }'
```

**Search DTC Codes:**
```bash
curl -X GET "https://api.autocognitix.hu/api/v1/dtc/search?q=P0101&limit=10" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

**Decode VIN:**
```bash
curl -X POST "https://api.autocognitix.hu/api/v1/vehicles/decode-vin" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"vin": "WVWZZZ3CZWE123456"}'
```

---

### Python

```python
import requests

BASE_URL = "https://api.autocognitix.hu/api/v1"

# Login
def login(email: str, password: str) -> dict:
    response = requests.post(
        f"{BASE_URL}/auth/login",
        data={"username": email, "password": password}
    )
    response.raise_for_status()
    return response.json()

# Analyze vehicle
def analyze_vehicle(token: str, data: dict) -> dict:
    response = requests.post(
        f"{BASE_URL}/diagnosis/analyze",
        headers={"Authorization": f"Bearer {token}"},
        json=data
    )
    response.raise_for_status()
    return response.json()

# Search DTC codes
def search_dtc(token: str, query: str, limit: int = 20) -> list:
    response = requests.get(
        f"{BASE_URL}/dtc/search",
        headers={"Authorization": f"Bearer {token}"},
        params={"q": query, "limit": limit}
    )
    response.raise_for_status()
    return response.json()

# Example usage
if __name__ == "__main__":
    # Login
    tokens = login("user@example.com", "yourpassword")
    access_token = tokens["access_token"]

    # Analyze vehicle
    diagnosis = analyze_vehicle(access_token, {
        "vehicle_make": "Volkswagen",
        "vehicle_model": "Golf",
        "vehicle_year": 2018,
        "dtc_codes": ["P0101"],
        "symptoms": "A motor nehezen indul hidegben."
    })

    print(f"Diagnosis confidence: {diagnosis['confidence_score']}")
    for cause in diagnosis["probable_causes"]:
        print(f"- {cause['title']}: {cause['confidence']:.0%}")
```

---

### JavaScript (Fetch)

```javascript
const BASE_URL = 'https://api.autocognitix.hu/api/v1';

// Login
async function login(email, password) {
  const response = await fetch(`${BASE_URL}/auth/login`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: new URLSearchParams({
      username: email,
      password: password,
    }),
  });

  if (!response.ok) {
    throw new Error(`Login failed: ${response.status}`);
  }

  return response.json();
}

// Analyze vehicle
async function analyzeVehicle(token, data) {
  const response = await fetch(`${BASE_URL}/diagnosis/analyze`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    throw new Error(`Analysis failed: ${response.status}`);
  }

  return response.json();
}

// Search DTC codes
async function searchDtc(token, query, limit = 20) {
  const params = new URLSearchParams({ q: query, limit: limit.toString() });
  const response = await fetch(`${BASE_URL}/dtc/search?${params}`, {
    headers: {
      'Authorization': `Bearer ${token}`,
    },
  });

  if (!response.ok) {
    throw new Error(`Search failed: ${response.status}`);
  }

  return response.json();
}

// Example usage
async function main() {
  // Login
  const tokens = await login('user@example.com', 'yourpassword');
  const accessToken = tokens.access_token;

  // Analyze vehicle
  const diagnosis = await analyzeVehicle(accessToken, {
    vehicle_make: 'Volkswagen',
    vehicle_model: 'Golf',
    vehicle_year: 2018,
    dtc_codes: ['P0101'],
    symptoms: 'A motor nehezen indul hidegben.',
  });

  console.log(`Diagnosis confidence: ${diagnosis.confidence_score}`);
  diagnosis.probable_causes.forEach(cause => {
    console.log(`- ${cause.title}: ${(cause.confidence * 100).toFixed(0)}%`);
  });
}

main().catch(console.error);
```

---

### JavaScript (Axios)

```javascript
import axios from 'axios';

const api = axios.create({
  baseURL: 'https://api.autocognitix.hu/api/v1',
});

// Add auth interceptor
api.interceptors.request.use(config => {
  const token = localStorage.getItem('accessToken');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// API functions
export const authApi = {
  login: (email, password) =>
    api.post('/auth/login',
      new URLSearchParams({ username: email, password }),
      { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } }
    ),

  refresh: (refreshToken) =>
    api.post('/auth/refresh', { refresh_token: refreshToken }),

  me: () => api.get('/auth/me'),
};

export const diagnosisApi = {
  analyze: (data) => api.post('/diagnosis/analyze', data),

  get: (id) => api.get(`/diagnosis/${id}`),

  history: (skip = 0, limit = 10) =>
    api.get('/diagnosis/history/list', { params: { skip, limit } }),

  quickAnalyze: (dtcCodes) =>
    api.post('/diagnosis/quick-analyze', null, { params: { dtc_codes: dtcCodes } }),
};

export const dtcApi = {
  search: (query, options = {}) =>
    api.get('/dtc/search', { params: { q: query, ...options } }),

  get: (code) => api.get(`/dtc/${code}`),

  related: (code) => api.get(`/dtc/${code}/related`),

  categories: () => api.get('/dtc/categories/list'),
};

export const vehicleApi = {
  decodeVin: (vin) => api.post('/vehicles/decode-vin', { vin }),

  makes: (search) => api.get('/vehicles/makes', { params: { search } }),

  models: (makeId, year) =>
    api.get(`/vehicles/models/${makeId}`, { params: { year } }),

  years: () => api.get('/vehicles/years'),

  recalls: (make, model, year) =>
    api.get(`/vehicles/${make}/${model}/${year}/recalls`),

  complaints: (make, model, year) =>
    api.get(`/vehicles/${make}/${model}/${year}/complaints`),
};
```

---

## DTC Code Format

DTC codes follow the standard OBD-II format:

| Position | Description | Values |
|----------|-------------|--------|
| 1 | Category | P (Powertrain), B (Body), C (Chassis), U (Network) |
| 2 | Type | 0 (Generic), 1 (Manufacturer-specific) |
| 3-5 | Specific code | 000-999 |

### Examples

| Code | Category | Type | Description |
|------|----------|------|-------------|
| P0101 | Powertrain | Generic | MAF sensor issue |
| P1234 | Powertrain | Manufacturer | OEM-specific |
| B0015 | Body | Generic | Body system issue |
| C0035 | Chassis | Generic | Chassis issue |
| U0100 | Network | Generic | CAN bus issue |

### Category Descriptions

| Code | Category | Hungarian | Systems |
|------|----------|-----------|---------|
| P | Powertrain | Hajtaslanc | Engine, transmission, emissions |
| B | Body | Karosszeria | Airbags, A/C, lighting |
| C | Chassis | Alvaz | ABS, steering, suspension |
| U | Network | Halozat | CAN bus, modules |

---

## External Data Sources

| Service | Purpose | Rate Limited |
|---------|---------|--------------|
| NHTSA vPIC API | VIN decoding | Yes |
| NHTSA Recalls API | Recall information | Yes |
| NHTSA Complaints API | Complaint data | Yes |

---

## Support

For API support:
- Email: support@autocognitix.hu
- Documentation: https://docs.autocognitix.hu
- Repository: https://github.com/autocognitix
