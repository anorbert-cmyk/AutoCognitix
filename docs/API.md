# AutoCognitix API Documentation

## Overview

AutoCognitix is an AI-powered vehicle diagnostic platform with Hungarian language support. This document describes all available REST API endpoints.

**Base URL:** `/api/v1`

**OpenAPI Documentation:**
- Swagger UI: `/api/v1/docs`
- ReDoc: `/api/v1/redoc`
- OpenAPI JSON: `/api/v1/openapi.json`

---

## Quick Start Guide

### 1. Register a User Account

```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123!",
    "full_name": "Kovacs Janos"
  }'
```

### 2. Login and Get Access Token

```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=SecurePass123!"
```

Save the `access_token` from the response for authenticated requests.

### 3. Run a Vehicle Diagnosis

```bash
curl -X POST "http://localhost:8000/api/v1/diagnosis/analyze" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -d '{
    "vehicle_make": "Volkswagen",
    "vehicle_model": "Golf",
    "vehicle_year": 2018,
    "dtc_codes": ["P0101", "P0171"],
    "symptoms": "A motor nehezen indul hidegben, egyenetlenul jar alapjaraton."
  }'
```

### 4. Quick DTC Code Lookup (No Auth Required)

```bash
curl -X POST "http://localhost:8000/api/v1/diagnosis/quick-analyze?dtc_codes=P0101&dtc_codes=P0171"
```

---

## Authentication

All authenticated endpoints require a Bearer token in the `Authorization` header:

```
Authorization: Bearer <access_token>
```

### Token Types

| Type | Expiry | Purpose |
|------|--------|---------|
| Access Token | 30 minutes | API access |
| Refresh Token | 7 days | Obtain new access token |

### Token Flow

```
1. POST /auth/login        -> Returns access_token + refresh_token
2. Use access_token        -> For all authenticated requests
3. POST /auth/refresh      -> When access_token expires, get new tokens
4. POST /auth/logout       -> Invalidate tokens
```

---

## Endpoints

### Health Check

#### GET `/health`

Check service health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "service": "autocognitix-backend",
  "environment": "development"
}
```

#### GET `/health/ready`

Kubernetes readiness probe - checks database connectivity.

#### GET `/health/live`

Kubernetes liveness probe - checks if application is running.

#### GET `/health/detailed`

Detailed health check including all services (PostgreSQL, Neo4j, Qdrant, Redis).

---

## Authentication Endpoints

Tag: `Authentication`

### POST `/api/v1/auth/register`

Register a new user.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "SecurePass123!",
  "full_name": "Kovacs Janos"
}
```

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| email | string | Yes | Valid email format |
| password | string | Yes | 8-100 chars, must include uppercase, lowercase, number, special char |
| full_name | string | No | Max 100 characters |

**Response (201 Created):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "full_name": "Kovacs Janos",
  "is_active": true,
  "role": "user"
}
```

**curl Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"SecurePass123!","full_name":"Kovacs Janos"}'
```

---

### POST `/api/v1/auth/login`

Authenticate user and obtain tokens.

**Request Body:** `application/x-www-form-urlencoded`

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

**curl Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=SecurePass123!"
```

---

### POST `/api/v1/auth/refresh`

Refresh access token using refresh token.

**Request Body:**
```json
{
  "refresh_token": "eyJ0eXAiOiJKV1..."
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJ0eXAiOiJKV1...",
  "refresh_token": "eyJ0eXAiOiJKV1...",
  "token_type": "bearer"
}
```

**curl Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/auth/refresh" \
  -H "Content-Type: application/json" \
  -d '{"refresh_token":"YOUR_REFRESH_TOKEN"}'
```

---

### POST `/api/v1/auth/logout`

Logout user and invalidate tokens.

**Headers:** `Authorization: Bearer <access_token>`

**Request Body (optional):**
```json
{
  "refresh_token": "eyJ0eXAiOiJKV1..."
}
```

**curl Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/auth/logout" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"refresh_token":"YOUR_REFRESH_TOKEN"}'
```

---

### GET `/api/v1/auth/me`

Get current authenticated user information.

**Headers:** `Authorization: Bearer <access_token>`

**Response (200 OK):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "full_name": "Kovacs Janos",
  "is_active": true,
  "role": "user",
  "created_at": "2024-02-03T10:30:00Z"
}
```

**curl Example:**
```bash
curl -X GET "http://localhost:8000/api/v1/auth/me" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

---

### PUT `/api/v1/auth/me`

Update current user profile.

**Headers:** `Authorization: Bearer <access_token>`

**Request Body:**
```json
{
  "full_name": "Updated Name",
  "email": "newemail@example.com"
}
```

---

### PUT `/api/v1/auth/me/password`

Change current user's password.

**Headers:** `Authorization: Bearer <access_token>`

**Request Body:**
```json
{
  "current_password": "OldPass123!",
  "new_password": "NewSecurePass456!"
}
```

---

### POST `/api/v1/auth/forgot-password`

Request password reset token.

**Request Body:**
```json
{
  "email": "user@example.com"
}
```

---

### POST `/api/v1/auth/reset-password`

Reset password using reset token.

**Request Body:**
```json
{
  "token": "reset_token_from_email",
  "new_password": "NewSecurePass456!"
}
```

---

## Diagnosis Endpoints

Tag: `Diagnosis`

### POST `/api/v1/diagnosis/analyze`

**Main diagnostic endpoint** - Analyze vehicle based on DTC codes and symptoms using AI.

**Headers:** `Authorization: Bearer <access_token>` (optional - saves to history if authenticated)

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
| dtc_codes | array[string] | Yes | 1-20 items |
| symptoms | string | Yes | 10-2000 characters (Hungarian supported) |
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
      "description": "A levegotomeg-mero szenzor hibas vagy szennyezett...",
      "confidence": 0.85,
      "related_dtc_codes": ["P0101"],
      "components": ["MAF szenzor", "Levegoszuro"]
    }
  ],
  "recommended_repairs": [
    {
      "title": "MAF szenzor tisztitasa/csereje",
      "description": "Ellenorizze es tisztitsa meg a MAF szenzort...",
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

**curl Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/diagnosis/analyze" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -d '{
    "vehicle_make": "Volkswagen",
    "vehicle_model": "Golf",
    "vehicle_year": 2018,
    "dtc_codes": ["P0101"],
    "symptoms": "A motor nehezen indul hidegben."
  }'
```

---

### GET `/api/v1/diagnosis/{diagnosis_id}`

Get a specific diagnosis by ID.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| diagnosis_id | UUID | Unique diagnosis identifier |

**curl Example:**
```bash
curl -X GET "http://localhost:8000/api/v1/diagnosis/550e8400-e29b-41d4-a716-446655440000"
```

---

### GET `/api/v1/diagnosis/history/list`

Get diagnosis history for the current user with pagination and filters.

**Headers:** `Authorization: Bearer <access_token>` (required)

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| skip | integer | 0 | Records to skip |
| limit | integer | 10 | Max records (1-100) |
| vehicle_make | string | - | Filter by make |
| vehicle_model | string | - | Filter by model |
| vehicle_year | integer | - | Filter by year |
| dtc_code | string | - | Filter by DTC code |
| date_from | datetime | - | Filter by start date |
| date_to | datetime | - | Filter by end date |

**Response (200 OK):**
```json
{
  "items": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "vehicle_make": "Volkswagen",
      "vehicle_model": "Golf",
      "vehicle_year": 2018,
      "dtc_codes": ["P0101"],
      "confidence_score": 0.85,
      "created_at": "2024-02-03T10:30:00Z"
    }
  ],
  "total": 15,
  "skip": 0,
  "limit": 10,
  "has_more": true
}
```

**curl Example:**
```bash
curl -X GET "http://localhost:8000/api/v1/diagnosis/history/list?limit=20&vehicle_make=Volkswagen" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

---

### DELETE `/api/v1/diagnosis/{diagnosis_id}`

Soft delete a diagnosis (requires authentication, user can only delete own diagnoses).

**Headers:** `Authorization: Bearer <access_token>` (required)

**curl Example:**
```bash
curl -X DELETE "http://localhost:8000/api/v1/diagnosis/550e8400-e29b-41d4-a716-446655440000" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

---

### GET `/api/v1/diagnosis/stats/summary`

Get diagnosis statistics for the current user.

**Headers:** `Authorization: Bearer <access_token>` (required)

**Response (200 OK):**
```json
{
  "total_diagnoses": 25,
  "avg_confidence": 0.78,
  "most_diagnosed_vehicles": [
    {"make": "Volkswagen", "model": "Golf", "count": 10}
  ],
  "most_common_dtcs": [
    {"code": "P0101", "count": 8}
  ],
  "diagnoses_by_month": [
    {"month": "2024-02", "count": 5}
  ]
}
```

---

### POST `/api/v1/diagnosis/quick-analyze`

Quick DTC code lookup without full RAG analysis (no authentication required).

**Query Parameters:**
| Parameter | Type | Required | Constraints |
|-----------|------|----------|-------------|
| dtc_codes | array[string] | Yes | 1-10 items |

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

**curl Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/diagnosis/quick-analyze?dtc_codes=P0101&dtc_codes=P0171"
```

---

## DTC Codes Endpoints

Tag: `DTC Codes`

### GET `/api/v1/dtc/search`

Search for DTC codes by code or description.

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| q | string | Yes | - | Search query (min 1 char) |
| category | string | No | - | Filter by category |
| make | string | No | - | Filter by vehicle make |
| limit | integer | No | 20 | Max results (1-100) |
| use_semantic | boolean | No | true | Use AI semantic search |
| skip_cache | boolean | No | false | Force fresh results |

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

**curl Examples:**
```bash
# Search by code
curl "http://localhost:8000/api/v1/dtc/search?q=P0101"

# Search by description (Hungarian)
curl "http://localhost:8000/api/v1/dtc/search?q=motor%20nehezen%20indul"

# Filter by category
curl "http://localhost:8000/api/v1/dtc/search?q=MAF&category=powertrain"
```

---

### GET `/api/v1/dtc/{code}`

Get detailed information about a specific DTC code.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| code | string | DTC code (e.g., P0101) |

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| include_graph | boolean | true | Include Neo4j relationships |
| skip_cache | boolean | false | Force fresh results |

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
    "Nehez inditas"
  ],
  "possible_causes": [
    "Szennyezett MAF szenzor",
    "Levegoszuro eltomodes",
    "Vakuumszivarga a szivórendszerben"
  ],
  "diagnostic_steps": [
    "1. Vizualisan ellenorizze a MAF szenzort es a levegoszurot",
    "2. Ellenorizze a MAF szenzor csatlakozojat es vezetékeit",
    "3. Tisztitsa meg a MAF szenzort specialis tisztitoval"
  ],
  "related_codes": ["P0100", "P0102", "P0103", "P0171", "P0174"],
  "common_vehicles": []
}
```

**curl Example:**
```bash
curl "http://localhost:8000/api/v1/dtc/P0101"
```

---

### GET `/api/v1/dtc/{code}/related`

Get DTC codes related to the specified code.

**curl Example:**
```bash
curl "http://localhost:8000/api/v1/dtc/P0101/related?limit=5"
```

---

### GET `/api/v1/dtc/categories/list`

Get list of DTC categories with descriptions in English and Hungarian.

**Response (200 OK):**
```json
[
  {
    "code": "P",
    "name": "Powertrain",
    "name_hu": "Hajtaslánc",
    "description": "Engine, transmission, and emission systems",
    "description_hu": "Motor, váltó és emissziós rendszerek"
  },
  {
    "code": "B",
    "name": "Body",
    "name_hu": "Karosszéria",
    "description": "Body systems including airbags, A/C, lighting",
    "description_hu": "Karosszéria rendszerek: légzsákok, klíma, világítás"
  },
  {
    "code": "C",
    "name": "Chassis",
    "name_hu": "Alváz",
    "description": "Chassis systems including ABS, steering, suspension",
    "description_hu": "Alváz rendszerek: ABS, kormányzás, felfüggesztés"
  },
  {
    "code": "U",
    "name": "Network",
    "name_hu": "Hálózat",
    "description": "Communication network and module systems",
    "description_hu": "Kommunikációs hálózat és vezérlő modulok"
  }
]
```

---

### POST `/api/v1/dtc/`

Create a new DTC code entry.

**Request Body:**
```json
{
  "code": "P0101",
  "description_en": "Mass Air Flow Circuit Range/Performance",
  "description_hu": "Levegotomeg-mero aramkor tartomany/teljesitmeny hiba",
  "category": "powertrain",
  "severity": "medium",
  "is_generic": true,
  "symptoms": ["Motor teljesitmenyvesztese"],
  "possible_causes": ["Szennyezett MAF szenzor"]
}
```

---

### POST `/api/v1/dtc/bulk`

Bulk import DTC codes.

**Request Body:**
```json
{
  "codes": [
    {
      "code": "P0101",
      "description_en": "Mass Air Flow Circuit Range/Performance",
      "category": "powertrain",
      "severity": "medium"
    }
  ],
  "overwrite_existing": false
}
```

**Response (201 Created):**
```json
{
  "created": 10,
  "updated": 5,
  "skipped": 2,
  "errors": [],
  "total": 17
}
```

---

## Vehicle Endpoints

Tag: `Vehicles`

### POST `/api/v1/vehicles/decode-vin`

Decode a VIN (Vehicle Identification Number) to get vehicle details.

**Request Body:**
```json
{
  "vin": "WVWZZZ3CZWE123456"
}
```

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| vin | string | Yes | Exactly 17 characters, no I/O/Q |

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

**curl Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/vehicles/decode-vin" \
  -H "Content-Type: application/json" \
  -d '{"vin":"WVWZZZ3CZWE123456"}'
```

---

### GET `/api/v1/vehicles/makes`

Get list of vehicle makes (manufacturers).

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| search | string | No | Search term for filtering |

**curl Example:**
```bash
curl "http://localhost:8000/api/v1/vehicles/makes?search=volk"
```

---

### GET `/api/v1/vehicles/models/{make_id}`

Get list of models for a specific make.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| make_id | string | ID of the vehicle make |

**Query Parameters:**
| Parameter | Type | Required | Constraints |
|-----------|------|----------|-------------|
| year | integer | No | 1900-2030 |

**curl Example:**
```bash
curl "http://localhost:8000/api/v1/vehicles/models/volkswagen?year=2020"
```

---

### GET `/api/v1/vehicles/years`

Get list of available vehicle years.

**Response (200 OK):**
```json
{
  "years": [2027, 2026, 2025, 2024, 2023, 2022, 2021, 2020, "...", 1980]
}
```

---

### GET `/api/v1/vehicles/{make}/{model}/{year}/recalls`

Get recall information for a specific vehicle from NHTSA.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| make | string | Vehicle manufacturer |
| model | string | Vehicle model |
| year | integer | Model year (1900-2030) |

**curl Example:**
```bash
curl "http://localhost:8000/api/v1/vehicles/Volkswagen/Golf/2018/recalls"
```

---

### GET `/api/v1/vehicles/{make}/{model}/{year}/complaints`

Get complaint information for a specific vehicle from NHTSA.

**curl Example:**
```bash
curl "http://localhost:8000/api/v1/vehicles/Volkswagen/Golf/2018/complaints"
```

---

## Common Use Cases

### Use Case 1: Quick DTC Lookup

For a quick lookup without full diagnosis:

```bash
# Get basic info about a DTC code
curl "http://localhost:8000/api/v1/dtc/P0101"

# Search for related codes
curl "http://localhost:8000/api/v1/dtc/P0101/related"
```

### Use Case 2: Full Vehicle Diagnosis

For a comprehensive AI-powered diagnosis:

```bash
# 1. Login
TOKEN=$(curl -s -X POST "http://localhost:8000/api/v1/auth/login" \
  -d "username=user@example.com&password=SecurePass123!" | jq -r '.access_token')

# 2. Run diagnosis
curl -X POST "http://localhost:8000/api/v1/diagnosis/analyze" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "vehicle_make": "Volkswagen",
    "vehicle_model": "Golf",
    "vehicle_year": 2018,
    "dtc_codes": ["P0101", "P0171"],
    "symptoms": "A motor nehezen indul hidegben, es a fogyasztas megnott."
  }'
```

### Use Case 3: Vehicle Recall Check

Check if a vehicle has any safety recalls:

```bash
# Decode VIN first
curl -X POST "http://localhost:8000/api/v1/vehicles/decode-vin" \
  -H "Content-Type: application/json" \
  -d '{"vin":"WVWZZZ3CZWE123456"}'

# Check recalls
curl "http://localhost:8000/api/v1/vehicles/Volkswagen/Golf/2018/recalls"
```

### Use Case 4: Hungarian Language Search

Search for DTC codes using Hungarian symptoms:

```bash
# Semantic search in Hungarian
curl "http://localhost:8000/api/v1/dtc/search?q=motor%20nehezen%20indul%20hidegben&use_semantic=true"
```

---

## Error Codes Reference

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | OK - Request successful |
| 201 | Created - Resource created successfully |
| 400 | Bad Request - Invalid input data |
| 401 | Unauthorized - Authentication required or token invalid |
| 403 | Forbidden - Access denied |
| 404 | Not Found - Resource not found |
| 422 | Unprocessable Entity - Validation error |
| 423 | Locked - Account locked (too many failed login attempts) |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error - Server-side error |
| 502 | Bad Gateway - External API error (NHTSA) |

### Error Response Format

```json
{
  "detail": "Error message describing the problem"
}
```

### Validation Error Format (422)

```json
{
  "detail": [
    {
      "loc": ["body", "field_name"],
      "msg": "Error message",
      "type": "error_type"
    }
  ]
}
```

### Common Error Messages

| Error | Hungarian Message | Cause |
|-------|------------------|-------|
| Invalid token | Ervenytelen vagy lejart token | Expired or malformed JWT |
| Invalid credentials | Hibas email cim vagy jelszo | Wrong email/password |
| Account locked | Fiok zarolva | Too many failed login attempts |
| Invalid DTC format | Invalid DTC code format | DTC code doesn't match pattern |
| VIN invalid | VIN must be exactly 17 characters | Wrong VIN length |

---

## Rate Limiting

| Limit | Value |
|-------|-------|
| Per minute | 60 requests |
| Per hour | 1000 requests |

Rate limit headers are included in responses:
- `X-RateLimit-Limit` - Maximum requests allowed
- `X-RateLimit-Remaining` - Requests remaining in window
- `X-RateLimit-Reset` - Unix timestamp when limit resets

When rate limited, you receive a 429 response:
```json
{
  "detail": "Rate limit exceeded. Try again in 45 seconds."
}
```

---

## DTC Code Format

DTC codes follow the standard OBD-II format:

| Position | Description | Values |
|----------|-------------|--------|
| 1 | Category | P (Powertrain), B (Body), C (Chassis), U (Network) |
| 2 | Type | 0 (Generic), 1 (Manufacturer-specific) |
| 3-5 | Specific code | 000-999 |

**Examples:**
- `P0101` - Generic powertrain code (MAF sensor)
- `P1234` - Manufacturer-specific powertrain code
- `B0015` - Generic body code
- `C0035` - Generic chassis code
- `U0100` - Generic network code (CAN bus)

---

## External Data Sources

The API integrates with the following external services:

| Service | Purpose | Rate Limited | Documentation |
|---------|---------|--------------|---------------|
| NHTSA vPIC API | VIN decoding | Yes | [NHTSA API](https://vpic.nhtsa.dot.gov/api/) |
| NHTSA Recalls API | Recall information | Yes | [NHTSA Recalls](https://www.nhtsa.gov/recalls-api) |
| NHTSA Complaints API | Complaint data | Yes | [NHTSA Complaints](https://www.nhtsa.gov/webapi) |

---

## SDK and Client Libraries

### Python Example

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

# Login
response = requests.post(
    f"{BASE_URL}/auth/login",
    data={"username": "user@example.com", "password": "SecurePass123!"}
)
token = response.json()["access_token"]

# Run diagnosis
headers = {"Authorization": f"Bearer {token}"}
diagnosis = requests.post(
    f"{BASE_URL}/diagnosis/analyze",
    headers=headers,
    json={
        "vehicle_make": "Volkswagen",
        "vehicle_model": "Golf",
        "vehicle_year": 2018,
        "dtc_codes": ["P0101"],
        "symptoms": "Motor nehezen indul"
    }
)
print(diagnosis.json())
```

### JavaScript/TypeScript Example

```javascript
const BASE_URL = "http://localhost:8000/api/v1";

// Login
const loginResponse = await fetch(`${BASE_URL}/auth/login`, {
  method: "POST",
  headers: { "Content-Type": "application/x-www-form-urlencoded" },
  body: "username=user@example.com&password=SecurePass123!"
});
const { access_token } = await loginResponse.json();

// Run diagnosis
const diagnosisResponse = await fetch(`${BASE_URL}/diagnosis/analyze`, {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${access_token}`,
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    vehicle_make: "Volkswagen",
    vehicle_model: "Golf",
    vehicle_year: 2018,
    dtc_codes: ["P0101"],
    symptoms: "Motor nehezen indul"
  })
});
const diagnosis = await diagnosisResponse.json();
console.log(diagnosis);
```

---

## Support

For API support:
- OpenAPI Documentation: `/api/v1/docs`
- ReDoc Documentation: `/api/v1/redoc`
- Project Repository: [AutoCognitix](https://github.com/autocognitix)
- Email: support@autocognitix.hu
