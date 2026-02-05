# API Guide

This guide covers the AutoCognitix REST API endpoints, authentication, and provides practical examples using curl.

## Table of Contents
- [Base URL](#base-url)
- [Authentication](#authentication)
- [Error Handling](#error-handling)
- [Endpoints](#endpoints)
  - [Authentication](#authentication-endpoints)
  - [Diagnosis](#diagnosis-endpoints)
  - [DTC Codes](#dtc-code-endpoints)
  - [Vehicles](#vehicle-endpoints)
  - [Health](#health-endpoints)

## Base URL

```
Development: http://localhost:8000/api/v1
Production:  https://your-domain.com/api/v1
```

## Authentication

AutoCognitix uses JWT (JSON Web Tokens) for authentication. Most endpoints require authentication.

### Token Types
- **Access Token**: Short-lived (30 minutes), used for API requests
- **Refresh Token**: Long-lived (7 days), used to obtain new access tokens

### Using Tokens
Include the access token in the Authorization header:
```bash
Authorization: Bearer <access_token>
```

## Error Handling

### Error Response Format
```json
{
  "detail": "Error message describing what went wrong"
}
```

### HTTP Status Codes
| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request - Invalid input |
| 401 | Unauthorized - Invalid/expired token |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found |
| 422 | Validation Error |
| 423 | Locked - Account locked |
| 500 | Internal Server Error |
| 502 | Bad Gateway - External API error |

---

## Endpoints

### Authentication Endpoints

#### Register New User
```bash
POST /api/v1/auth/register
```

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "SecurePass123!",
  "full_name": "Kovacs Janos"
}
```

**Password Requirements:**
- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one number
- At least one special character

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123!",
    "full_name": "Kovacs Janos"
  }'
```

**Response (201):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "full_name": "Kovacs Janos",
  "is_active": true,
  "role": "user",
  "created_at": "2024-02-01T10:30:00Z"
}
```

---

#### Login
```bash
POST /api/v1/auth/login
```

**Request:** (form-urlencoded)
```
username=user@example.com&password=SecurePass123!
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=SecurePass123!"
```

**Response (200):**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer"
}
```

---

#### Get Current User
```bash
GET /api/v1/auth/me
```

**Example:**
```bash
curl http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer <access_token>"
```

**Response (200):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "full_name": "Kovacs Janos",
  "is_active": true,
  "role": "user",
  "created_at": "2024-02-01T10:30:00Z"
}
```

---

#### Refresh Tokens
```bash
POST /api/v1/auth/refresh
```

**Request Body:**
```json
{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "your_refresh_token_here"}'
```

---

#### Logout
```bash
POST /api/v1/auth/logout
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/auth/logout \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "your_refresh_token_here"}'
```

---

### Diagnosis Endpoints

#### Analyze Vehicle (Main Diagnostic Endpoint)
```bash
POST /api/v1/diagnosis/analyze
```

This is the primary diagnostic endpoint that uses AI to analyze vehicle issues.

**Request Body:**
```json
{
  "vehicle_make": "Volkswagen",
  "vehicle_model": "Golf",
  "vehicle_year": 2018,
  "vehicle_engine": "2.0 TSI",
  "vin": "WVWZZZ3CZWE123456",
  "dtc_codes": ["P0101", "P0171"],
  "symptoms": "A motor nehezen indul hidegben, egyenetlenul jar alapjaraton, es a fogyasztas megott.",
  "additional_context": "A problema telen rosszabb."
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/diagnosis/analyze \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <access_token>" \
  -d '{
    "vehicle_make": "Volkswagen",
    "vehicle_model": "Golf",
    "vehicle_year": 2018,
    "dtc_codes": ["P0101", "P0171"],
    "symptoms": "A motor nehezen indul hidegben, egyenetlenul jar alapjaraton."
  }'
```

**Response (201):**
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
      "description": "A levegotomeg-mero szenzor hibas vagy szennyezett.",
      "confidence": 0.85,
      "related_dtc_codes": ["P0101"],
      "components": ["MAF szenzor", "Levegoszuro"]
    }
  ],
  "recommended_repairs": [
    {
      "title": "MAF szenzor tisztitasa/csereje",
      "description": "Ellenorizze es tisztitsa meg a MAF szenzort specialis tisztitoval.",
      "estimated_cost_min": 5000,
      "estimated_cost_max": 45000,
      "estimated_cost_currency": "HUF",
      "difficulty": "intermediate",
      "parts_needed": ["MAF szenzor tisztito"],
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

---

#### Quick DTC Analysis
```bash
POST /api/v1/diagnosis/quick-analyze
```

Fast DTC code lookup without full AI analysis.

**Query Parameters:**
- `dtc_codes` (required): Array of DTC codes (1-10)

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/diagnosis/quick-analyze?dtc_codes=P0101&dtc_codes=P0171"
```

**Response (200):**
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

---

#### Get Diagnosis by ID
```bash
GET /api/v1/diagnosis/{diagnosis_id}
```

**Example:**
```bash
curl http://localhost:8000/api/v1/diagnosis/550e8400-e29b-41d4-a716-446655440000
```

---

#### Get Diagnosis History
```bash
GET /api/v1/diagnosis/history/list
```

**Query Parameters:**
- `skip` (optional): Number of records to skip (default: 0)
- `limit` (optional): Maximum records (1-100, default: 10)
- `vehicle_make` (optional): Filter by make
- `vehicle_model` (optional): Filter by model
- `vehicle_year` (optional): Filter by year
- `dtc_code` (optional): Filter by DTC code
- `date_from` (optional): Start date (ISO format)
- `date_to` (optional): End date (ISO format)

**Example:**
```bash
curl "http://localhost:8000/api/v1/diagnosis/history/list?limit=10&vehicle_make=Volkswagen" \
  -H "Authorization: Bearer <access_token>"
```

**Response (200):**
```json
{
  "items": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "vehicle_make": "Volkswagen",
      "vehicle_model": "Golf",
      "vehicle_year": 2018,
      "dtc_codes": ["P0101", "P0171"],
      "symptoms_text": "Motor nehezen indul",
      "confidence_score": 0.85,
      "created_at": "2024-02-01T10:30:00Z"
    }
  ],
  "total": 25,
  "skip": 0,
  "limit": 10,
  "has_more": true
}
```

---

#### Get Diagnosis Statistics
```bash
GET /api/v1/diagnosis/stats/summary
```

**Example:**
```bash
curl http://localhost:8000/api/v1/diagnosis/stats/summary \
  -H "Authorization: Bearer <access_token>"
```

**Response (200):**
```json
{
  "total_diagnoses": 42,
  "avg_confidence": 0.75,
  "most_diagnosed_vehicles": [
    {"make": "Volkswagen", "model": "Golf", "count": 15},
    {"make": "BMW", "model": "3 Series", "count": 8}
  ],
  "most_common_dtcs": [
    {"code": "P0171", "count": 12},
    {"code": "P0300", "count": 8}
  ],
  "diagnoses_by_month": [
    {"month": "2024-02", "count": 8},
    {"month": "2024-01", "count": 12}
  ]
}
```

---

#### Delete Diagnosis
```bash
DELETE /api/v1/diagnosis/{diagnosis_id}
```

**Example:**
```bash
curl -X DELETE http://localhost:8000/api/v1/diagnosis/550e8400-e29b-41d4-a716-446655440000 \
  -H "Authorization: Bearer <access_token>"
```

---

### DTC Code Endpoints

#### Search DTC Codes
```bash
GET /api/v1/dtc/search
```

**Query Parameters:**
- `q` (required): Search query (code or description)
- `category` (optional): Filter by category (powertrain, body, chassis, network)
- `make` (optional): Filter by vehicle make
- `limit` (optional): Maximum results (1-100, default: 20)
- `use_semantic` (optional): Use semantic search (default: true)

**Example:**
```bash
curl "http://localhost:8000/api/v1/dtc/search?q=MAF+szenzor&limit=10"
```

**Response (200):**
```json
[
  {
    "code": "P0101",
    "description_en": "Mass Air Flow Circuit Range/Performance",
    "description_hu": "Levegotomeg-mero aramkor tartomany/teljesitmeny hiba",
    "category": "powertrain",
    "is_generic": true,
    "severity": "medium",
    "relevance_score": 0.95
  }
]
```

---

#### Get DTC Details
```bash
GET /api/v1/dtc/{code}
```

**Query Parameters:**
- `include_graph` (optional): Include Neo4j relationships (default: true)

**Example:**
```bash
curl http://localhost:8000/api/v1/dtc/P0101
```

**Response (200):**
```json
{
  "code": "P0101",
  "description_en": "Mass Air Flow Circuit Range/Performance",
  "description_hu": "Levegotomeg-mero aramkor tartomany/teljesitmeny hiba",
  "category": "powertrain",
  "is_generic": true,
  "severity": "medium",
  "system": "Fuel and Air Metering",
  "symptoms": [
    "Motor teljesitmenyvesztese",
    "Egyenetlen alapjarat",
    "Megnovekedett uzemanyag-fogyasztas"
  ],
  "possible_causes": [
    "Szennyezett MAF szenzor",
    "Levegoszuro eltomodes",
    "Levegobeszivas szivas"
  ],
  "diagnostic_steps": [
    "Ellenorizze a MAF szenzor csatlakozasait",
    "Tisztitsa meg a MAF szenzort specialis tisztitoval",
    "Ellenorizze a levegoszurot"
  ],
  "related_codes": ["P0100", "P0102", "P0103", "P0171"],
  "manufacturer_code": null
}
```

---

#### Get Related DTC Codes
```bash
GET /api/v1/dtc/{code}/related
```

**Example:**
```bash
curl http://localhost:8000/api/v1/dtc/P0101/related?limit=5
```

---

#### Get DTC Categories
```bash
GET /api/v1/dtc/categories/list
```

**Example:**
```bash
curl http://localhost:8000/api/v1/dtc/categories/list
```

**Response (200):**
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
  }
]
```

---

### Vehicle Endpoints

#### Decode VIN
```bash
POST /api/v1/vehicles/decode-vin
```

**Request Body:**
```json
{
  "vin": "WVWZZZ3CZWE123456"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/vehicles/decode-vin \
  -H "Content-Type: application/json" \
  -d '{"vin": "WVWZZZ3CZWE123456"}'
```

**Response (200):**
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

---

#### Get Vehicle Makes
```bash
GET /api/v1/vehicles/makes
```

**Query Parameters:**
- `search` (optional): Filter makes by name

**Example:**
```bash
curl "http://localhost:8000/api/v1/vehicles/makes?search=volk"
```

---

#### Get Vehicle Models
```bash
GET /api/v1/vehicles/models/{make_id}
```

**Query Parameters:**
- `year` (optional): Filter models by year

**Example:**
```bash
curl "http://localhost:8000/api/v1/vehicles/models/volkswagen?year=2020"
```

---

#### Get Recalls
```bash
GET /api/v1/vehicles/{make}/{model}/{year}/recalls
```

**Example:**
```bash
curl http://localhost:8000/api/v1/vehicles/Toyota/Camry/2020/recalls
```

---

#### Get Complaints
```bash
GET /api/v1/vehicles/{make}/{model}/{year}/complaints
```

**Example:**
```bash
curl http://localhost:8000/api/v1/vehicles/Toyota/Camry/2020/complaints
```

---

### Health Endpoints

#### Basic Health Check
```bash
GET /api/v1/health
```

**Example:**
```bash
curl http://localhost:8000/api/v1/health
```

**Response (200):**
```json
{
  "status": "healthy",
  "timestamp": "2024-02-03T10:30:00Z"
}
```

---

## Rate Limiting

The API implements rate limiting to prevent abuse:
- **60 requests per minute** per IP
- **1000 requests per hour** per IP

When rate limited, you will receive a `429 Too Many Requests` response.

## Pagination

Endpoints returning lists support pagination:
- `skip`: Number of items to skip (default: 0)
- `limit`: Maximum items per page (varies by endpoint)

Response includes:
- `total`: Total number of items
- `has_more`: Boolean indicating if more items exist

## SDK Examples

### Python
```python
import httpx

BASE_URL = "http://localhost:8000/api/v1"

async def login(email: str, password: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/auth/login",
            data={"username": email, "password": password}
        )
        return response.json()

async def diagnose_vehicle(token: str, data: dict) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/diagnosis/analyze",
            json=data,
            headers={"Authorization": f"Bearer {token}"}
        )
        return response.json()
```

### JavaScript/TypeScript
```typescript
const BASE_URL = 'http://localhost:8000/api/v1';

async function login(email: string, password: string) {
  const response = await fetch(`${BASE_URL}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ username: email, password })
  });
  return response.json();
}

async function diagnoseVehicle(token: string, data: object) {
  const response = await fetch(`${BASE_URL}/diagnosis/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify(data)
  });
  return response.json();
}
```

## Interactive Documentation

The API provides interactive documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
