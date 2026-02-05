# AutoCognitix

**AI-Powered Vehicle Diagnostic Platform with Hungarian Language Support**

[![CI](https://github.com/norbertbarna/AutoCognitix/actions/workflows/ci.yml/badge.svg)](https://github.com/norbertbarna/AutoCognitix/actions/workflows/ci.yml)
[![CD](https://github.com/norbertbarna/AutoCognitix/actions/workflows/cd.yml/badge.svg)](https://github.com/norbertbarna/AutoCognitix/actions/workflows/cd.yml)
[![Security](https://github.com/norbertbarna/AutoCognitix/actions/workflows/security.yml/badge.svg)](https://github.com/norbertbarna/AutoCognitix/actions/workflows/security.yml)
[![codecov](https://codecov.io/gh/norbertbarna/AutoCognitix/branch/main/graph/badge.svg)](https://codecov.io/gh/norbertbarna/AutoCognitix)
[![License](https://img.shields.io/badge/license-proprietary-red.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Node.js](https://img.shields.io/badge/node.js-20+-green.svg)](https://nodejs.org/)

---

AutoCognitix is an intelligent vehicle diagnostic platform that combines AI-powered analysis with comprehensive DTC (Diagnostic Trouble Code) databases to help mechanics and car owners diagnose vehicle issues. The platform features full Hungarian language support and manual DTC code/symptom input without requiring diagnostic hardware.

## Key Features

- **AI-Powered Diagnosis** - Uses RAG (Retrieval-Augmented Generation) with LangChain for intelligent problem analysis
- **DTC Code Lookup** - Comprehensive database of OBD-II diagnostic trouble codes with Hungarian descriptions
- **Symptom Analysis** - Natural language processing for Hungarian symptom descriptions using huBERT embeddings
- **Knowledge Graph** - Neo4j-based diagnostic paths connecting DTCs, symptoms, components, and repairs
- **Vector Search** - Semantic similarity search using Qdrant for finding related issues
- **VIN Decoding** - Decode Vehicle Identification Numbers using NHTSA API
- **Recall Information** - Access to NHTSA recall database and historical complaint data

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Backend** | FastAPI, SQLAlchemy 2.0, PostgreSQL 16, Neo4j 5.x, Qdrant, Redis 7 |
| **Frontend** | React 18, TypeScript, TailwindCSS, TanStack Query, Vite |
| **AI/NLP** | LangChain, huBERT (SZTAKI-HLT), Anthropic Claude / OpenAI GPT-4 |
| **Infrastructure** | Docker, Railway, GitHub Actions |

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- Node.js 20+

### 1. Clone and Configure
```bash
git clone https://github.com/norbertbarna/AutoCognitix.git
cd AutoCognitix
cp .env.example .env
# Edit .env with your configuration
```

### 2. Start Services
```bash
docker-compose up -d
```

### 3. Access the Application
| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |
| Neo4j Browser | http://localhost:7474 |

## Project Structure

```
AutoCognitix/
├── backend/           # FastAPI application
│   ├── app/
│   │   ├── api/v1/   # API endpoints & schemas
│   │   ├── core/     # Config, security, logging
│   │   ├── db/       # PostgreSQL, Neo4j, Qdrant, Redis
│   │   └── services/ # Business logic
│   ├── alembic/      # Database migrations
│   └── tests/        # Test suite
├── frontend/          # React application
│   └── src/
│       ├── pages/    # Page components
│       ├── components/
│       └── services/ # API client
├── data/             # DTC codes data
├── scripts/          # Utility scripts
├── docs/             # Documentation
└── .github/          # CI/CD workflows
    └── workflows/
        ├── ci.yml        # Continuous Integration
        ├── cd.yml        # Continuous Deployment
        └── security.yml  # Security Scanning
```

## CI/CD Pipeline

### Continuous Integration (CI)
Runs on every push and PR to `main` and `develop`:
- **Backend**: Ruff linting, MyPy type checking, pytest with coverage
- **Frontend**: ESLint, TypeScript, Vite build, Vitest
- **Security**: Bandit (Python), npm audit
- **Docker**: Build verification for both services

### Continuous Deployment (CD)
Triggers on releases and tags:
- Multi-platform Docker image builds (amd64/arm64)
- Push to GitHub Container Registry (GHCR)
- Deploy to Railway
- Database migrations
- Smoke tests with health checks
- Automatic rollback on failure

### Security Scanning
Daily scheduled + on changes:
- CodeQL SAST analysis (Python, TypeScript)
- Dependency vulnerability checks (Safety, pip-audit, npm audit)
- Secret scanning (Gitleaks)
- Container scanning (Trivy)
- License compliance checks

### Dependabot
Automatic dependency updates:
- Python packages (weekly)
- NPM packages (weekly)
- GitHub Actions (weekly)
- Docker base images (weekly)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/auth/register` | POST | Register new user |
| `/api/v1/auth/login` | POST | User login |
| `/api/v1/diagnosis/analyze` | POST | Main diagnostic analysis |
| `/api/v1/dtc/search` | GET | Search DTC codes |
| `/api/v1/dtc/{code}` | GET | Get DTC details |
| `/api/v1/vehicles/decode-vin` | POST | Decode VIN |
| `/health` | GET | Health check |

## Documentation

- [Installation Guide](./docs/INSTALLATION.md)
- [API Reference](./docs/API_REFERENCE.md)
- [Development Guide](./docs/DEVELOPMENT.md)
- [Deployment Guide](./docs/DEPLOYMENT.md)
- [User Manual (Hungarian)](./docs/USER_MANUAL_HU.md)

## Development

### Running Tests
```bash
# Backend tests
cd backend && pytest tests -v --cov=app

# Frontend tests
cd frontend && npm run test
```

### Code Quality
```bash
# Backend linting
cd backend && ruff check app tests
cd backend && ruff format app tests

# Frontend linting
cd frontend && npm run lint
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `NEO4J_URI` | Neo4j connection URI |
| `NEO4J_PASSWORD` | Neo4j password |
| `QDRANT_URL` | Qdrant server URL |
| `QDRANT_API_KEY` | Qdrant API key |
| `REDIS_URL` | Redis connection string |
| `JWT_SECRET_KEY` | JWT signing secret |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key |

## License

This project is proprietary software. All rights reserved.

## Support

For questions or support, please contact the development team.
