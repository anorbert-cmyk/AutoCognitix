# AutoCognitix

**AI-Powered Vehicle Diagnostic Platform with Hungarian Language Support**

AutoCognitix is an intelligent vehicle diagnostic platform that combines AI-powered analysis with comprehensive DTC (Diagnostic Trouble Code) databases to help mechanics and car owners diagnose vehicle issues. The platform features full Hungarian language support and manual DTC code/symptom input without requiring diagnostic hardware.

## Key Features

### Diagnostic Capabilities
- **AI-Powered Diagnosis**: Uses RAG (Retrieval-Augmented Generation) with LangChain for intelligent problem analysis
- **DTC Code Lookup**: Comprehensive database of OBD-II diagnostic trouble codes with Hungarian descriptions
- **Symptom Analysis**: Natural language processing for Hungarian symptom descriptions using huBERT embeddings
- **Knowledge Graph**: Neo4j-based diagnostic paths connecting DTCs, symptoms, components, and repairs
- **Vector Search**: Semantic similarity search using Qdrant for finding related issues

### Vehicle Information
- **VIN Decoding**: Decode Vehicle Identification Numbers using NHTSA API
- **Recall Information**: Access to NHTSA recall database
- **Complaint Data**: Historical complaint information from NHTSA

### User Features
- **User Authentication**: Secure JWT-based authentication system
- **Diagnosis History**: Save and review past diagnoses
- **Statistics Dashboard**: Track diagnostic patterns and most common issues

## Tech Stack

### Backend
| Component | Technology |
|-----------|------------|
| Framework | FastAPI + Pydantic V2 |
| ORM | SQLAlchemy 2.0 (async) |
| Primary Database | PostgreSQL 16 |
| Graph Database | Neo4j 5.x |
| Vector Database | Qdrant |
| Cache | Redis 7 |
| Authentication | JWT (python-jose) |

### Frontend
| Component | Technology |
|-----------|------------|
| Framework | React 18 + TypeScript |
| Styling | TailwindCSS |
| State Management | TanStack Query |
| Build Tool | Vite |

### AI/NLP
| Component | Technology |
|-----------|------------|
| RAG Pipeline | LangChain |
| Hungarian NLP | huBERT (SZTAKI-HLT/hubert-base-cc) |
| Embeddings | 768-dimensional vectors |
| LLM Provider | Anthropic Claude / OpenAI GPT-4 |

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- Node.js 18+
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/autocognitix.git
cd autocognitix
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Start with Docker
```bash
docker-compose up -d
```

### 4. Access the Application
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Neo4j Browser: http://localhost:7474

## Project Structure

```
AutoCognitix/
├── backend/                 # FastAPI application
│   ├── app/
│   │   ├── api/v1/         # API endpoints
│   │   │   ├── endpoints/  # Route handlers
│   │   │   └── schemas/    # Pydantic models
│   │   ├── core/           # Config, security, logging
│   │   ├── db/             # Database connections
│   │   │   ├── postgres/   # PostgreSQL models & repositories
│   │   │   ├── neo4j_models.py
│   │   │   ├── qdrant_client.py
│   │   │   └── redis_cache.py
│   │   ├── services/       # Business logic
│   │   │   ├── diagnosis_service.py
│   │   │   ├── embedding_service.py
│   │   │   ├── rag_service.py
│   │   │   └── nhtsa_service.py
│   │   └── prompts/        # LLM prompt templates
│   ├── alembic/            # Database migrations
│   └── tests/              # Test suite
├── frontend/               # React application
│   └── src/
│       ├── pages/          # Page components
│       ├── components/     # Reusable components
│       ├── services/       # API client
│       └── contexts/       # React contexts
├── data/                   # Data files (DTC codes)
├── scripts/                # Utility scripts
├── docs/                   # Documentation
└── docker-compose.yml      # Docker services
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/auth/register` | POST | Register new user |
| `/api/v1/auth/login` | POST | User login |
| `/api/v1/auth/me` | GET | Get current user |
| `/api/v1/diagnosis/analyze` | POST | Main diagnostic analysis |
| `/api/v1/diagnosis/quick-analyze` | POST | Quick DTC lookup |
| `/api/v1/diagnosis/history/list` | GET | Get diagnosis history |
| `/api/v1/dtc/search` | GET | Search DTC codes |
| `/api/v1/dtc/{code}` | GET | Get DTC details |
| `/api/v1/vehicles/decode-vin` | POST | Decode VIN |
| `/api/v1/vehicles/{make}/{model}/{year}/recalls` | GET | Get recalls |

## Documentation

- [Installation Guide](./INSTALLATION.md) - Detailed setup instructions
- [API Guide](./API_GUIDE.md) - API usage with examples
- [User Manual (Hungarian)](./USER_MANUAL_HU.md) - Platform hasznalati utmutato
- [Development Guide](./DEVELOPMENT.md) - For contributors
- [Deployment Guide](./DEPLOYMENT.md) - Production deployment

## License

This project is proprietary software. All rights reserved.

## Support

For questions or support, please contact the development team.
