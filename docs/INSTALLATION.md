# Installation Guide

This guide covers the complete installation process for AutoCognitix, including Docker-based development setup and manual installation.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start with Docker](#quick-start-with-docker)
- [Manual Installation](#manual-installation)
- [Environment Configuration](#environment-configuration)
- [Database Initialization](#database-initialization)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software
- **Docker** 24.0+ and **Docker Compose** 2.20+
- **Python** 3.11+ (for manual installation)
- **Node.js** 18+ and **npm** 9+ (for manual installation)
- **Git**

### Hardware Requirements
- **Minimum**: 4GB RAM, 2 CPU cores, 20GB disk space
- **Recommended**: 8GB+ RAM, 4+ CPU cores, 50GB disk space
- **Note**: The huBERT embedding model requires approximately 500MB memory

## Quick Start with Docker

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/autocognitix.git
cd autocognitix
```

### 2. Configure Environment Variables
```bash
# Copy the example environment file
cp .env.example .env

# Generate secure secrets
echo "SECRET_KEY=$(openssl rand -hex 32)" >> .env
echo "JWT_SECRET_KEY=$(openssl rand -hex 32)" >> .env
```

Edit `.env` and configure the required variables (see [Environment Configuration](#environment-configuration)).

### 3. Start Services
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

### 4. Initialize Databases
```bash
# Run database migrations
docker-compose exec backend alembic upgrade head

# Seed initial data (optional)
docker-compose exec backend python -m scripts.seed_database
```

### 5. Access the Application
| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |
| Neo4j Browser | http://localhost:7474 |
| Qdrant Dashboard | http://localhost:6333/dashboard |

## Manual Installation

### Backend Setup

#### 1. Create Python Virtual Environment
```bash
cd backend
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

#### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Install Optional Dependencies
```bash
# For Hungarian NLP (requires additional setup)
pip install huspacy
python -m spacy download hu_core_news_lg
```

#### 4. Configure Environment
```bash
cd ..
cp .env.example .env
# Edit .env with your configuration
```

#### 5. Start Backend Server
```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Setup

#### 1. Install Dependencies
```bash
cd frontend
npm install
```

#### 2. Configure Environment
```bash
# Create frontend environment file
echo "VITE_API_URL=http://localhost:8000" > .env.local
```

#### 3. Start Development Server
```bash
npm run dev
```

The frontend will be available at http://localhost:5173 (Vite default).

### Database Setup

#### PostgreSQL
```bash
# Create database
createdb autocognitix

# Or using Docker
docker run -d \
  --name autocognitix-postgres \
  -e POSTGRES_USER=autocognitix \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_DB=autocognitix \
  -p 5432:5432 \
  postgres:16-alpine
```

#### Neo4j
```bash
# Using Docker
docker run -d \
  --name autocognitix-neo4j \
  -e NEO4J_AUTH=neo4j/your_password \
  -e NEO4J_PLUGINS='["apoc"]' \
  -p 7474:7474 \
  -p 7687:7687 \
  neo4j:5.15-community
```

#### Qdrant
```bash
# Using Docker
docker run -d \
  --name autocognitix-qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  qdrant/qdrant:v1.7.4
```

#### Redis
```bash
# Using Docker
docker run -d \
  --name autocognitix-redis \
  -p 6379:6379 \
  redis:7-alpine
```

## Environment Configuration

### Required Variables

```bash
# ============================================
# Database Configuration
# ============================================
# PostgreSQL
DATABASE_URL=postgresql+asyncpg://autocognitix:password@localhost:5432/autocognitix

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Redis
REDIS_URL=redis://localhost:6379/0

# ============================================
# Security
# ============================================
# Generate with: openssl rand -hex 32
SECRET_KEY=your_secret_key_here
JWT_SECRET_KEY=your_jwt_secret_key_here

# ============================================
# LLM Configuration (choose one)
# ============================================
# Anthropic (recommended)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key

# Or OpenAI
# LLM_PROVIDER=openai
# OPENAI_API_KEY=your_openai_api_key
```

### Optional Variables

```bash
# ============================================
# Application Settings
# ============================================
DEBUG=true
ENVIRONMENT=development
LOG_LEVEL=INFO

# ============================================
# Hungarian NLP
# ============================================
HUBERT_MODEL=SZTAKI-HLT/hubert-base-cc
EMBEDDING_DIMENSION=768

# ============================================
# Rate Limiting
# ============================================
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000

# ============================================
# External APIs (optional)
# ============================================
# YouTube API for video indexing
YOUTUBE_API_KEY=your_youtube_api_key

# CarMD API (paid)
CARMD_API_KEY=your_carmd_api_key
CARMD_PARTNER_TOKEN=your_partner_token
```

## Database Initialization

### Run Migrations
```bash
# With Docker
docker-compose exec backend alembic upgrade head

# Without Docker
cd backend
alembic upgrade head
```

### Seed Initial Data
```bash
# Seed DTC codes and Neo4j graph
docker-compose exec backend python -m scripts.seed_database
```

### Create Qdrant Collection
The Qdrant collection is automatically created when the application starts. If you need to manually create it:

```bash
docker-compose exec backend python -c "
from app.db.qdrant_client import qdrant_client
import asyncio
asyncio.run(qdrant_client.initialize())
"
```

## Troubleshooting

### Common Issues

#### Docker Permission Denied
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in
```

#### PostgreSQL Connection Failed
```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# Check logs
docker-compose logs postgres

# Verify connection string in .env
```

#### Neo4j Authentication Error
```bash
# Reset Neo4j password
docker-compose exec neo4j neo4j-admin dbms set-initial-password new_password

# Or remove data and restart
docker-compose down
docker volume rm autocognitix_neo4j_data
docker-compose up -d neo4j
```

#### Qdrant Collection Issues
```bash
# Delete and recreate collection
curl -X DELETE http://localhost:6333/collections/dtc_codes
docker-compose restart backend
```

#### Memory Issues with huBERT
If you encounter memory issues with the embedding model:
```bash
# Reduce batch size in environment
echo "EMBEDDING_BATCH_SIZE=8" >> .env
docker-compose restart backend
```

### Verify Installation

#### Check Backend Health
```bash
curl http://localhost:8000/api/v1/health
# Expected: {"status": "healthy", ...}
```

#### Check Database Connections
```bash
curl http://localhost:8000/api/v1/health/detailed
```

#### Run Tests
```bash
# Backend tests
docker-compose exec backend pytest

# Frontend tests
cd frontend && npm test
```

## Next Steps

- [API Guide](./API_GUIDE.md) - Learn how to use the API
- [User Manual](./USER_MANUAL_HU.md) - Platform usage guide
- [Development Guide](./DEVELOPMENT.md) - Contribute to the project
