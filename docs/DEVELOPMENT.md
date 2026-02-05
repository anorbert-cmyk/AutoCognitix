# Development Guide

This guide is for developers who want to contribute to AutoCognitix or extend its functionality.

## Table of Contents
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Code Style](#code-style)
- [Adding New DTC Codes](#adding-new-dtc-codes)
- [Extending the RAG Pipeline](#extending-the-rag-pipeline)
- [Database Operations](#database-operations)
- [Testing](#testing)
- [Contributing](#contributing)

---

## Development Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker and Docker Compose
- Git

### Initial Setup

```bash
# Clone repository
git clone https://github.com/your-org/autocognitix.git
cd autocognitix

# Copy environment file
cp .env.example .env
# Edit .env with your API keys and configuration

# Start database services
docker-compose up -d postgres neo4j qdrant redis

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt

# Run migrations
alembic upgrade head

# Seed database
python -m scripts.seed_database

# Start backend
uvicorn app.main:app --reload

# Frontend setup (new terminal)
cd frontend
npm install
npm run dev
```

### IDE Configuration

#### VS Code Extensions
- Python
- Pylance
- ESLint
- Prettier
- Tailwind CSS IntelliSense

#### Recommended settings.json
```json
{
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter"
  },
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  }
}
```

---

## Project Structure

```
AutoCognitix/
├── backend/
│   ├── app/
│   │   ├── api/v1/
│   │   │   ├── endpoints/      # Route handlers
│   │   │   │   ├── auth.py
│   │   │   │   ├── diagnosis.py
│   │   │   │   ├── dtc_codes.py
│   │   │   │   ├── vehicles.py
│   │   │   │   └── health.py
│   │   │   ├── schemas/        # Pydantic models
│   │   │   └── router.py       # API router aggregation
│   │   ├── core/               # Core functionality
│   │   │   ├── config.py       # Settings management
│   │   │   ├── security.py     # JWT, password hashing
│   │   │   ├── logging.py      # Structured logging
│   │   │   └── exceptions.py   # Custom exceptions
│   │   ├── db/                 # Database layer
│   │   │   ├── postgres/
│   │   │   │   ├── models.py   # SQLAlchemy models
│   │   │   │   ├── repositories.py
│   │   │   │   └── session.py
│   │   │   ├── neo4j_models.py
│   │   │   ├── qdrant_client.py
│   │   │   └── redis_cache.py
│   │   ├── services/           # Business logic
│   │   │   ├── diagnosis_service.py
│   │   │   ├── embedding_service.py
│   │   │   ├── rag_service.py
│   │   │   ├── llm_provider.py
│   │   │   └── nhtsa_service.py
│   │   ├── prompts/            # LLM prompts
│   │   │   └── diagnosis_hu.py
│   │   └── main.py             # FastAPI application
│   ├── alembic/                # Database migrations
│   │   └── versions/
│   └── tests/
│       ├── integration/
│       └── unit/
├── frontend/
│   └── src/
│       ├── pages/
│       ├── components/
│       ├── services/           # API client
│       └── contexts/
├── data/                       # Data files
│   └── dtc_codes/
├── scripts/                    # Utility scripts
└── docs/                       # Documentation
```

### Key Files

| File | Purpose |
|------|---------|
| `backend/app/core/config.py` | Environment configuration |
| `backend/app/api/v1/schemas/diagnosis.py` | Main API contracts |
| `backend/app/services/diagnosis_service.py` | Diagnosis business logic |
| `backend/app/services/rag_service.py` | RAG pipeline |
| `backend/app/db/neo4j_models.py` | Graph schema |

---

## Code Style

### Python

We use Black for formatting and Flake8 for linting.

```bash
# Format code
black backend/

# Check linting
flake8 backend/

# Type checking
mypy backend/
```

### TypeScript

We use ESLint and Prettier.

```bash
# Lint
cd frontend && npm run lint

# Format
npm run format
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## Adding New DTC Codes

### Method 1: Direct Database Import

```python
# scripts/import_dtc_codes.py
import asyncio
from app.db.postgres.session import async_session_factory
from app.db.postgres.repositories import DTCCodeRepository

async def import_codes():
    async with async_session_factory() as session:
        repo = DTCCodeRepository(session)

        await repo.create({
            "code": "P0XXX",
            "description_en": "English description",
            "description_hu": "Magyar leiras",
            "category": "powertrain",
            "severity": "medium",
            "is_generic": True,
            "system": "Engine",
            "symptoms": ["Symptom 1", "Symptom 2"],
            "possible_causes": ["Cause 1", "Cause 2"],
            "diagnostic_steps": ["Step 1", "Step 2"],
            "related_codes": ["P0001", "P0002"],
        })

        await session.commit()

asyncio.run(import_codes())
```

### Method 2: JSON Import

Create a JSON file in `data/dtc_codes/`:

```json
{
  "codes": [
    {
      "code": "P0XXX",
      "description_en": "English description",
      "description_hu": "Magyar leiras",
      "category": "powertrain",
      "severity": "medium",
      "is_generic": true,
      "symptoms": ["Symptom 1"],
      "possible_causes": ["Cause 1"],
      "diagnostic_steps": ["Step 1"]
    }
  ]
}
```

Run import script:
```bash
python scripts/import_dtc_codes.py data/dtc_codes/your_file.json
```

### Method 3: API Bulk Import

```bash
curl -X POST http://localhost:8000/api/v1/dtc/bulk \
  -H "Content-Type: application/json" \
  -d '{
    "codes": [...],
    "overwrite_existing": false
  }'
```

### Adding Neo4j Relationships

```python
# Add to Neo4j graph
from app.db.neo4j_models import DTCNode, SymptomNode, ComponentNode

async def add_graph_relationships():
    # Create DTC node
    dtc = DTCNode(
        code="P0XXX",
        description="Description",
        category="powertrain"
    ).save()

    # Create symptom and link
    symptom = SymptomNode(
        name="Engine misfire",
        description_hu="Motor kihagyás"
    ).save()
    dtc.symptoms.connect(symptom)

    # Create component and link
    component = ComponentNode(
        name="Spark Plug",
        failure_mode="Worn electrode"
    ).save()
    dtc.components.connect(component)
```

### Generating Embeddings

After adding codes, generate embeddings for semantic search:

```python
from app.services.embedding_service import get_embedding_service
from app.db.qdrant_client import qdrant_client

async def index_new_codes(codes: list):
    embedding_service = get_embedding_service()

    for code_data in codes:
        # Combine text for embedding
        text = f"{code_data['description_hu']} {' '.join(code_data.get('symptoms', []))}"

        # Generate embedding
        embedding = embedding_service.embed_text(text, preprocess=True)

        # Index in Qdrant
        await qdrant_client.upsert_dtc(
            code=code_data['code'],
            vector=embedding,
            payload={
                "code": code_data['code'],
                "description_hu": code_data['description_hu'],
                "category": code_data['category'],
            }
        )
```

---

## Extending the RAG Pipeline

### RAG Architecture

```
User Query
    │
    ▼
┌──────────────┐
│ Preprocessing │ ─── Hungarian NLP (HuSpaCy)
└──────────────┘
    │
    ▼
┌──────────────┐
│  Embedding   │ ─── huBERT (768-dim)
└──────────────┘
    │
    ▼
┌──────────────┐
│ Vector Search│ ─── Qdrant
└──────────────┘
    │
    ▼
┌──────────────┐
│ Graph Query  │ ─── Neo4j (DTC → Symptom → Component → Repair)
└──────────────┘
    │
    ▼
┌──────────────┐
│ Context Build│ ─── Merge results
└──────────────┘
    │
    ▼
┌──────────────┐
│ LLM Generate │ ─── Claude/GPT-4
└──────────────┘
    │
    ▼
Diagnosis Response
```

### Adding New Context Sources

Edit `backend/app/services/rag_service.py`:

```python
class RAGService:
    async def get_context(self, query: str, dtc_codes: list) -> dict:
        # Existing sources
        vector_results = await self._search_vectors(query)
        graph_results = await self._query_graph(dtc_codes)

        # Add new source
        forum_results = await self._search_forums(query)

        return {
            "vector_results": vector_results,
            "graph_results": graph_results,
            "forum_results": forum_results,  # New
        }

    async def _search_forums(self, query: str) -> list:
        """Search community forums for relevant discussions."""
        # Implementation here
        pass
```

### Customizing Prompts

Edit `backend/app/prompts/diagnosis_hu.py`:

```python
DIAGNOSIS_SYSTEM_PROMPT = """
Te egy tapasztalt autoszerelo AI asszisztens vagy, aki magyar nyelven kommunikal.

Feladatod:
1. Elemezd a DTC hibakodokat es a leirt tuneteket
2. Azonositsd a valoszinu okokat
3. Adj javitasi javaslatokat

Fontos szabályok:
- Mindig magyarul valaszolj
- Adj konkret, hasznalhato tanacsokat
- Jelold a biztonsagi kockazatokat
"""

DIAGNOSIS_USER_TEMPLATE = """
Jármű: {make} {model} ({year})
Motor: {engine}

DTC kódok: {dtc_codes}

Tünetek: {symptoms}

Kapcsolódó információk:
{context}

Készíts részletes diagnózist a fenti információk alapján.
"""
```

### Adding New LLM Providers

Edit `backend/app/services/llm_provider.py`:

```python
class LLMProvider:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER

    async def generate(self, prompt: str) -> str:
        if self.provider == "anthropic":
            return await self._anthropic_generate(prompt)
        elif self.provider == "openai":
            return await self._openai_generate(prompt)
        elif self.provider == "ollama":
            return await self._ollama_generate(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    async def _ollama_generate(self, prompt: str) -> str:
        """Local LLM using Ollama."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": settings.OLLAMA_MODEL,
                    "prompt": prompt,
                }
            )
            return response.json()["response"]
```

---

## Database Operations

### Creating Migrations

```bash
# Create new migration
cd backend
alembic revision --autogenerate -m "Add new column to dtc_codes"

# Review the generated migration in alembic/versions/

# Apply migration
alembic upgrade head

# Rollback
alembic downgrade -1
```

### Common Migration Patterns

```python
# alembic/versions/xxx_add_new_column.py
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.add_column('dtc_codes',
        sa.Column('new_field', sa.String(100), nullable=True)
    )

    # Add index
    op.create_index('ix_dtc_codes_new_field', 'dtc_codes', ['new_field'])

def downgrade():
    op.drop_index('ix_dtc_codes_new_field', 'dtc_codes')
    op.drop_column('dtc_codes', 'new_field')
```

### Neo4j Graph Operations

```python
# Creating nodes and relationships
from neomodel import db

# Raw Cypher query
results, meta = db.cypher_query("""
    MATCH (d:DTCCode {code: $code})-[:HAS_SYMPTOM]->(s:Symptom)
    RETURN d, s
""", {"code": "P0101"})

# Using neomodel
from app.db.neo4j_models import DTCNode

dtc = DTCNode.nodes.get(code="P0101")
symptoms = dtc.symptoms.all()
```

### Redis Cache Operations

```python
from app.db.redis_cache import get_cache_service

async def example():
    cache = await get_cache_service()

    # Set value
    await cache.set("key", {"data": "value"}, ttl=3600)

    # Get value
    value = await cache.get("key")

    # Delete
    await cache.delete("key")

    # Delete pattern
    await cache.delete_pattern("dtc:*")
```

---

## Testing

### Running Tests

```bash
# All tests
cd backend
pytest

# With coverage
pytest --cov=app --cov-report=html

# Specific test file
pytest tests/test_diagnosis_service.py

# Specific test
pytest tests/test_diagnosis_service.py::test_analyze_vehicle

# Integration tests
pytest tests/integration/

# Verbose output
pytest -v
```

### Writing Tests

```python
# tests/test_example.py
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_dtc_search(client):
    response = await client.get("/api/v1/dtc/search?q=P0101")
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
```

### Test Fixtures

```python
# tests/conftest.py
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from app.db.postgres.models import Base

@pytest.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSession(engine) as session:
        yield session

@pytest.fixture
def sample_diagnosis_request():
    return {
        "vehicle_make": "Volkswagen",
        "vehicle_model": "Golf",
        "vehicle_year": 2018,
        "dtc_codes": ["P0101"],
        "symptoms": "Motor nehezen indul hidegben",
    }
```

---

## Contributing

### Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and write tests
4. Run tests: `pytest`
5. Format code: `black backend/ && npm run lint`
6. Commit changes: `git commit -m "Add my feature"`
7. Push to fork: `git push origin feature/my-feature`
8. Create Pull Request

### Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Examples:**
```
feat(diagnosis): add confidence scoring
fix(auth): handle expired refresh tokens
docs(api): update endpoint examples
```

### Code Review Checklist

- [ ] Tests pass
- [ ] Code is formatted
- [ ] No new linting errors
- [ ] Documentation updated
- [ ] Migration tested (if applicable)
- [ ] Hungarian translations included (if user-facing)
