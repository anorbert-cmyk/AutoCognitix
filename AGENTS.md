# AGENTS.md

## Cursor Cloud specific instructions

### Architecture

AutoCognitix is a monorepo with two main services:
- **Backend**: FastAPI (Python 3.12) at port 8000
- **Frontend**: React 18 + Vite (Node 22) at port 3000

Four database services run via Docker Compose: PostgreSQL 16, Neo4j 5.15, Qdrant 1.7.4, Redis 7.

### Starting databases

```bash
sudo dockerd &  # if Docker daemon not running
sudo docker compose up -d postgres neo4j qdrant redis
```

### Environment file

The backend loads `.env` from the working directory. A symlink at `backend/.env -> ../.env` ensures settings load whether running from `backend/` or workspace root. Copy `.env.local.example` to `.env` for local dev values.

The `SECRET_KEY` and `JWT_SECRET_KEY` must each be >= 32 characters or the app will refuse to start.

### Running services

See `CLAUDE.md` "Gyakori Parancsok" section and `README.md` "Quick Start" for standard commands.

- **Backend dev**: `source backend/venv/bin/activate && cd backend && uvicorn app.main:app --reload`
- **Frontend dev**: `cd frontend && npm run dev`
- **Migrations**: `cd backend && alembic upgrade head`

### Linting & testing

| Service | Lint | Test |
|---------|------|------|
| Backend | `cd backend && python3 -m ruff check app tests` | `cd backend && python3 -m pytest tests -v` |
| Frontend | `cd frontend && npx eslint . --ext ts,tsx` | `cd frontend && npx vitest run` |

### Gotchas

- The Qdrant Docker container healthcheck may show "unhealthy" due to missing `wget` in the container, but the service works fine (verify with `curl http://localhost:6333/readyz`).
- Backend tests in `tests/test_api_endpoints.py` (DTC-related) fail because `conftest.py` defaults `DATABASE_URL` to a `test` user that doesn't exist in the dev PostgreSQL; unit tests (e.g., `test_dtc_validation.py`, `test_translation_fixer.py`) pass cleanly.
- The Qdrant client version (1.17) is newer than the Qdrant server (1.7.4); this produces a warning but functions correctly for basic operations.
- The backend `requirements.txt` includes `torch==2.2.0` (~2 GB), so initial `pip install` is slow.
- The `/demo` route on the frontend shows a pre-filled P0300 diagnostic report with real Hungarian parts pricing — useful for quick visual verification without needing seed data or LLM keys.
