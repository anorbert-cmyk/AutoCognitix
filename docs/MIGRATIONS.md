# AutoCognitix Database Migrations Guide

This document explains how to manage database migrations for the AutoCognitix project.

## Overview

AutoCognitix uses three database systems:

| Database   | Purpose                      | Migration Tool                 |
|------------|------------------------------|--------------------------------|
| PostgreSQL | Relational data (users, DTCs)| Alembic                        |
| Neo4j      | Diagnostic graph             | Cypher scripts                 |
| Qdrant     | Vector embeddings            | Python initialization scripts  |

## Prerequisites

Before running migrations, ensure you have:

1. **PostgreSQL** running (local Docker or Railway)
2. **Neo4j** running (local Docker or Neo4j Aura)
3. **Qdrant** running (local Docker or Qdrant Cloud)
4. Environment variables configured (see `.env.example`)

## PostgreSQL Migrations (Alembic)

### Configuration

The Alembic configuration is in:
- `backend/alembic.ini` - Main configuration
- `backend/alembic/env.py` - Async-enabled migration environment
- `backend/alembic/versions/` - Migration scripts

### Running Migrations

```bash
# Navigate to backend directory
cd backend

# Apply all pending migrations
alembic upgrade head

# Apply specific migration
alembic upgrade 001_initial_schema

# Downgrade one step
alembic downgrade -1

# Downgrade to specific revision
alembic downgrade 002_add_dtc_sources_column

# Downgrade all (DANGEROUS - drops all tables)
alembic downgrade base

# Check current version
alembic current

# Show migration history
alembic history --verbose
```

### Creating New Migrations

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "description of changes"

# Create empty migration (for manual SQL)
alembic revision -m "description of changes"
```

### Migration Files

| File                                    | Description                                    |
|-----------------------------------------|------------------------------------------------|
| `001_initial_schema.py`                 | Core tables: users, vehicles, DTCs, sessions   |
| `002_add_dtc_sources_column.py`         | Add sources tracking to DTC codes              |
| `003_add_vehicle_recalls_complaints.py` | NHTSA recalls and complaints tables            |
| `004_add_performance_indexes.py`        | Performance indexes for all tables             |
| `005_vehicle_comprehensive_schema.py`   | Extended vehicle schema (engines, platforms)   |

### Migration Best Practices

1. **Always test migrations locally first**
   ```bash
   # Create fresh database
   docker-compose down -v
   docker-compose up -d postgres
   alembic upgrade head
   ```

2. **Backup before production migrations**
   ```bash
   pg_dump -h localhost -U autocognitix autocognitix > backup.sql
   ```

3. **Use transactions** - All migrations run in transactions by default

4. **Write reversible migrations** - Always implement `downgrade()`

5. **Don't modify existing migrations** - Create new ones instead

## Neo4j Schema Setup

### Configuration

The Neo4j schema is defined in:
- `scripts/neo4j_schema.cypher` - Cypher DDL commands
- `scripts/setup_neo4j_indexes.py` - Python management script

### Running Schema Setup

```bash
# Using Python script (recommended)
python scripts/setup_neo4j_indexes.py

# Verify existing indexes
python scripts/setup_neo4j_indexes.py --verify

# Drop and recreate all indexes
python scripts/setup_neo4j_indexes.py --drop

# Print Cypher commands (for manual execution)
python scripts/setup_neo4j_indexes.py --print-cypher
```

### Using Cypher Shell

```bash
# Run schema file directly
cypher-shell -u neo4j -p <password> -f scripts/neo4j_schema.cypher

# Or in Neo4j Browser, paste the contents of neo4j_schema.cypher
```

### Neo4j Indexes Created

| Index Type         | Node Label      | Properties                        |
|--------------------|-----------------|-----------------------------------|
| Unique Constraint  | DTCNode         | code                              |
| B-tree Index       | SymptomNode     | name                              |
| B-tree Index       | ComponentNode   | name, system                      |
| B-tree Index       | RepairNode      | name                              |
| B-tree Index       | PartNode        | name, part_number                 |
| B-tree Index       | VehicleNode     | make, model                       |
| Full-text Index    | DTCNode         | description_hu, description_en    |
| Full-text Index    | SymptomNode     | description, description_hu       |
| Composite Index    | VehicleNode     | make + model                      |

## Qdrant Collection Setup

### Configuration

The Qdrant collections are managed by:
- `scripts/init_qdrant.py` - Initialization script
- `backend/app/db/qdrant_client.py` - Runtime client

### Running Initialization

```bash
# Initialize all collections
python scripts/init_qdrant.py

# Verify collections exist
python scripts/init_qdrant.py --verify

# Drop and recreate (for schema changes)
python scripts/init_qdrant.py --recreate

# Show collection info
python scripts/init_qdrant.py --info

# Include legacy collections
python scripts/init_qdrant.py --include-legacy
```

### Qdrant Collections

| Collection                  | Vector Size | Description                        |
|-----------------------------|-------------|------------------------------------|
| dtc_embeddings_hu           | 768         | Hungarian DTC descriptions         |
| symptom_embeddings_hu       | 768         | Hungarian symptom descriptions     |
| known_issue_embeddings_hu   | 768         | Known issue descriptions           |
| dtc_embeddings (legacy)     | 384         | English DTC (backward compat)      |
| symptom_embeddings (legacy) | 384         | English symptoms (backward compat) |

### Indexing Data

After initializing collections, populate them with:

```bash
# Index DTC codes and symptoms
python scripts/index_qdrant.py --all

# Index only DTC codes
python scripts/index_qdrant.py --dtc

# Recreate and reindex
python scripts/index_qdrant.py --all --recreate
```

## Complete Setup from Scratch

For a fresh installation or after `docker-compose down -v`:

```bash
# 1. Start all services
docker-compose up -d

# 2. Wait for services to be ready (10-15 seconds)
sleep 15

# 3. Run PostgreSQL migrations
cd backend
alembic upgrade head

# 4. Set up Neo4j indexes
python scripts/setup_neo4j_indexes.py

# 5. Initialize Qdrant collections
python scripts/init_qdrant.py

# 6. Seed initial data
python scripts/seed_database.py

# 7. Index data into Qdrant
python scripts/index_qdrant.py --all

# 8. Verify setup
python scripts/health_check.py
```

## Railway Deployment

For Railway deployment, migrations run automatically:

1. **PostgreSQL**: Migrations run on deploy via `railway.toml` start command
2. **Neo4j Aura**: Run `setup_neo4j_indexes.py` manually after connecting
3. **Qdrant Cloud**: Run `init_qdrant.py` manually after connecting

```bash
# Connect to Railway and run migrations
railway run alembic upgrade head

# Or run via Railway shell
railway shell
cd backend && alembic upgrade head
```

## Troubleshooting

### PostgreSQL Issues

**Connection refused**
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# Check logs
docker logs autocognitix_postgres
```

**Migration fails**
```bash
# Check current state
alembic current

# Show recent history
alembic history -v

# Stamp database to specific version (fix out-of-sync state)
alembic stamp 001_initial_schema
```

### Neo4j Issues

**Connection failed**
```bash
# Verify connection settings
python -c "from backend.app.core.config import settings; print(settings.NEO4J_URI)"

# Test connection
python scripts/setup_neo4j_indexes.py --verify
```

**Index already exists**
```bash
# Drop and recreate
python scripts/setup_neo4j_indexes.py --drop
```

### Qdrant Issues

**Connection failed**
```bash
# Check if Qdrant is running
curl http://localhost:6333/health

# Verify settings
python -c "from backend.app.core.config import settings; print(settings.QDRANT_HOST, settings.QDRANT_PORT)"
```

**Collection schema mismatch**
```bash
# Recreate collections (WARNING: deletes data)
python scripts/init_qdrant.py --recreate
```

## Version History

| Version | Date       | Changes                                      |
|---------|------------|----------------------------------------------|
| 001     | 2024-01-01 | Initial schema                               |
| 002     | 2026-02-05 | Add DTC sources column                       |
| 003     | 2026-02-05 | NHTSA recalls and complaints                 |
| 004     | 2026-02-05 | Performance indexes for all tables           |
| 005     | 2026-02-05 | Extended vehicle schema (engines, platforms) |

## Related Documentation

- [CLAUDE.md](../CLAUDE.md) - Project overview and conventions
- [.env.example](../.env.example) - Environment configuration
- [docker-compose.yml](../docker-compose.yml) - Service definitions
