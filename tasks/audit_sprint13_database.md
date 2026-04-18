# Sprint 13 Database Audit

## A) Alembic drift
- **Severity**: MEDIUM
- **Találatok** (TBD)

## B) Cross-DB source of truth
PostgreSQL az egyértelmű source of truth. A `scripts/data_sync.py` kizárólag PostgreSQL -> Neo4j és PostgreSQL -> Qdrant irányban hidratál (`sync_postgres_to_neo4j` line 429, `sync_postgres_to_qdrant` line 521): PG-ből olvassa a DTC kódokat és abból hoz létre/frissít Neo4j `DTCNode`-okat, illetve Qdrant vektor pointokat. Visszairányú sync (`sync_neo4j_to_postgres` line 653) is létezik, de a `--all` flag az első kettőt futtatja; a Neo4j -> PG külön opcióval hívható ("fallback"), nem default pipeline. A Neo4j és Qdrant tehát derived store, PG a kanonikus.

## C) Tranzakció határok
_(In progress.)_
