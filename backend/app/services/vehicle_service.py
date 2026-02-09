"""
Vehicle Service with Neo4j primary and PostgreSQL fallback.

Queries Neo4j first for vehicle data. Falls back to PostgreSQL
vehicle_makes/vehicle_models tables if Neo4j returns empty results.
"""

import asyncio
from typing import Any

from neomodel import db as neomodel_db
from sqlalchemy import func, select

from app.core.logging import get_logger
from app.db.postgres.models import VehicleMake, VehicleModel as VehicleModelDB
from app.db.postgres.session import async_session_maker

logger = get_logger(__name__)


class VehicleService:
    """
    Service for querying vehicle data.

    Primary: Neo4j VehicleNode
    Fallback: PostgreSQL vehicle_makes/vehicle_models (seeded via migration 010)
    """

    async def get_all_makes(
        self,
        search: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Get all vehicle makes. Tries Neo4j first, falls back to PostgreSQL."""
        try:
            makes, total = await self._get_makes_neo4j(search, limit, offset)
            if total > 0:
                return makes, total
        except Exception as e:
            logger.warning(f"Neo4j makes query failed, using PostgreSQL: {e}")

        return await self._get_makes_postgres(search, limit, offset)

    async def _get_makes_neo4j(
        self,
        search: str | None,
        limit: int,
        offset: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Get makes from Neo4j VehicleNode."""
        if search:
            query = """
                MATCH (v:VehicleNode)
                WHERE toLower(v.make) CONTAINS toLower($search)
                WITH DISTINCT v.make AS make
                RETURN make
                ORDER BY make
                SKIP $offset
                LIMIT $limit
            """
            count_query = """
                MATCH (v:VehicleNode)
                WHERE toLower(v.make) CONTAINS toLower($search)
                RETURN COUNT(DISTINCT v.make) AS total
            """
            params = {"search": search, "limit": limit, "offset": offset}
            count_params = {"search": search}
        else:
            query = """
                MATCH (v:VehicleNode)
                WITH DISTINCT v.make AS make
                RETURN make
                ORDER BY make
                SKIP $offset
                LIMIT $limit
            """
            count_query = """
                MATCH (v:VehicleNode)
                RETURN COUNT(DISTINCT v.make) AS total
            """
            params = {"limit": limit, "offset": offset}
            count_params = {}

        results, _ = await asyncio.to_thread(neomodel_db.cypher_query, query, params)
        count_results, _ = await asyncio.to_thread(
            neomodel_db.cypher_query, count_query, count_params
        )

        total = count_results[0][0] if count_results else 0
        makes = []
        for row in results:
            make_name = row[0]
            makes.append(
                {
                    "id": make_name.lower().replace(" ", "_").replace("-", "_"),
                    "name": make_name,
                    "country": self._get_country_for_make(make_name),
                }
            )

        return makes, total

    async def _get_makes_postgres(
        self,
        search: str | None,
        limit: int,
        offset: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Get makes from PostgreSQL vehicle_makes table."""
        async with async_session_maker() as session:
            stmt = select(VehicleMake)
            count_stmt = select(func.count(VehicleMake.id))

            if search:
                stmt = stmt.where(VehicleMake.name.ilike(f"%{search}%"))
                count_stmt = count_stmt.where(VehicleMake.name.ilike(f"%{search}%"))

            stmt = stmt.order_by(VehicleMake.name).offset(offset).limit(limit)

            result = await session.execute(stmt)
            count_result = await session.execute(count_stmt)

            total = count_result.scalar() or 0
            makes = [
                {"id": m.id, "name": m.name, "country": m.country} for m in result.scalars().all()
            ]

        return makes, total

    async def get_models_for_make(
        self,
        make: str,
        search: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Get models for a make. Tries Neo4j first, falls back to PostgreSQL."""
        try:
            models, total = await self._get_models_neo4j(make, search, limit, offset)
            if total > 0:
                return models, total
        except Exception as e:
            logger.warning(f"Neo4j models query failed, using PostgreSQL: {e}")

        return await self._get_models_postgres(make, search, limit, offset)

    async def _get_models_neo4j(
        self,
        make: str,
        search: str | None,
        limit: int,
        offset: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Get models from Neo4j VehicleNode."""
        if search:
            query = """
                MATCH (v:VehicleNode)
                WHERE toLower(v.make) = toLower($make)
                  AND toLower(v.model) CONTAINS toLower($search)
                WITH DISTINCT v.model AS model,
                     MIN(v.year_start) AS year_start,
                     MAX(v.year_end) AS year_end,
                     COLLECT(DISTINCT v.body_types) AS body_types
                RETURN model, year_start, year_end, body_types
                ORDER BY model
                SKIP $offset
                LIMIT $limit
            """
            count_query = """
                MATCH (v:VehicleNode)
                WHERE toLower(v.make) = toLower($make)
                  AND toLower(v.model) CONTAINS toLower($search)
                RETURN COUNT(DISTINCT v.model) AS total
            """
            params = {"make": make, "search": search, "limit": limit, "offset": offset}
            count_params = {"make": make, "search": search}
        else:
            query = """
                MATCH (v:VehicleNode)
                WHERE toLower(v.make) = toLower($make)
                WITH DISTINCT v.model AS model,
                     MIN(v.year_start) AS year_start,
                     MAX(v.year_end) AS year_end,
                     COLLECT(DISTINCT v.body_types) AS body_types
                RETURN model, year_start, year_end, body_types
                ORDER BY model
                SKIP $offset
                LIMIT $limit
            """
            count_query = """
                MATCH (v:VehicleNode)
                WHERE toLower(v.make) = toLower($make)
                RETURN COUNT(DISTINCT v.model) AS total
            """
            params = {"make": make, "limit": limit, "offset": offset}
            count_params = {"make": make}

        results, _ = await asyncio.to_thread(neomodel_db.cypher_query, query, params)
        count_results, _ = await asyncio.to_thread(
            neomodel_db.cypher_query, count_query, count_params
        )

        total = count_results[0][0] if count_results else 0
        make_id = make.lower().replace(" ", "_").replace("-", "_")

        models = []
        for row in results:
            model_name, year_start, year_end, body_types_raw = row
            body_types = []
            for bt in body_types_raw:
                if isinstance(bt, list):
                    body_types.extend(bt)
                elif bt:
                    body_types.append(bt)

            models.append(
                {
                    "id": model_name.lower().replace(" ", "_").replace("-", "_"),
                    "name": model_name,
                    "make_id": make_id,
                    "year_start": year_start,
                    "year_end": year_end,
                    "body_types": list(set(body_types)),
                }
            )

        return models, total

    async def _get_models_postgres(
        self,
        make: str,
        search: str | None,
        limit: int,
        offset: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Get models from PostgreSQL vehicle_models table."""
        make_id = make.lower().replace(" ", "_").replace("-", "_")

        async with async_session_maker() as session:
            make_obj_result = await session.execute(
                select(VehicleMake).where(
                    (VehicleMake.id == make_id) | (VehicleMake.name.ilike(make))
                )
            )
            make_obj = make_obj_result.scalar_one_or_none()
            if not make_obj:
                return [], 0

            stmt = select(VehicleModelDB).where(VehicleModelDB.make_id == make_obj.id)
            count_stmt = select(func.count(VehicleModelDB.id)).where(
                VehicleModelDB.make_id == make_obj.id
            )

            if search:
                stmt = stmt.where(VehicleModelDB.name.ilike(f"%{search}%"))
                count_stmt = count_stmt.where(VehicleModelDB.name.ilike(f"%{search}%"))

            stmt = stmt.order_by(VehicleModelDB.name).offset(offset).limit(limit)

            result = await session.execute(stmt)
            count_result = await session.execute(count_stmt)

            total = count_result.scalar() or 0
            models: list[dict[str, Any]] = [
                {
                    "id": m.id,
                    "name": m.name,
                    "make_id": make_obj.id,
                    "year_start": m.year_start,
                    "year_end": m.year_end,
                    "body_types": [],
                }
                for m in result.scalars().all()
            ]

        return models, total

    async def get_years_for_vehicle(
        self,
        make: str,
        model: str,
    ) -> list[int]:
        """Get all available years for a specific make and model."""
        query = """
            MATCH (v:VehicleNode)
            WHERE toLower(v.make) = toLower($make)
              AND toLower(v.model) = toLower($model)
            RETURN v.year_start AS year_start, v.year_end AS year_end
        """
        params = {"make": make, "model": model}

        try:
            results, _ = await asyncio.to_thread(neomodel_db.cypher_query, query, params)
        except Exception:
            results = []

        years: set[int] = set()
        current_year = 2026

        for row in results:
            year_start, year_end = row
            if year_start:
                start = int(year_start)
                end = int(year_end) if year_end else current_year
                years.update(range(start, end + 1))

        # Fallback: if no Neo4j results, use PostgreSQL year_start
        if not years:
            try:
                async with async_session_maker() as session:
                    make_id = make.lower().replace(" ", "_").replace("-", "_")
                    stmt = select(VehicleModelDB).where(
                        (VehicleModelDB.make_id == make_id) & (VehicleModelDB.name.ilike(model))
                    )
                    result = await session.execute(stmt)
                    for m in result.scalars().all():
                        if m.year_start:
                            end = m.year_end if m.year_end else current_year
                            years.update(range(m.year_start, end + 1))
            except Exception as e:
                logger.warning(f"PostgreSQL years fallback failed: {e}")

        return sorted(years, reverse=True)

    async def get_vehicle_by_id(
        self,
        vehicle_id: str,
    ) -> dict[str, Any] | None:
        """Get vehicle details by Neo4j UID."""
        query = """
            MATCH (v:VehicleNode {uid: $vehicle_id})
            RETURN v
        """
        params = {"vehicle_id": vehicle_id}

        results, _ = await asyncio.to_thread(neomodel_db.cypher_query, query, params)

        if not results:
            return None

        node = results[0][0]
        return self._node_to_dict(node)

    async def find_vehicle(
        self,
        make: str,
        model: str,
        year: int | None = None,
    ) -> dict[str, Any] | None:
        """Find a vehicle by make, model, and optional year."""
        if year:
            query = """
                MATCH (v:VehicleNode)
                WHERE toLower(v.make) = toLower($make)
                  AND toLower(v.model) = toLower($model)
                  AND (v.year_start IS NULL OR v.year_start <= $year)
                  AND (v.year_end IS NULL OR v.year_end >= $year)
                RETURN v
                LIMIT 1
            """
            params = {"make": make, "model": model, "year": year}
        else:
            query = """
                MATCH (v:VehicleNode)
                WHERE toLower(v.make) = toLower($make)
                  AND toLower(v.model) = toLower($model)
                RETURN v
                LIMIT 1
            """
            params = {"make": make, "model": model}

        results, _ = await asyncio.to_thread(neomodel_db.cypher_query, query, params)

        if not results:
            return None

        node = results[0][0]
        return self._node_to_dict(node)

    async def get_vehicle_common_issues(
        self,
        make: str,
        model: str,
        year: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get common DTC issues for a vehicle from Neo4j."""
        if year:
            query = """
                MATCH (v:VehicleNode)-[r:HAS_COMMON_ISSUE]->(d:DTCNode)
                WHERE toLower(v.make) = toLower($make)
                  AND toLower(v.model) = toLower($model)
                  AND (v.year_start IS NULL OR v.year_start <= $year)
                  AND (v.year_end IS NULL OR v.year_end >= $year)
                RETURN d.code AS code,
                       d.description_en AS description_en,
                       d.description_hu AS description_hu,
                       d.severity AS severity,
                       r.frequency AS frequency,
                       r.occurrence_count AS occurrence_count
                ORDER BY r.occurrence_count DESC
            """
            params = {"make": make, "model": model, "year": year}
        else:
            query = """
                MATCH (v:VehicleNode)-[r:HAS_COMMON_ISSUE]->(d:DTCNode)
                WHERE toLower(v.make) = toLower($make)
                  AND toLower(v.model) = toLower($model)
                RETURN d.code AS code,
                       d.description_en AS description_en,
                       d.description_hu AS description_hu,
                       d.severity AS severity,
                       r.frequency AS frequency,
                       r.occurrence_count AS occurrence_count
                ORDER BY r.occurrence_count DESC
            """
            params = {"make": make, "model": model}

        results, _ = await asyncio.to_thread(neomodel_db.cypher_query, query, params)

        return [
            {
                "code": row[0],
                "description_en": row[1],
                "description_hu": row[2],
                "severity": row[3],
                "frequency": row[4],
                "occurrence_count": row[5],
            }
            for row in results
        ]

    def _node_to_dict(self, node: Any) -> dict[str, Any]:
        """Convert a Neo4j node to a dictionary."""
        if hasattr(node, "__dict__"):
            return {
                "id": node.get("uid") if hasattr(node, "get") else getattr(node, "uid", None),
                "make": node.get("make") if hasattr(node, "get") else getattr(node, "make", None),
                "model": node.get("model")
                if hasattr(node, "get")
                else getattr(node, "model", None),
                "year_start": node.get("year_start")
                if hasattr(node, "get")
                else getattr(node, "year_start", None),
                "year_end": node.get("year_end")
                if hasattr(node, "get")
                else getattr(node, "year_end", None),
                "platform": node.get("platform")
                if hasattr(node, "get")
                else getattr(node, "platform", None),
                "engine_codes": node.get("engine_codes")
                if hasattr(node, "get")
                else getattr(node, "engine_codes", []),
                "body_types": node.get("body_types")
                if hasattr(node, "get")
                else getattr(node, "body_types", []),
            }
        elif isinstance(node, dict):
            return {
                "id": node.get("uid"),
                "make": node.get("make"),
                "model": node.get("model"),
                "year_start": node.get("year_start"),
                "year_end": node.get("year_end"),
                "platform": node.get("platform"),
                "engine_codes": node.get("engine_codes", []),
                "body_types": node.get("body_types", []),
            }
        return {}

    @staticmethod
    def _get_country_for_make(make: str) -> str | None:
        """Get country of origin for a vehicle make."""
        make_countries = {
            "volkswagen": "Germany",
            "audi": "Germany",
            "bmw": "Germany",
            "mercedes-benz": "Germany",
            "mercedes": "Germany",
            "opel": "Germany",
            "porsche": "Germany",
            "mini": "Germany",
            "toyota": "Japan",
            "honda": "Japan",
            "nissan": "Japan",
            "mazda": "Japan",
            "suzuki": "Japan",
            "subaru": "Japan",
            "mitsubishi": "Japan",
            "lexus": "Japan",
            "infiniti": "Japan",
            "acura": "Japan",
            "daihatsu": "Japan",
            "isuzu": "Japan",
            "hyundai": "South Korea",
            "kia": "South Korea",
            "genesis": "South Korea",
            "ssangyong": "South Korea",
            "ford": "USA",
            "chevrolet": "USA",
            "gmc": "USA",
            "dodge": "USA",
            "jeep": "USA",
            "chrysler": "USA",
            "ram": "USA",
            "buick": "USA",
            "cadillac": "USA",
            "lincoln": "USA",
            "tesla": "USA",
            "renault": "France",
            "peugeot": "France",
            "citroen": "France",
            "citroën": "France",
            "dacia": "France",
            "alpine": "France",
            "fiat": "Italy",
            "alfa romeo": "Italy",
            "alfa": "Italy",
            "ferrari": "Italy",
            "lamborghini": "Italy",
            "maserati": "Italy",
            "lancia": "Italy",
            "jaguar": "UK",
            "land rover": "UK",
            "bentley": "UK",
            "rolls-royce": "UK",
            "aston martin": "UK",
            "lotus": "UK",
            "mclaren": "UK",
            "mg": "UK",
            "skoda": "Czech Republic",
            "škoda": "Czech Republic",
            "seat": "Spain",
            "cupra": "Spain",
            "volvo": "Sweden",
            "saab": "Sweden",
            "byd": "China",
            "geely": "China",
            "great wall": "China",
            "nio": "China",
            "xpeng": "China",
            "li auto": "China",
        }
        return make_countries.get(make.lower())


# Singleton instance
_vehicle_service: VehicleService | None = None


def get_vehicle_service() -> VehicleService:
    """Get or create vehicle service instance."""
    global _vehicle_service
    if _vehicle_service is None:
        _vehicle_service = VehicleService()
    return _vehicle_service
