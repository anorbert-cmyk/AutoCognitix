"""
Vehicle Service for Neo4j queries.

Provides async methods to query vehicle data from Neo4j:
- Get all makes (manufacturers)
- Get models for a make
- Get years for a make/model
- Get vehicle details by ID
- Get complaints and recalls for a vehicle
"""

import asyncio
from typing import Any

from neomodel import db as neomodel_db

from app.core.logging import get_logger

logger = get_logger(__name__)


class VehicleService:
    """
    Service for querying vehicle data from Neo4j.

    Uses Neomodel for ORM-style access and raw Cypher for complex queries.
    """

    async def get_all_makes(
        self,
        search: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Get all unique vehicle makes from Neo4j.

        Args:
            search: Optional search term to filter makes
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            Tuple of (list of make dictionaries, total count)
        """
        # Build the query
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

        # Execute queries in thread pool (neomodel is sync)
        loop = asyncio.get_event_loop()
        results, _ = await loop.run_in_executor(
            None, lambda: neomodel_db.cypher_query(query, params)
        )
        count_results, _ = await loop.run_in_executor(
            None, lambda: neomodel_db.cypher_query(count_query, count_params)
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

    async def get_models_for_make(
        self,
        make: str,
        search: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Get all models for a specific make from Neo4j.

        Args:
            make: Vehicle make (manufacturer)
            search: Optional search term to filter models
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            Tuple of (list of model dictionaries, total count)
        """
        # Normalize the make name for case-insensitive comparison
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

        loop = asyncio.get_event_loop()
        results, _ = await loop.run_in_executor(
            None, lambda: neomodel_db.cypher_query(query, params)
        )
        count_results, _ = await loop.run_in_executor(
            None, lambda: neomodel_db.cypher_query(count_query, count_params)
        )

        total = count_results[0][0] if count_results else 0
        make_id = make.lower().replace(" ", "_").replace("-", "_")

        models = []
        for row in results:
            model_name, year_start, year_end, body_types_raw = row
            # Flatten body types (may be nested lists)
            body_types = []
            for bt in body_types_raw:
                if isinstance(bt, list):
                    body_types.extend(bt)
                elif bt:
                    body_types.append(bt)
            body_types = list(set(body_types))

            models.append(
                {
                    "id": model_name.lower().replace(" ", "_").replace("-", "_"),
                    "name": model_name,
                    "make_id": make_id,
                    "year_start": year_start,
                    "year_end": year_end,
                    "body_types": body_types,
                }
            )

        return models, total

    async def get_years_for_vehicle(
        self,
        make: str,
        model: str,
    ) -> list[int]:
        """
        Get all available years for a specific make and model.

        Args:
            make: Vehicle make
            model: Vehicle model

        Returns:
            List of years in descending order
        """
        query = """
            MATCH (v:VehicleNode)
            WHERE toLower(v.make) = toLower($make)
              AND toLower(v.model) = toLower($model)
            RETURN v.year_start AS year_start, v.year_end AS year_end
        """
        params = {"make": make, "model": model}

        loop = asyncio.get_event_loop()
        results, _ = await loop.run_in_executor(
            None, lambda: neomodel_db.cypher_query(query, params)
        )

        years = set()
        current_year = 2026  # Default max year

        for row in results:
            year_start, year_end = row
            if year_start:
                start = int(year_start)
                end = int(year_end) if year_end else current_year
                years.update(range(start, end + 1))

        return sorted(years, reverse=True)

    async def get_vehicle_by_id(
        self,
        vehicle_id: str,
    ) -> dict[str, Any] | None:
        """
        Get vehicle details by Neo4j UID.

        Args:
            vehicle_id: Vehicle UID

        Returns:
            Vehicle dictionary or None if not found
        """
        query = """
            MATCH (v:VehicleNode {uid: $vehicle_id})
            RETURN v
        """
        params = {"vehicle_id": vehicle_id}

        loop = asyncio.get_event_loop()
        results, _ = await loop.run_in_executor(
            None, lambda: neomodel_db.cypher_query(query, params)
        )

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
        """
        Find a vehicle by make, model, and optional year.

        Args:
            make: Vehicle make
            model: Vehicle model
            year: Optional year

        Returns:
            Vehicle dictionary or None if not found
        """
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

        loop = asyncio.get_event_loop()
        results, _ = await loop.run_in_executor(
            None, lambda: neomodel_db.cypher_query(query, params)
        )

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
        """
        Get common DTC issues for a vehicle.

        Args:
            make: Vehicle make
            model: Vehicle model
            year: Optional year filter

        Returns:
            List of common DTC issues
        """
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

        loop = asyncio.get_event_loop()
        results, _ = await loop.run_in_executor(
            None, lambda: neomodel_db.cypher_query(query, params)
        )

        issues = []
        for row in results:
            issues.append(
                {
                    "code": row[0],
                    "description_en": row[1],
                    "description_hu": row[2],
                    "severity": row[3],
                    "frequency": row[4],
                    "occurrence_count": row[5],
                }
            )

        return issues

    def _node_to_dict(self, node: Any) -> dict[str, Any]:
        """Convert a Neo4j node to a dictionary."""
        if hasattr(node, "__dict__"):
            # Neomodel node
            return {
                "id": node.get("uid")
                if hasattr(node, "get")
                else node.uid
                if hasattr(node, "uid")
                else None,
                "make": node.get("make")
                if hasattr(node, "get")
                else node.make
                if hasattr(node, "make")
                else None,
                "model": node.get("model")
                if hasattr(node, "get")
                else node.model
                if hasattr(node, "model")
                else None,
                "year_start": node.get("year_start")
                if hasattr(node, "get")
                else node.year_start
                if hasattr(node, "year_start")
                else None,
                "year_end": node.get("year_end")
                if hasattr(node, "get")
                else node.year_end
                if hasattr(node, "year_end")
                else None,
                "platform": node.get("platform")
                if hasattr(node, "get")
                else node.platform
                if hasattr(node, "platform")
                else None,
                "engine_codes": node.get("engine_codes")
                if hasattr(node, "get")
                else node.engine_codes
                if hasattr(node, "engine_codes")
                else [],
                "body_types": node.get("body_types")
                if hasattr(node, "get")
                else node.body_types
                if hasattr(node, "body_types")
                else [],
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
            # German
            "volkswagen": "Germany",
            "audi": "Germany",
            "bmw": "Germany",
            "mercedes-benz": "Germany",
            "mercedes": "Germany",
            "opel": "Germany",
            "porsche": "Germany",
            "mini": "Germany",
            # Japanese
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
            # Korean
            "hyundai": "South Korea",
            "kia": "South Korea",
            "genesis": "South Korea",
            "ssangyong": "South Korea",
            # American
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
            # French
            "renault": "France",
            "peugeot": "France",
            "citroen": "France",
            "citroën": "France",
            "dacia": "France",
            "alpine": "France",
            # Italian
            "fiat": "Italy",
            "alfa romeo": "Italy",
            "alfa": "Italy",
            "ferrari": "Italy",
            "lamborghini": "Italy",
            "maserati": "Italy",
            "lancia": "Italy",
            # British
            "jaguar": "UK",
            "land rover": "UK",
            "bentley": "UK",
            "rolls-royce": "UK",
            "aston martin": "UK",
            "lotus": "UK",
            "mclaren": "UK",
            "mg": "UK",
            # Czech
            "skoda": "Czech Republic",
            "škoda": "Czech Republic",
            # Spanish
            "seat": "Spain",
            "cupra": "Spain",
            # Swedish
            "volvo": "Sweden",
            "saab": "Sweden",
            # Chinese
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
    """
    Get or create vehicle service instance.

    Returns:
        VehicleService instance
    """
    global _vehicle_service
    if _vehicle_service is None:
        _vehicle_service = VehicleService()
    return _vehicle_service
