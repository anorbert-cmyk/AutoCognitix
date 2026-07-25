"""
Vehicle Service with Neo4j primary and PostgreSQL fallback.

Queries Neo4j first for vehicle data. Falls back to PostgreSQL
vehicle_makes/vehicle_models tables if Neo4j returns empty results.
"""

import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from neomodel import db as neomodel_db
from sqlalchemy import case, func, select

if TYPE_CHECKING:  # local-variable annotation only - never evaluated at runtime
    from sqlalchemy.sql.elements import ColumnElement

from app.core.log_sanitizer import sanitize_exception, sanitize_log
from app.core.logging import get_logger
from app.core.sql_utils import escape_ilike
from app.db.postgres.models import (
    DTCCode,
    VehicleComplaint,
    VehicleMake,
    VehicleModel as VehicleModelDB,
)
from app.db.postgres.session import async_session_maker

logger = get_logger(__name__)

# Default cap on the number of ranked common-issue rows returned.
_DEFAULT_COMMON_ISSUES_LIMIT = 20

# Default cap on the number of ranked NHTSA complaint components returned.
_DEFAULT_COMPONENTS_LIMIT = 10

# Label used for complaints whose NHTSA component field is missing/blank. This
# is the same sentinel the flat-file importer writes (scripts/import_flat_complaints.py
# -> `fields[11].strip() or "UNKNOWN"`), so NULL rows fold into the corpus's own
# bucket instead of forming a second, differently-named "unknown" group.
_UNKNOWN_COMPONENT = "UNKNOWN"

# Bare-label complaint aggregation for a vehicle's most common DTC issues.
#
# The real Neo4j graph (26k+ nodes) stores vehicles, complaints and DTCs with
# BARE labels; the loaders never created the legacy
# (:VehicleNode)-[:HAS_COMMON_ISSUE]->(:DTCNode) path with a precomputed
# occurrence_count that the old query ordered by (hence the production 500s).
# Complaints link to DTCs via MENTIONS_DTC (sync_neo4j_sprint9) OR MENTIONED_IN
# (load_all_to_neo4j) - opposite directions - so the relationship match is
# undirected to cover both. count(DISTINCT c) keeps a complaint counted once per
# DTC code even when both loaders linked it (MENTIONS_DTC *and* MENTIONED_IN).
#
# The Complaint node carries make/model/year natively (sync_neo4j_sprint9
# load_complaints), so we filter on it directly instead of hopping through
# (:Vehicle)-[:HAS_COMPLAINT]->. That hop was an avoidable dependency on an
# edge set created LAST in the loader run order, i.e. the first casualty of
# Aura Free's 400K relationship cap. Model uses STARTS WITH because NHTSA
# stores trim-qualified model names ("GOLF GTI", "GOLF R") that an exact match
# would silently drop; the make filter keeps the prefix scoped.
#
# SARGABILITY - why `c.make` is bare and `c.model` is not.
#
# This predicate used to read `toLower(c.make) = toLower($make)`. Wrapping the
# PROPERTY in a function makes the comparison non-sargable: no index on
# Complaint.make can serve it, so every call degenerated into a full label scan
# over the whole Complaint set (~50K nodes today, and the full-corpus DTC
# extraction in sync_neo4j_sprint9 is designed to add tens of thousands more).
#
# The obvious fix - move the function to the parameter side, `c.make =
# toUpper($make)` - was REJECTED after checking every writer of the property.
# Only two of the four upper-case it:
#
#   sync_neo4j_sprint9.load_complaints      make .upper(), model .upper()   OK
#   sync_neo4j_sprint9.normalize_complaint  make .upper(), model .upper()   OK
#   load_all_to_neo4j.load_complaints       make .upper(), model RAW        MIXED
#   load_neo4j_robust.load_complaints       make RAW,      model RAW        MIXED
#
# The raw values are not the flat file's uppercase MAKETXT/MODELTXT: those two
# loaders read the API-sourced dumps written by import_nhtsa_complaints.py /
# sync_nhtsa.py, which stamp every record with the Title-Case make/model from
# their own hardcoded target lists ("Volkswagen", "Mercedes-Benz", "Golf",
# "C-Class"). Both loaders also create the MENTIONED_IN edges this very query
# matches on, so those Title-Case nodes are squarely in scope. `toUpper($model)`
# would silently drop them - the exact failure mode this filter must not have.
#
# So the fix splits by predicate:
#   - `c.make IN $make_variants` - bare property, hence an index SEEK on
#     :Complaint(make). The list holds the stored spellings (upper/title/lower x
#     hyphen/space, see _neo4j_make_variants), so nothing is dropped. make is
#     the selective half, so this alone removes the full label scan.
#   - `toLower(c.model) STARTS WITH toLower($model)` - deliberately left
#     case-folded on the property. Model case CANNOT be enumerated safely
#     (str.title() mangles "RAV4"->"Rav4", "CR-V"->"Cr-V", "GLC"->"Glc"), and a
#     wrong guess here loses rows. It is now only a post-filter over the
#     make-scoped seek result, not over the whole label, so it is cheap.
# Both sides use Cypher's own toLower() so the two operands are folded by the
# identical function - never mix a Python .lower() with a Cypher toLower().
_COMMON_ISSUES_CYPHER = """
    MATCH (c:Complaint)-[:MENTIONS_DTC|MENTIONED_IN]-(d:DTC)
    WHERE c.make IN $make_variants
      AND toLower(c.model) STARTS WITH toLower($model){year_filter}
    RETURN d.code AS code,
           d.description_en AS description_en,
           d.description_hu AS description_hu,
           d.severity AS severity,
           count(DISTINCT c) AS occurrence_count
    ORDER BY occurrence_count DESC, code ASC
    LIMIT $limit
"""

# NHTSA component label -> Hungarian label.
#
# Keys are the EXACT uppercase strings NHTSA ships in FLAT_CMPL COMPDESC (the
# `vehicle_complaints.components` column); lookup is by `value.upper()`.
# Seeded from the real corpus distribution measured at import time
# (data/nhtsa/complaints_flat/flat_import_stats.json `top_components`), which
# covers the overwhelming majority of rows out of ~752 distinct values.
#
# Deliberately partial: an unmapped component yields ``component_hu = None`` so
# the UI can fall back to the raw English label. Never guess a translation -
# a wrong Hungarian component name is worse than an honest English one.
_COMPONENT_LABELS_HU: Dict[str, str] = {
    "AIR BAGS": "Légzsákok",
    "AIR BAGS:FRONTAL": "Légzsákok: frontális",
    "BACK OVER PREVENTION": "Hátramenet-figyelő rendszer",
    "CHILD SEAT": "Gyermekülés",
    "ELECTRICAL SYSTEM": "Elektromos rendszer",
    "ELECTRONIC STABILITY CONTROL (ESC)": "Elektronikus menetstabilizátor (ESC)",
    "ENGINE": "Motor",
    "ENGINE AND ENGINE COOLING": "Motor és motorhűtés",
    "ENGINE AND ENGINE COOLING:ENGINE": "Motor és motorhűtés: motor",
    "EQUIPMENT": "Felszerelés",
    "EXTERIOR LIGHTING": "Külső világítás",
    "EXTERIOR LIGHTING:HEADLIGHTS": "Külső világítás: fényszórók",
    "FORWARD COLLISION AVOIDANCE: AUTOMATIC EMERGENCY BRAKING": (
        "Frontális ütközés elkerülése: automatikus vészfékezés"
    ),
    "FUEL SYSTEM, DIESEL": "Üzemanyagrendszer (dízel)",
    "FUEL SYSTEM, GASOLINE": "Üzemanyagrendszer (benzin)",
    "FUEL/PROPULSION SYSTEM": "Üzemanyag- és hajtásrendszer",
    "LATCHES/LOCKS/LINKAGES": "Zárak, reteszek, csuklópontok",
    "PARKING BRAKE": "Rögzítőfék",
    "POWER TRAIN": "Hajtáslánc",
    "POWER TRAIN:AUTOMATIC TRANSMISSION": "Hajtáslánc: automata váltó",
    "POWER TRAIN:MANUAL TRANSMISSION": "Hajtáslánc: kézi váltó",
    "SEAT BELTS": "Biztonsági övek",
    "SEATS": "Ülések",
    "SERVICE BRAKES": "Üzemi fék",
    "SERVICE BRAKES, AIR": "Üzemi fék (levegős)",
    "SERVICE BRAKES, HYDRAULIC": "Üzemi fék (hidraulikus)",
    "STEERING": "Kormányzás",
    "STRUCTURE": "Karosszéria és váz",
    "STRUCTURE:BODY": "Karosszéria",
    "SUSPENSION": "Futómű",
    "TIRES": "Gumiabroncsok",
    "TRAILER HITCHES": "Vonóhorog",
    _UNKNOWN_COMPONENT: "Ismeretlen",
    "UNKNOWN OR OTHER": "Ismeretlen vagy egyéb",
    "VEHICLE SPEED CONTROL": "Sebességszabályozás",
    "VISIBILITY/WIPER": "Látási viszonyok / ablaktörlő",
    "WHEELS": "Kerekek",
}


def _make_spelling_variants(make: str) -> List[str]:
    """Lowercased NHTSA `make` spellings to match a caller-supplied make against.

    NHTSA's own corpus is not internally consistent: the flat file ships BOTH
    ``MERCEDES-BENZ`` (17,347 rows) and ``MERCEDES BENZ`` (11,134 rows) as
    separate makes. A single equality on the canonical hyphenated spelling
    would silently drop 39% of that brand's complaints.

    Returns 1-3 lowercase variants (hyphen/space interchange, de-duplicated,
    order-stable). Kept as an ``IN`` list rather than a ``LIKE``/regex so the
    planner can still use the ``lower(make)`` index (BitmapOr of equality
    scans).
    """
    normalized = " ".join(make.strip().lower().split())
    variants = [normalized, normalized.replace("-", " "), normalized.replace(" ", "-")]
    return list(dict.fromkeys(v for v in variants if v))


def _neo4j_make_variants(make: str) -> List[str]:
    """Stored ``Complaint.make`` spellings to seek for a caller-supplied make.

    Neo4j's ``Complaint.make`` is not consistently normalised - two of the four
    loaders that write it store the Title-Case make from their hardcoded target
    lists instead of NHTSA's uppercase MAKETXT (see the SARGABILITY note on
    ``_COMMON_ISSUES_CYPHER``). Enumerating the spellings on the PARAMETER side
    is what lets the query compare a bare ``c.make`` - an index seek - without
    dropping the Title-Case rows that ``toUpper($make)`` would miss.

    Layered on :func:`_make_spelling_variants` so the NHTSA hyphen/space
    knowledge (``MERCEDES-BENZ`` 17,347 rows vs ``MERCEDES BENZ`` 11,134 rows -
    39% of the brand) lives in exactly ONE place and the Neo4j and PostgreSQL
    rankings agree on which spellings belong to a brand.

    Case coverage is enumerated, not guessed: every make either comes from the
    uppercase flat file or from one of the 20 Title-Case constants in
    ``import_nhtsa_complaints.TOP_MAKES`` / ``sync_nhtsa.POPULAR_MAKES``, and
    ``{upper, title}`` reproduces all 20 exactly (``.upper()`` covers "BMW",
    which ``.title()`` would mangle to "Bmw"). The caller's own spelling is kept
    as a further fallback. Bounded at 3 spellings x 3 cases + 1, and it collapses
    to 3 entries for a single-spelling make like "Volkswagen".

    Returns ``[]`` for a blank make, which makes ``c.make IN []`` match nothing.
    """
    as_typed = " ".join(make.strip().split())
    variants: List[str] = [as_typed] if as_typed else []
    for lowered in _make_spelling_variants(make):
        variants.extend((lowered.upper(), lowered.title(), lowered))
    return list(dict.fromkeys(variants))


class VehicleService:
    """
    Service for querying vehicle data.

    Primary: Neo4j VehicleNode
    Fallback: PostgreSQL vehicle_makes/vehicle_models (seeded via migration 010)
    """

    async def get_all_makes(
        self,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
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
        search: Optional[str],
        limit: int,
        offset: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
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
        search: Optional[str],
        limit: int,
        offset: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Get makes from PostgreSQL vehicle_makes table."""
        async with async_session_maker() as session:
            stmt = select(VehicleMake)
            count_stmt = select(func.count(VehicleMake.id))

            if search:
                escaped_search = escape_ilike(search)
                stmt = stmt.where(VehicleMake.name.ilike(f"%{escaped_search}%"))
                count_stmt = count_stmt.where(VehicleMake.name.ilike(f"%{escaped_search}%"))

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
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
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
        search: Optional[str],
        limit: int,
        offset: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
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
        search: Optional[str],
        limit: int,
        offset: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Get models from PostgreSQL vehicle_models table."""
        make_id = make.lower().replace(" ", "_").replace("-", "_")

        async with async_session_maker() as session:
            make_obj_result = await session.execute(
                select(VehicleMake).where(
                    (VehicleMake.id == make_id) | (VehicleMake.name.ilike(escape_ilike(make)))
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
                escaped_search = escape_ilike(search)
                stmt = stmt.where(VehicleModelDB.name.ilike(f"%{escaped_search}%"))
                count_stmt = count_stmt.where(VehicleModelDB.name.ilike(f"%{escaped_search}%"))

            stmt = stmt.order_by(VehicleModelDB.name).offset(offset).limit(limit)

            result = await session.execute(stmt)
            count_result = await session.execute(count_stmt)

            total = count_result.scalar() or 0
            models: List[Dict[str, Any]] = [
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
    ) -> List[int]:
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

        years: Set[int] = set()
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
                        (VehicleModelDB.make_id == make_id)
                        & (VehicleModelDB.name.ilike(escape_ilike(model)))
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
    ) -> Optional[Dict[str, Any]]:
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
        year: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
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
        year: Optional[int] = None,
        limit: int = _DEFAULT_COMMON_ISSUES_LIMIT,
    ) -> Optional[List[Dict[str, Any]]]:
        """Get the most common DTC issues for a vehicle from the complaint graph.

        Aggregates the DTC codes mentioned across NHTSA complaints linked to the
        given make/model (optionally filtered by complaint year) and ranks them by
        how many distinct complaints mention each code. Descriptions and severity
        are enriched from the curated PostgreSQL DTC table when available, falling
        back to the graph node's own properties.

        Degrades gracefully on any Neo4j/driver error - this endpoint must never
        surface a 500 - but the two "nothing to show" outcomes are DISTINCT in the
        return value, not merely in the logs:

        - ``[]``  the graph answered and has no matching row (genuine absence),
          logged at INFO with ``rows=0`` by ``_get_common_issues_neo4j``.
        - ``None`` the graph did not answer (outage, wrong NEO4J_URI/credentials,
          Aura free-tier auto-pause), logged at ERROR with the marker
          ``common-issues neo4j QUERY FAILED``.

        Never collapse the two: the caller turns ``None`` into an explicit
        ``sources.issues = "unavailable"`` on the wire, which is what stops the UI
        from presenting an outage as a factual absence of data.
        """
        make_clean = make.strip()
        model_clean = model.strip()
        if not make_clean or not model_clean:
            # Same guard as the PostgreSQL sibling below, for the same reason:
            # Cypher's `'golf gti' STARTS WITH ''` is true for EVERY row, so a
            # blank model would rank the make's ENTIRE complaint history under the
            # user's chosen model (and run an unbounded aggregation on the graph).
            # A blank make is already inert (`_neo4j_make_variants("")` -> `[]`),
            # but is rejected here too so both halves fail the same way.
            # Genuine absence, not an outage: `[]`, never `None`.
            return []

        try:
            rows = await self._get_common_issues_neo4j(make_clean, model_clean, year, limit)
        except Exception as e:
            logger.error(
                "common-issues neo4j QUERY FAILED (graph unreachable or query invalid) "
                "make=%s model=%s year=%s error=%s: %s",
                sanitize_log(make_clean),
                sanitize_log(model_clean),
                sanitize_log(str(year)),
                type(e).__name__,
                sanitize_exception(e),
            )
            return None

        if not rows:
            return []

        try:
            return await self._enrich_common_issues(rows)
        except Exception as e:
            logger.warning(
                "PostgreSQL enrichment for common issues failed, using graph data only: %s",
                sanitize_exception(e),
            )
            return [self._issue_from_graph_row(row) for row in rows]

    async def _get_common_issues_neo4j(
        self,
        make: str,
        model: str,
        year: Optional[int],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Run the bare-label complaint aggregation against Neo4j.

        See ``_COMMON_ISSUES_CYPHER`` for the schema and sargability rationale.
        make/model/year are all filtered on the ``Complaint`` node, which carries
        them natively. ``make`` is expanded into its stored spellings here rather
        than case-folded in Cypher, so the property side stays bare and the
        predicate can use the :Complaint(make) index.

        The year filter is a STATIC template fragment; the user-supplied year is
        bound as the ``$year`` parameter. No caller-controlled value is ever
        interpolated into the Cypher string - keep it that way. The make variants
        are derived from the caller's value but likewise travel as a bound
        parameter, never as query text.

        Raises on any driver/query error so the caller can log it distinctly from
        a genuine empty result.
        """
        params: Dict[str, Any] = {
            "make_variants": _neo4j_make_variants(make),
            "model": model,
            "limit": limit,
        }
        year_filter = ""
        if year is not None:
            year_filter = "\n      AND c.year = $year"
            params["year"] = year

        query = _COMMON_ISSUES_CYPHER.format(year_filter=year_filter)
        results, _ = await asyncio.to_thread(neomodel_db.cypher_query, query, params)

        rows: List[Dict[str, Any]] = []
        for row in results or []:
            code = row[0]
            if not code:  # skip malformed / null DTC codes (code is required downstream)
                continue
            rows.append(
                {
                    "code": code,
                    "description_en": row[1],
                    "description_hu": row[2],
                    "severity": row[3],
                    "occurrence_count": row[4],
                }
            )

        # Success marker - `rows=0` here means the graph genuinely has no match,
        # NOT that Neo4j is down (that path logs ERROR in the caller).
        logger.info(
            "common-issues neo4j OK: make=%s model=%s year=%s rows=%d",
            sanitize_log(make),
            sanitize_log(model),
            sanitize_log(str(year)),
            len(rows),
        )
        return rows

    async def _enrich_common_issues(
        self,
        rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Enrich graph aggregation rows with curated PostgreSQL DTC data.

        Preserves the graph's occurrence-based ranking. Curated PostgreSQL values
        take precedence; the graph node's own properties are used as a fallback for
        codes absent from the DTC table (the graph carries many more codes than the
        curated set). Mirrors the "PostgreSQL is the DTC source of truth, Neo4j
        supplies the graph relationships" pattern used by the DTC detail endpoint.
        """
        codes = [row["code"] for row in rows if row.get("code")]
        curated: Dict[str, DTCCode] = {}
        if codes:
            async with async_session_maker() as session:
                result = await session.execute(select(DTCCode).where(DTCCode.code.in_(codes)))
                curated = {dtc.code: dtc for dtc in result.scalars().all()}

        issues: List[Dict[str, Any]] = []
        for row in rows:
            code = row.get("code")
            if not code:
                continue
            dtc = curated.get(code)
            occurrence_count = row.get("occurrence_count")
            issues.append(
                {
                    "code": code,
                    "description_en": (dtc.description_en if dtc else None)
                    or row.get("description_en"),
                    "description_hu": (dtc.description_hu if dtc else None)
                    or row.get("description_hu"),
                    "severity": (dtc.severity if dtc else None) or row.get("severity"),
                    "frequency": self._frequency_bucket(occurrence_count),
                    "occurrence_count": occurrence_count,
                }
            )
        return issues

    async def get_vehicle_complaint_components(
        self,
        make: str,
        model: str,
        year: Optional[int] = None,
        limit: int = _DEFAULT_COMPONENTS_LIMIT,
    ) -> Tuple[Optional[List[Dict[str, Any]]], int]:
        """Rank a vehicle's NHTSA complaint components by report frequency.

        This is the *real* common-issues signal. The DTC-based ranking above can
        only see complaints whose free-text narrative literally quotes a fault
        code, which consumers essentially never do - the whole graph holds ~107
        complaint->DTC edges. Component frequency, by contrast, is a first-class
        NHTSA field present on every one of the imported complaint rows, so it
        actually has data to show.

        Returns ``(components, total_complaints)`` where ``total_complaints`` is
        the denominator across ALL component groups for this vehicle (not just
        the ``limit`` returned ones), so ``share`` stays meaningful and a caller
        can tell "no data for this vehicle" from "data exists, list truncated".

        Degrades gracefully - this endpoint must never surface a 500 - and, like
        its Neo4j sibling, keeps the two "nothing to show" outcomes DISTINCT in
        the return value rather than only in the logs:

        - ``([], 0)``   PostgreSQL answered and holds no complaint for this
          vehicle (genuine absence), logged at INFO with ``rows=0 total=0``.
        - ``(None, 0)`` PostgreSQL did not answer, logged at ERROR with the
          marker ``common-issues components QUERY FAILED``.

        Never collapse the two: the caller turns ``None`` into an explicit
        ``sources.components = "unavailable"`` on the wire.
        """
        make_clean = make.strip()
        model_clean = model.strip()
        if not make_clean or not model_clean:
            # A blank make/model is not a vehicle. Guarded explicitly because an
            # empty model would otherwise build the pattern "%" and aggregate the
            # make's ENTIRE complaint history under the user's chosen model.
            # Genuine absence, not an outage: `[]`, never `None`.
            return [], 0

        try:
            rows = await self._query_complaint_components(make_clean, model_clean, year, limit)
        except Exception as e:
            logger.error(
                "common-issues components QUERY FAILED (PostgreSQL unreachable or query "
                "invalid) make=%s model=%s year=%s error=%s: %s",
                sanitize_log(make_clean),
                sanitize_log(model_clean),
                sanitize_log(str(year)),
                type(e).__name__,
                sanitize_exception(e),
            )
            return None, 0

        total = int(rows[0]["total_complaints"] or 0) if rows else 0

        components: List[Dict[str, Any]] = []
        for row in rows:
            component = row["component"]
            count = int(row["complaint_count"] or 0)
            components.append(
                {
                    "component": component,
                    "component_hu": _COMPONENT_LABELS_HU.get(component.upper()),
                    "complaint_count": count,
                    # Guarded division: `total` can only be 0 when there are no
                    # rows at all, but a 0 denominator must never raise here.
                    "share": round(count / total, 4) if total else 0.0,
                    "crash_count": int(row["crash_count"] or 0),
                    "fire_count": int(row["fire_count"] or 0),
                    "injury_count": int(row["injury_count"] or 0),
                    "death_count": int(row["death_count"] or 0),
                }
            )

        # Success marker - `rows=0` here means PostgreSQL genuinely has no
        # complaint for this vehicle, NOT that the DB is down (that path logs
        # ERROR above).
        logger.info(
            "common-issues components OK: make=%s model=%s year=%s rows=%d total=%d",
            sanitize_log(make_clean),
            sanitize_log(model_clean),
            sanitize_log(str(year)),
            len(components),
            total,
        )
        return components, total

    async def _query_complaint_components(
        self,
        make: str,
        model: str,
        year: Optional[int],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Single grouped aggregation over ``vehicle_complaints``.

        Emits one statement, executed once on one AsyncSession - concurrent
        ``session.execute()`` calls on a single AsyncSession raise InterfaceError,
        so the grand total is carried by a window function on the SAME query
        instead of a second round trip::

            SELECT COALESCE(NULLIF(TRIM(components), ''), 'UNKNOWN') AS component,
                   count(*)                                          AS complaint_count,
                   COALESCE(sum(CASE WHEN crash THEN 1 ELSE 0 END), 0) AS crash_count,
                   COALESCE(sum(CASE WHEN fire  THEN 1 ELSE 0 END), 0) AS fire_count,
                   COALESCE(sum(injuries), 0)                        AS injury_count,
                   COALESCE(sum(deaths), 0)                          AS death_count,
                   sum(count(*)) OVER ()                             AS total_complaints
              FROM vehicle_complaints
             WHERE lower(make) IN (:make_variants)
               AND lower(model) LIKE :model_prefix ESCAPE '\\'
               [AND model_year = :year]
             GROUP BY 1
             ORDER BY complaint_count DESC, component ASC
             LIMIT :limit

        ``sum(count(*)) OVER ()`` is evaluated after GROUP BY but before
        ORDER BY/LIMIT, so it is the total across every component group, not
        just the returned page.

        Matching notes (verified against the live NHTSA corpus, not assumed):
        - Stored values are UPPERCASE (``VOLKSWAGEN``, ``GOLF GTI``) while
          callers pass user spellings, hence ``lower()`` on both sides. It is
          ``lower(col) = ...`` / ``lower(col) LIKE ...`` rather than ``ilike``
          precisely so the ``lower(make), lower(model)`` functional index
          (migration 020) is usable - ``ILIKE`` can never use a btree index.
        - The model is a PREFIX match because NHTSA stores trim-qualified names:
          ``GOLF`` -> {GOLF, GOLF GTI, GOLF R, GOLF SPORTWAGEN}, and the app's
          seeded ``GLC``/``XC60`` only reach {GLC-CLASS...}/{XC60 T5, T6, T8}
          this way. Exact matching returns zero rows for those. The make
          equality keeps the prefix scoped to one brand.
        - ``escape_ilike`` neutralises ``%``/``_``/``\\`` in the user value even
          though the endpoint's parameter regex already rejects them; the escape
          character is passed explicitly because SQLite (test harness) has no
          default LIKE escape while PostgreSQL does.
        """
        component_expr = func.coalesce(
            func.nullif(func.trim(VehicleComplaint.components), ""), _UNKNOWN_COMPONENT
        )
        complaint_count = func.count().label("complaint_count")

        # Explicitly annotated: mypy would otherwise narrow the list to
        # BinaryExpression from the first element and reject the year clause.
        conditions: List[ColumnElement[bool]] = [
            func.lower(VehicleComplaint.make).in_(_make_spelling_variants(make)),
            func.lower(VehicleComplaint.model).like(f"{escape_ilike(model.lower())}%", escape="\\"),
        ]
        if year is not None:
            conditions.append(VehicleComplaint.model_year == year)

        stmt = (
            select(
                component_expr.label("component"),
                complaint_count,
                func.coalesce(
                    func.sum(case((VehicleComplaint.crash.is_(True), 1), else_=0)), 0
                ).label("crash_count"),
                func.coalesce(
                    func.sum(case((VehicleComplaint.fire.is_(True), 1), else_=0)), 0
                ).label("fire_count"),
                func.coalesce(func.sum(VehicleComplaint.injuries), 0).label("injury_count"),
                func.coalesce(func.sum(VehicleComplaint.deaths), 0).label("death_count"),
                func.sum(func.count()).over().label("total_complaints"),
            )
            .where(*conditions)
            .group_by(component_expr)
            .order_by(complaint_count.desc(), component_expr.asc())
            .limit(limit)
        )

        async with async_session_maker() as session:
            result = await session.execute(stmt)
            return [dict(row) for row in result.mappings().all()]

    def _issue_from_graph_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Map a raw graph aggregation row to a VehicleCommonIssue dict (no enrichment)."""
        occurrence_count = row.get("occurrence_count")
        return {
            "code": row["code"],
            "description_en": row.get("description_en"),
            "description_hu": row.get("description_hu"),
            "severity": row.get("severity"),
            "frequency": self._frequency_bucket(occurrence_count),
            "occurrence_count": occurrence_count,
        }

    @staticmethod
    def _frequency_bucket(occurrence_count: Optional[int]) -> str:
        """Derive a qualitative frequency label from the real occurrence count.

        Uses the same vocabulary as the historical ``HasCommonIssueRel.frequency``
        property (rare / uncommon / common / very_common). This is a truthful
        summary of the aggregated complaint counts, not a fabricated value.
        """
        count = occurrence_count or 0
        if count >= 20:
            return "very_common"
        if count >= 5:
            return "common"
        if count >= 2:
            return "uncommon"
        return "rare"

    def _node_to_dict(self, node: Any) -> Dict[str, Any]:
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
    def _get_country_for_make(make: str) -> Optional[str]:
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
_vehicle_service: Optional[VehicleService] = None


def get_vehicle_service() -> VehicleService:
    """Get or create vehicle service instance."""
    global _vehicle_service
    if _vehicle_service is None:
        _vehicle_service = VehicleService()
    return _vehicle_service
