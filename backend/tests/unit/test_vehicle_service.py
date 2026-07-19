"""
Unit tests for VehicleService.

Mocks Neo4j (neomodel_db) and PostgreSQL (async_session_maker) to test
all service methods without real database connections.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.vehicle_service import VehicleService, get_vehicle_service


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def service():
    """Fresh VehicleService instance for each test."""
    return VehicleService()


# ---------------------------------------------------------------------------
# Helper: mock neomodel_db.cypher_query via asyncio.to_thread
# ---------------------------------------------------------------------------


def _patch_neo4j(side_effect=None, return_value=None):
    """Patch asyncio.to_thread so that neomodel_db.cypher_query calls are intercepted."""
    if side_effect is not None:
        return patch("asyncio.to_thread", new_callable=lambda: AsyncMock, side_effect=side_effect)
    return patch(
        "asyncio.to_thread",
        new_callable=lambda: AsyncMock,
        return_value=return_value,
    )


# ===========================================================================
# _get_country_for_make
# ===========================================================================


class TestGetCountryForMake:
    def test_known_german_make(self):
        assert VehicleService._get_country_for_make("Volkswagen") == "Germany"

    def test_known_japanese_make(self):
        assert VehicleService._get_country_for_make("Toyota") == "Japan"

    def test_known_korean_make(self):
        assert VehicleService._get_country_for_make("Hyundai") == "South Korea"

    def test_known_us_make(self):
        assert VehicleService._get_country_for_make("Ford") == "USA"

    def test_known_french_make(self):
        assert VehicleService._get_country_for_make("Renault") == "France"

    def test_known_italian_make(self):
        assert VehicleService._get_country_for_make("Fiat") == "Italy"

    def test_known_uk_make(self):
        assert VehicleService._get_country_for_make("Jaguar") == "UK"

    def test_known_czech_make(self):
        assert VehicleService._get_country_for_make("Skoda") == "Czech Republic"

    def test_known_swedish_make(self):
        assert VehicleService._get_country_for_make("Volvo") == "Sweden"

    def test_known_chinese_make(self):
        assert VehicleService._get_country_for_make("BYD") == "China"

    def test_unknown_make_returns_none(self):
        assert VehicleService._get_country_for_make("UnknownBrand") is None

    def test_case_insensitive(self):
        assert VehicleService._get_country_for_make("VOLKSWAGEN") == "Germany"
        assert VehicleService._get_country_for_make("volkswagen") == "Germany"


# ===========================================================================
# _node_to_dict
# ===========================================================================


class TestNodeToDict:
    def test_dict_node(self, service):
        node = {
            "uid": "abc-123",
            "make": "Toyota",
            "model": "Corolla",
            "year_start": 2015,
            "year_end": 2020,
            "platform": "E170",
            "engine_codes": ["1ZR-FE"],
            "body_types": ["sedan"],
        }
        result = service._node_to_dict(node)
        assert result["id"] == "abc-123"
        assert result["make"] == "Toyota"
        assert result["model"] == "Corolla"
        assert result["year_start"] == 2015
        assert result["year_end"] == 2020
        assert result["platform"] == "E170"
        assert result["engine_codes"] == ["1ZR-FE"]
        assert result["body_types"] == ["sedan"]

    def test_dict_node_missing_optional_fields(self, service):
        node = {"uid": "x", "make": "BMW", "model": "3"}
        result = service._node_to_dict(node)
        assert result["id"] == "x"
        assert result["year_start"] is None
        assert result["engine_codes"] == []
        assert result["body_types"] == []

    def test_object_node_with_get(self, service):
        """Node that has __dict__ AND .get() (like a Neo4j Node)."""
        data = {
            "uid": "obj-1",
            "make": "Audi",
            "model": "A4",
            "year_start": 2016,
            "year_end": 2023,
            "platform": "B9",
            "engine_codes": ["CVNA"],
            "body_types": ["sedan", "wagon"],
        }

        class Neo4jLikeNode:
            def __init__(self, d):
                self._data = d

            def get(self, key, default=None):
                return self._data.get(key, default)

        node = Neo4jLikeNode(data)
        result = service._node_to_dict(node)
        assert result["id"] == "obj-1"
        assert result["make"] == "Audi"
        assert result["body_types"] == ["sedan", "wagon"]

    def test_object_node_without_get(self, service):
        """Node that has __dict__ but no .get() method."""

        class SimpleNode:
            uid = "sn-1"
            make = "Honda"
            model = "Civic"
            year_start = 2020
            year_end = None
            platform = None
            engine_codes = []
            body_types = ["hatchback"]

        # SimpleNode has __dict__ but no .get()
        result = service._node_to_dict(SimpleNode())
        assert result["id"] == "sn-1"
        assert result["make"] == "Honda"
        assert result["model"] == "Civic"
        assert result["body_types"] == ["hatchback"]

    def test_non_dict_non_object_returns_empty(self, service):
        """A primitive or object without __dict__ returns {}."""
        assert service._node_to_dict(42) == {}
        assert service._node_to_dict("string") == {}


# ===========================================================================
# get_all_makes  (Neo4j path)
# ===========================================================================


class TestGetAllMakes:
    @pytest.mark.asyncio
    async def test_neo4j_returns_makes_no_search(self, service):
        """Neo4j returns results -> skip PostgreSQL."""
        call_count = 0

        async def _to_thread(fn, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # main query
                return [["Volkswagen"], ["Toyota"]], None
            else:
                # count query
                return [[2]], None

        with patch("asyncio.to_thread", side_effect=_to_thread):
            makes, total = await service.get_all_makes()

        assert total == 2
        assert len(makes) == 2
        assert makes[0]["name"] == "Volkswagen"
        assert makes[0]["country"] == "Germany"
        assert makes[1]["name"] == "Toyota"
        assert makes[1]["country"] == "Japan"
        assert makes[0]["id"] == "volkswagen"

    @pytest.mark.asyncio
    async def test_neo4j_returns_makes_with_search(self, service):
        call_count = 0

        async def _to_thread(fn, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [["Volkswagen"]], None
            else:
                return [[1]], None

        with patch("asyncio.to_thread", side_effect=_to_thread):
            makes, total = await service.get_all_makes(search="volks")

        assert total == 1
        assert makes[0]["name"] == "Volkswagen"

    @pytest.mark.asyncio
    async def test_neo4j_empty_falls_back_to_postgres(self, service):
        """Neo4j returns 0 results -> call PostgreSQL fallback."""
        call_count = 0

        async def _to_thread(fn, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [], None
            else:
                return [[0]], None

        mock_make = MagicMock()
        mock_make.id = "ford"
        mock_make.name = "Ford"
        mock_make.country = "USA"

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_make]

        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=[mock_result, mock_count_result])

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("asyncio.to_thread", side_effect=_to_thread),
            patch(
                "app.services.vehicle_service.async_session_maker",
                return_value=mock_session_ctx,
            ),
        ):
            makes, total = await service.get_all_makes()

        assert total == 1
        assert makes[0]["name"] == "Ford"
        assert makes[0]["country"] == "USA"

    @pytest.mark.asyncio
    async def test_neo4j_exception_falls_back_to_postgres(self, service):
        """Neo4j raises -> fall back to PostgreSQL."""
        mock_make = MagicMock()
        mock_make.id = "bmw"
        mock_make.name = "BMW"
        mock_make.country = "Germany"

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_make]

        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=[mock_result, mock_count_result])

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("asyncio.to_thread", side_effect=Exception("Neo4j down")),
            patch(
                "app.services.vehicle_service.async_session_maker",
                return_value=mock_session_ctx,
            ),
        ):
            makes, total = await service.get_all_makes()

        assert total == 1
        assert makes[0]["name"] == "BMW"

    @pytest.mark.asyncio
    async def test_postgres_fallback_with_search(self, service):
        """PostgreSQL fallback respects the search parameter."""
        call_count = 0

        async def _to_thread(fn, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [], None
            else:
                return [[0]], None

        mock_make = MagicMock()
        mock_make.id = "audi"
        mock_make.name = "Audi"
        mock_make.country = "Germany"

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_make]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=[mock_result, mock_count_result])

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("asyncio.to_thread", side_effect=_to_thread),
            patch(
                "app.services.vehicle_service.async_session_maker",
                return_value=mock_session_ctx,
            ),
        ):
            makes, total = await service.get_all_makes(search="aud")

        assert total == 1
        assert makes[0]["name"] == "Audi"


# ===========================================================================
# get_models_for_make
# ===========================================================================


class TestGetModelsForMake:
    @pytest.mark.asyncio
    async def test_neo4j_returns_models_no_search(self, service):
        call_count = 0

        async def _to_thread(fn, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [["Golf", 2012, 2024, [["hatchback", "wagon"]]]], None
            else:
                return [[1]], None

        with patch("asyncio.to_thread", side_effect=_to_thread):
            models, total = await service.get_models_for_make("Volkswagen")

        assert total == 1
        assert models[0]["name"] == "Golf"
        assert models[0]["make_id"] == "volkswagen"
        assert models[0]["year_start"] == 2012
        assert "hatchback" in models[0]["body_types"]

    @pytest.mark.asyncio
    async def test_neo4j_models_with_search(self, service):
        call_count = 0

        async def _to_thread(fn, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [["3 Series", 2018, 2025, ["sedan"]]], None
            else:
                return [[1]], None

        with patch("asyncio.to_thread", side_effect=_to_thread):
            models, total = await service.get_models_for_make("BMW", search="3")

        assert total == 1
        assert models[0]["name"] == "3 Series"

    @pytest.mark.asyncio
    async def test_neo4j_body_types_flatten_nested_lists(self, service):
        """Body types may come as nested lists from Neo4j COLLECT."""
        call_count = 0

        async def _to_thread(fn, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [
                    ["A4", 2016, 2023, [["sedan", "wagon"], "convertible"]],
                ], None
            else:
                return [[1]], None

        with patch("asyncio.to_thread", side_effect=_to_thread):
            models, _total = await service.get_models_for_make("Audi")

        body = models[0]["body_types"]
        assert "sedan" in body
        assert "wagon" in body
        assert "convertible" in body

    @pytest.mark.asyncio
    async def test_neo4j_empty_falls_back_to_postgres(self, service):
        call_count = 0

        async def _to_thread(fn, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [], None
            else:
                return [[0]], None

        mock_make_obj = MagicMock()
        mock_make_obj.id = "ford"

        mock_make_result = MagicMock()
        mock_make_result.scalar_one_or_none.return_value = mock_make_obj

        mock_model = MagicMock()
        mock_model.id = "focus"
        mock_model.name = "Focus"
        mock_model.make_id = "ford"
        mock_model.year_start = 2018
        mock_model.year_end = 2025
        mock_model.body_types = []

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_model]
        mock_models_result = MagicMock()
        mock_models_result.scalars.return_value = mock_scalars

        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            side_effect=[mock_make_result, mock_models_result, mock_count_result]
        )

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("asyncio.to_thread", side_effect=_to_thread),
            patch(
                "app.services.vehicle_service.async_session_maker",
                return_value=mock_session_ctx,
            ),
        ):
            models, total = await service.get_models_for_make("Ford")

        assert total == 1
        assert models[0]["name"] == "Focus"

    @pytest.mark.asyncio
    async def test_postgres_make_not_found_returns_empty(self, service):
        """PostgreSQL fallback: make not found in DB."""
        call_count = 0

        async def _to_thread(fn, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [], None
            else:
                return [[0]], None

        mock_make_result = MagicMock()
        mock_make_result.scalar_one_or_none.return_value = None

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_make_result)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("asyncio.to_thread", side_effect=_to_thread),
            patch(
                "app.services.vehicle_service.async_session_maker",
                return_value=mock_session_ctx,
            ),
        ):
            models, total = await service.get_models_for_make("NonExistent")

        assert total == 0
        assert models == []

    @pytest.mark.asyncio
    async def test_neo4j_exception_falls_back_to_postgres(self, service):
        mock_make_result = MagicMock()
        mock_make_result.scalar_one_or_none.return_value = None

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_make_result)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("asyncio.to_thread", side_effect=Exception("Neo4j error")),
            patch(
                "app.services.vehicle_service.async_session_maker",
                return_value=mock_session_ctx,
            ),
        ):
            models, total = await service.get_models_for_make("Honda")

        assert total == 0
        assert models == []

    @pytest.mark.asyncio
    async def test_neo4j_body_types_filters_falsy_values(self, service):
        """Falsy body_type entries (None, '') should be skipped."""
        call_count = 0

        async def _to_thread(fn, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [["Polo", 2017, 2024, ["hatchback", None, "", "sedan"]]], None
            else:
                return [[1]], None

        with patch("asyncio.to_thread", side_effect=_to_thread):
            models, _total = await service.get_models_for_make("Volkswagen")

        body = models[0]["body_types"]
        assert "hatchback" in body
        assert "sedan" in body
        assert None not in body
        assert "" not in body

    @pytest.mark.asyncio
    async def test_postgres_fallback_with_search(self, service):
        """PostgreSQL models fallback respects the search parameter."""
        call_count = 0

        async def _to_thread(fn, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [], None
            else:
                return [[0]], None

        mock_make_obj = MagicMock()
        mock_make_obj.id = "toyota"

        mock_make_result = MagicMock()
        mock_make_result.scalar_one_or_none.return_value = mock_make_obj

        mock_model = MagicMock()
        mock_model.id = "camry"
        mock_model.name = "Camry"
        mock_model.make_id = "toyota"
        mock_model.year_start = 2018
        mock_model.year_end = 2024
        mock_model.body_types = []

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_model]
        mock_models_result = MagicMock()
        mock_models_result.scalars.return_value = mock_scalars

        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            side_effect=[mock_make_result, mock_models_result, mock_count_result]
        )

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("asyncio.to_thread", side_effect=_to_thread),
            patch(
                "app.services.vehicle_service.async_session_maker",
                return_value=mock_session_ctx,
            ),
        ):
            models, total = await service.get_models_for_make("Toyota", search="cam")

        assert total == 1
        assert models[0]["name"] == "Camry"


# ===========================================================================
# get_years_for_vehicle
# ===========================================================================


class TestGetYearsForVehicle:
    @pytest.mark.asyncio
    async def test_neo4j_returns_year_range(self, service):
        async def _to_thread(fn, *args, **kwargs):
            return [[2018, 2022]], None

        with patch("asyncio.to_thread", side_effect=_to_thread):
            years = await service.get_years_for_vehicle("Toyota", "Corolla")

        assert years == [2022, 2021, 2020, 2019, 2018]

    @pytest.mark.asyncio
    async def test_neo4j_no_year_end_uses_current_year(self, service):
        async def _to_thread(fn, *args, **kwargs):
            return [[2024, None]], None

        with patch("asyncio.to_thread", side_effect=_to_thread):
            years = await service.get_years_for_vehicle("BMW", "3 Series")

        assert 2026 in years
        assert 2024 in years

    @pytest.mark.asyncio
    async def test_neo4j_empty_falls_back_to_postgres(self, service):
        async def _to_thread(fn, *args, **kwargs):
            return [], None

        mock_model = MagicMock()
        mock_model.year_start = 2020
        mock_model.year_end = 2023

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_model]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("asyncio.to_thread", side_effect=_to_thread),
            patch(
                "app.services.vehicle_service.async_session_maker",
                return_value=mock_session_ctx,
            ),
        ):
            years = await service.get_years_for_vehicle("Ford", "Focus")

        assert years == [2023, 2022, 2021, 2020]

    @pytest.mark.asyncio
    async def test_neo4j_exception_falls_back_to_postgres(self, service):
        async def _to_thread(fn, *args, **kwargs):
            raise Exception("Neo4j error")

        mock_model = MagicMock()
        mock_model.year_start = 2019
        mock_model.year_end = 2021

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_model]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("asyncio.to_thread", side_effect=_to_thread),
            patch(
                "app.services.vehicle_service.async_session_maker",
                return_value=mock_session_ctx,
            ),
        ):
            years = await service.get_years_for_vehicle("Ford", "Focus")

        assert years == [2021, 2020, 2019]

    @pytest.mark.asyncio
    async def test_postgres_fallback_no_year_end(self, service):
        """PostgreSQL model with no year_end -> uses current_year (2026)."""

        async def _to_thread(fn, *args, **kwargs):
            return [], None

        mock_model = MagicMock()
        mock_model.year_start = 2024
        mock_model.year_end = None

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_model]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("asyncio.to_thread", side_effect=_to_thread),
            patch(
                "app.services.vehicle_service.async_session_maker",
                return_value=mock_session_ctx,
            ),
        ):
            years = await service.get_years_for_vehicle("Kia", "Sportage")

        assert 2026 in years
        assert 2024 in years

    @pytest.mark.asyncio
    async def test_postgres_fallback_also_fails(self, service):
        """Both Neo4j and PostgreSQL fail -> empty list."""

        async def _to_thread(fn, *args, **kwargs):
            return [], None

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(side_effect=Exception("DB down"))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("asyncio.to_thread", side_effect=_to_thread),
            patch(
                "app.services.vehicle_service.async_session_maker",
                return_value=mock_session_ctx,
            ),
        ):
            years = await service.get_years_for_vehicle("X", "Y")

        assert years == []

    @pytest.mark.asyncio
    async def test_multiple_neo4j_rows_merge_years(self, service):
        async def _to_thread(fn, *args, **kwargs):
            return [[2015, 2018], [2019, 2022]], None

        with patch("asyncio.to_thread", side_effect=_to_thread):
            years = await service.get_years_for_vehicle("VW", "Golf")

        assert years == list(range(2022, 2014, -1))

    @pytest.mark.asyncio
    async def test_neo4j_row_null_year_start_skipped(self, service):
        async def _to_thread(fn, *args, **kwargs):
            return [[None, 2020]], None

        with patch("asyncio.to_thread", side_effect=_to_thread):
            years = await service.get_years_for_vehicle("X", "Y")

        assert years == []

    @pytest.mark.asyncio
    async def test_postgres_fallback_null_year_start_skipped(self, service):
        """PostgreSQL model with year_start=None should be skipped."""

        async def _to_thread(fn, *args, **kwargs):
            return [], None

        mock_model = MagicMock()
        mock_model.year_start = None
        mock_model.year_end = 2023

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_model]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("asyncio.to_thread", side_effect=_to_thread),
            patch(
                "app.services.vehicle_service.async_session_maker",
                return_value=mock_session_ctx,
            ),
        ):
            years = await service.get_years_for_vehicle("X", "Y")

        assert years == []


# ===========================================================================
# get_vehicle_by_id
# ===========================================================================


class TestGetVehicleById:
    @pytest.mark.asyncio
    async def test_found(self, service):
        node = {
            "uid": "abc",
            "make": "Tesla",
            "model": "Model 3",
            "year_start": 2017,
            "year_end": None,
            "platform": None,
            "engine_codes": [],
            "body_types": ["sedan"],
        }

        async def _to_thread(fn, *args, **kwargs):
            return [[node]], None

        with patch("asyncio.to_thread", side_effect=_to_thread):
            result = await service.get_vehicle_by_id("abc")

        assert result is not None
        assert result["id"] == "abc"
        assert result["make"] == "Tesla"

    @pytest.mark.asyncio
    async def test_not_found(self, service):
        async def _to_thread(fn, *args, **kwargs):
            return [], None

        with patch("asyncio.to_thread", side_effect=_to_thread):
            result = await service.get_vehicle_by_id("nonexistent")

        assert result is None


# ===========================================================================
# find_vehicle
# ===========================================================================


class TestFindVehicle:
    @pytest.mark.asyncio
    async def test_found_with_year(self, service):
        node = {
            "uid": "v1",
            "make": "Honda",
            "model": "Civic",
            "year_start": 2019,
            "year_end": 2024,
            "platform": "FC",
            "engine_codes": ["L15B7"],
            "body_types": ["sedan", "hatchback"],
        }

        async def _to_thread(fn, *args, **kwargs):
            return [[node]], None

        with patch("asyncio.to_thread", side_effect=_to_thread):
            result = await service.find_vehicle("Honda", "Civic", year=2021)

        assert result is not None
        assert result["model"] == "Civic"

    @pytest.mark.asyncio
    async def test_found_without_year(self, service):
        node = {
            "uid": "v2",
            "make": "Mazda",
            "model": "3",
            "year_start": 2019,
            "year_end": None,
            "platform": None,
            "engine_codes": [],
            "body_types": [],
        }

        async def _to_thread(fn, *args, **kwargs):
            return [[node]], None

        with patch("asyncio.to_thread", side_effect=_to_thread):
            result = await service.find_vehicle("Mazda", "3")

        assert result is not None
        assert result["id"] == "v2"

    @pytest.mark.asyncio
    async def test_not_found(self, service):
        async def _to_thread(fn, *args, **kwargs):
            return [], None

        with patch("asyncio.to_thread", side_effect=_to_thread):
            result = await service.find_vehicle("Nonexistent", "Car", year=2020)

        assert result is None


# ===========================================================================
# get_vehicle_common_issues
# ===========================================================================


class TestGetVehicleCommonIssues:
    """Common-issues aggregation over the bare-label complaint graph.

    Regression coverage for the production 500 caused by querying the
    never-populated (:VehicleNode)-[:HAS_COMMON_ISSUE]->(:DTCNode) path.
    Neo4j aggregation rows are [code, description_en, description_hu, severity,
    occurrence_count]; enrichment is a batched PostgreSQL lookup by code.
    """

    @staticmethod
    def _dtc(code, description_en=None, description_hu=None, severity=None):
        """Build a DTCCode-like mock as returned by the PostgreSQL enrichment query."""
        m = MagicMock()
        m.code = code
        m.description_en = description_en
        m.description_hu = description_hu
        m.severity = severity
        return m

    @classmethod
    def _mock_pg_session(cls, dtc_rows):
        """Mock async_session_maker() -> session whose execute() yields dtc_rows."""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = dtc_rows

        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        return mock_ctx

    @pytest.mark.asyncio
    async def test_returns_ranked_issues_with_year(self, service):
        async def _to_thread(fn, *args, **kwargs):
            return [
                ["P0301", "Cylinder 1 Misfire", None, None, 42],
                ["P0420", "Catalyst Below Threshold", None, None, 7],
            ], None

        pg = self._mock_pg_session(
            [
                self._dtc(
                    "P0301", "Cylinder 1 Misfire Detected", "1. henger gyujtaskihagyas", "high"
                ),
                self._dtc(
                    "P0420", "Catalyst System Efficiency", "Katalizator hatekonysag", "medium"
                ),
            ]
        )

        with (
            patch("asyncio.to_thread", side_effect=_to_thread),
            patch("app.services.vehicle_service.async_session_maker", return_value=pg),
        ):
            issues = await service.get_vehicle_common_issues("Volkswagen", "Golf", year=2018)

        # Ranked by occurrence_count DESC (as returned by the graph)
        assert [i["code"] for i in issues] == ["P0301", "P0420"]
        assert issues[0]["occurrence_count"] == 42
        assert issues[0]["frequency"] == "very_common"
        # Enriched from the curated PostgreSQL table
        assert issues[0]["description_hu"] == "1. henger gyujtaskihagyas"
        assert issues[0]["severity"] == "high"
        assert issues[1]["frequency"] == "common"

    @pytest.mark.asyncio
    async def test_returns_issues_without_year_falls_back_to_graph(self, service):
        async def _to_thread(fn, *args, **kwargs):
            return [["P0171", "System Too Lean", "Rendszer tul sovany", "medium", 10]], None

        pg = self._mock_pg_session([])  # code not in curated table -> graph fallback

        with (
            patch("asyncio.to_thread", side_effect=_to_thread),
            patch("app.services.vehicle_service.async_session_maker", return_value=pg),
        ):
            issues = await service.get_vehicle_common_issues("Toyota", "Corolla")

        assert len(issues) == 1
        assert issues[0]["code"] == "P0171"
        assert issues[0]["description_hu"] == "Rendszer tul sovany"
        assert issues[0]["severity"] == "medium"
        assert issues[0]["frequency"] == "common"
        assert issues[0]["occurrence_count"] == 10

    @pytest.mark.asyncio
    async def test_enrichment_prefers_postgres_over_graph(self, service):
        async def _to_thread(fn, *args, **kwargs):
            # Graph node carries a stale English-only description, no hu / severity
            return [["P0128", "Coolant Thermostat", None, None, 3]], None

        pg = self._mock_pg_session(
            [self._dtc("P0128", "Coolant Thermostat (curated)", "Hutofolyadek termosztat", "low")]
        )

        with (
            patch("asyncio.to_thread", side_effect=_to_thread),
            patch("app.services.vehicle_service.async_session_maker", return_value=pg),
        ):
            issues = await service.get_vehicle_common_issues("Volkswagen", "Golf")

        assert issues[0]["description_en"] == "Coolant Thermostat (curated)"
        assert issues[0]["description_hu"] == "Hutofolyadek termosztat"
        assert issues[0]["severity"] == "low"
        assert issues[0]["frequency"] == "uncommon"

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_matches(self, service):
        """Zero graph matches -> empty list, no PostgreSQL call, no error."""

        async def _to_thread(fn, *args, **kwargs):
            return [], None

        with patch("asyncio.to_thread", side_effect=_to_thread):
            issues = await service.get_vehicle_common_issues("Mazda", "MX-5")

        assert issues == []

    @pytest.mark.asyncio
    async def test_neo4j_error_returns_empty_not_raise(self, service):
        """Driver/query error degrades to [] so the endpoint never returns 500."""
        with patch("asyncio.to_thread", side_effect=Exception("Neo4j down")):
            issues = await service.get_vehicle_common_issues("Volkswagen", "Golf", year=2018)

        assert issues == []

    @pytest.mark.asyncio
    async def test_postgres_enrichment_error_falls_back_to_graph(self, service):
        """PostgreSQL enrichment failure still returns graph data (never 500)."""

        async def _to_thread(fn, *args, **kwargs):
            return [["P0300", "Random Misfire", "Veletlenszeru kihagyas", "high", 30]], None

        broken_ctx = AsyncMock()
        broken_ctx.__aenter__ = AsyncMock(side_effect=Exception("PG down"))
        broken_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("asyncio.to_thread", side_effect=_to_thread),
            patch("app.services.vehicle_service.async_session_maker", return_value=broken_ctx),
        ):
            issues = await service.get_vehicle_common_issues("Volkswagen", "Golf")

        assert len(issues) == 1
        assert issues[0]["code"] == "P0300"
        assert issues[0]["description_hu"] == "Veletlenszeru kihagyas"
        assert issues[0]["frequency"] == "very_common"

    @pytest.mark.asyncio
    async def test_skips_rows_with_empty_code(self, service):
        """Null/empty DTC codes are dropped (VehicleCommonIssue.code is required)."""

        async def _to_thread(fn, *args, **kwargs):
            return [
                [None, "x", None, None, 5],
                ["", "y", None, None, 4],
                ["P0101", "MAF", None, None, 3],
            ], None

        pg = self._mock_pg_session([])

        with (
            patch("asyncio.to_thread", side_effect=_to_thread),
            patch("app.services.vehicle_service.async_session_maker", return_value=pg),
        ):
            issues = await service.get_vehicle_common_issues("Volkswagen", "Golf")

        assert [i["code"] for i in issues] == ["P0101"]

    @pytest.mark.asyncio
    async def test_uses_bare_label_aggregation_query(self, service):
        """Revert guard: the Cypher must use the bare-label complaint aggregation,
        never the dead VehicleNode / HAS_COMMON_ISSUE / DTCNode path.

        This test fails if the label fix is reverted.
        """
        captured = {}

        async def _to_thread(fn, *args, **kwargs):
            captured["query"] = args[0]
            captured["params"] = args[1] if len(args) > 1 else None
            return [], None

        with patch("asyncio.to_thread", side_effect=_to_thread):
            await service.get_vehicle_common_issues("Volkswagen", "Golf", year=2018)

        query = captured["query"]
        # New bare-label schema is present
        assert "(v:Vehicle)" in query
        assert "HAS_COMPLAINT" in query
        assert "(c:Complaint)" in query
        assert "MENTIONS_DTC" in query
        assert "MENTIONED_IN" in query
        assert "(d:DTC)" in query
        assert "count(DISTINCT c)" in query
        assert "ORDER BY occurrence_count DESC" in query
        # Dead path is gone
        assert "VehicleNode" not in query
        assert "HAS_COMMON_ISSUE" not in query
        assert "DTCNode" not in query
        # Year filters on the complaint
        assert "c.year" in query
        assert captured["params"]["year"] == 2018

    @pytest.mark.asyncio
    async def test_no_year_omits_year_filter(self, service):
        captured = {}

        async def _to_thread(fn, *args, **kwargs):
            captured["query"] = args[0]
            captured["params"] = args[1] if len(args) > 1 else None
            return [], None

        with patch("asyncio.to_thread", side_effect=_to_thread):
            await service.get_vehicle_common_issues("Volkswagen", "Golf")

        assert "c.year" not in captured["query"]
        assert "year" not in captured["params"]

    @pytest.mark.asyncio
    async def test_frequency_buckets(self, service):
        assert service._frequency_bucket(25) == "very_common"
        assert service._frequency_bucket(20) == "very_common"
        assert service._frequency_bucket(19) == "common"
        assert service._frequency_bucket(5) == "common"
        assert service._frequency_bucket(4) == "uncommon"
        assert service._frequency_bucket(2) == "uncommon"
        assert service._frequency_bucket(1) == "rare"
        assert service._frequency_bucket(0) == "rare"
        assert service._frequency_bucket(None) == "rare"


# ===========================================================================
# get_vehicle_service (singleton)
# ===========================================================================


class TestGetVehicleServiceSingleton:
    def test_returns_vehicle_service_instance(self):
        with patch("app.services.vehicle_service._vehicle_service", None):
            svc = get_vehicle_service()
            assert isinstance(svc, VehicleService)

    def test_returns_same_instance(self):
        with patch("app.services.vehicle_service._vehicle_service", None):
            svc1 = get_vehicle_service()
            with patch("app.services.vehicle_service._vehicle_service", svc1):
                svc2 = get_vehicle_service()
            assert svc1 is svc2
