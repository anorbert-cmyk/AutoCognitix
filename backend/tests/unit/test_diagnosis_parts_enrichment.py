"""
Unit tests for parts-price enrichment and response building in DiagnosisService.

These tests protect the streaming/non-streaming parity fix: the streaming
diagnosis path (``_run_pipeline``) must enrich with parts prices exactly like
``analyze_vehicle`` Step 5.5 does. They exercise the two service primitives the
stream relies on in isolation - no DB or HTTP:

- ``_enrich_with_parts_prices`` (P0171 is in the static DTC_PARTS_MAPPING) and
  its failure isolation contract.
- ``_build_response`` with and without ``parts_data``.

``_enrich_with_parts_prices`` and ``_build_response`` never touch ``self.db``,
so an ``AsyncMock`` session is sufficient. The Redis-backed parts cache is
patched to an in-process no-op so the tests stay hermetic and fast.
"""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.api.v1.schemas.diagnosis import DiagnosisRequest
from app.services.diagnosis_service import DiagnosisService
from app.services.parts_price_service import PartsPriceCache


def _make_service() -> DiagnosisService:
    """Build a DiagnosisService with a mock db (enrichment/build never use it)."""
    return DiagnosisService(db=AsyncMock())


def _minimal_rag_result() -> dict:
    """Smallest rag_result dict accepted by _build_response."""
    return {
        "probable_causes": [],
        "recommended_repairs": [],
        "confidence_score": 0.6,
        "sources": [],
    }


def _sample_parts_data() -> dict:
    """A valid parts_data payload shaped like _enrich_with_parts_prices output."""
    return {
        "parts": [
            {
                "id": "spark_plug",
                "name": "Gyujtogyertya",
                "name_en": "Spark Plug",
                "category": "ignition",
                "price_range_min": 1500,
                "price_range_max": 8000,
                "labor_hours": 0.5,
                "currency": "HUF",
            }
        ],
        "cost_estimate": {
            "parts_cost_min": 1500,
            "parts_cost_max": 8000,
            "labor_cost_min": 6000,
            "labor_cost_max": 12500,
            "total_cost_min": 7500,
            "total_cost_max": 20500,
            "estimated_hours": 0.5,
            "difficulty": "easy",
            "disclaimer": "Tajekoztato jellegu becsles.",
        },
    }


class TestEnrichWithPartsPrices:
    """Unit tests for DiagnosisService._enrich_with_parts_prices."""

    @pytest.mark.asyncio
    async def test_p0171_returns_parts_and_cost_estimate(self):
        """P0171 maps to static parts, so enrichment yields parts + a cost estimate."""
        service = _make_service()
        with (
            patch.object(PartsPriceCache, "get", new_callable=AsyncMock, return_value=None),
            patch.object(PartsPriceCache, "set", new_callable=AsyncMock, return_value=None),
        ):
            result = await service._enrich_with_parts_prices(
                dtc_codes=["P0171"],
                vehicle_make="Volkswagen",
                vehicle_model="Golf",
                vehicle_year=2018,
            )

        assert result["parts"], "Expected non-empty parts for P0171"
        assert result["cost_estimate"], "Expected a truthy cost_estimate for P0171"
        assert result["cost_estimate"]["total_cost_max"] > 0

    @pytest.mark.asyncio
    async def test_failure_is_isolated_and_returns_empty(self):
        """A broken parts source must degrade to empty results, never raise."""
        service = _make_service()
        with patch(
            "app.services.diagnosis_service.get_parts_price_service",
            side_effect=RuntimeError("parts backend down"),
        ):
            result = await service._enrich_with_parts_prices(
                dtc_codes=["P0171"],
                vehicle_make="Volkswagen",
                vehicle_model="Golf",
                vehicle_year=2018,
            )

        assert result == {"parts": [], "cost_estimate": None}


class TestBuildResponsePartsData:
    """Unit tests for DiagnosisService._build_response parts handling."""

    def _request(self) -> DiagnosisRequest:
        return DiagnosisRequest(
            vehicle_make="Volkswagen",
            vehicle_model="Golf",
            vehicle_year=2018,
            dtc_codes=["P0171"],
            symptoms="A motor egyenetlenul jar alapjaraton, reszletes leiras.",
        )

    def test_with_parts_data_populates_prices_and_total(self):
        """parts_data present -> parts_with_prices populated and total_cost_estimate set."""
        service = _make_service()
        response = service._build_response(
            diagnosis_id=uuid4(),
            request=self._request(),
            dtc_details=[],
            rag_result=_minimal_rag_result(),
            recalls=[],
            complaints=[],
            parts_data=_sample_parts_data(),
        )

        assert len(response.parts_with_prices) == 1
        assert response.parts_with_prices[0].id == "spark_plug"
        assert response.total_cost_estimate is not None
        assert response.total_cost_estimate.total_max == 20500

    def test_without_parts_data_leaves_prices_empty(self):
        """parts_data=None -> empty parts_with_prices and total_cost_estimate is None."""
        service = _make_service()
        response = service._build_response(
            diagnosis_id=uuid4(),
            request=self._request(),
            dtc_details=[],
            rag_result=_minimal_rag_result(),
            recalls=[],
            complaints=[],
            parts_data=None,
        )

        assert response.parts_with_prices == []
        assert response.total_cost_estimate is None
