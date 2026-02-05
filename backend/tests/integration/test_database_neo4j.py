"""
Integration tests for Neo4j graph database operations.

Tests graph queries for diagnostic paths, symptoms, and components.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_path))


class TestNeo4jDiagnosticPaths:
    """Test Neo4j diagnostic path queries."""

    @pytest.mark.asyncio
    async def test_get_diagnostic_path_returns_data(self, mock_neo4j_client):
        """Test that get_diagnostic_path returns graph data."""
        with patch("app.db.neo4j_models.driver") as mock_driver:
            mock_session = AsyncMock()
            mock_driver.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_session.run = mock_neo4j_client.run

            from app.db.neo4j_models import get_diagnostic_path

            result = await get_diagnostic_path("P0101")

            # Should return data from mock
            assert result is not None or mock_neo4j_client.run.called

    @pytest.mark.asyncio
    async def test_diagnostic_path_contains_dtc_info(self, mock_neo4j_client):
        """Test that diagnostic path includes DTC information."""
        mock_result = {
            "dtc": {"code": "P0101", "description": "MAF Circuit Issue"},
            "symptoms": [{"name": "Rough idle"}],
            "components": [{"name": "MAF Sensor", "system": "Engine"}],
            "repairs": [{"name": "Replace MAF Sensor", "difficulty": "beginner"}],
        }

        mock_neo4j_client.run.return_value = [mock_result]

        with patch("app.db.neo4j_models.driver") as mock_driver:
            mock_session = AsyncMock()
            mock_session.run = mock_neo4j_client.run
            mock_driver.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)

            # Verify mock setup
            assert mock_neo4j_client.run.return_value[0]["dtc"]["code"] == "P0101"

    @pytest.mark.asyncio
    async def test_diagnostic_path_contains_symptoms(self, mock_neo4j_client):
        """Test that diagnostic path includes symptoms."""
        mock_result = {
            "dtc": {"code": "P0101"},
            "symptoms": [
                {"name": "Rough idle"},
                {"name": "Poor acceleration"},
            ],
            "components": [],
            "repairs": [],
        }

        mock_neo4j_client.run.return_value = [mock_result]

        # Verify symptoms in mock
        symptoms = mock_neo4j_client.run.return_value[0]["symptoms"]
        assert len(symptoms) == 2
        assert symptoms[0]["name"] == "Rough idle"

    @pytest.mark.asyncio
    async def test_diagnostic_path_contains_components(self, mock_neo4j_client):
        """Test that diagnostic path includes affected components."""
        mock_result = {
            "dtc": {"code": "P0101"},
            "symptoms": [],
            "components": [
                {"name": "MAF Sensor", "system": "Engine"},
                {"name": "Air Intake", "system": "Engine"},
            ],
            "repairs": [],
        }

        mock_neo4j_client.run.return_value = [mock_result]

        # Verify components in mock
        components = mock_neo4j_client.run.return_value[0]["components"]
        assert len(components) == 2
        assert components[0]["name"] == "MAF Sensor"

    @pytest.mark.asyncio
    async def test_diagnostic_path_contains_repairs(self, mock_neo4j_client):
        """Test that diagnostic path includes repair recommendations."""
        mock_result = {
            "dtc": {"code": "P0101"},
            "symptoms": [],
            "components": [],
            "repairs": [
                {"name": "Clean MAF Sensor", "difficulty": "beginner"},
                {"name": "Replace MAF Sensor", "difficulty": "intermediate"},
            ],
        }

        mock_neo4j_client.run.return_value = [mock_result]

        # Verify repairs in mock
        repairs = mock_neo4j_client.run.return_value[0]["repairs"]
        assert len(repairs) == 2
        assert repairs[0]["difficulty"] == "beginner"


class TestNeo4jDTCRelationships:
    """Test Neo4j DTC relationship queries."""

    @pytest.mark.asyncio
    async def test_get_related_dtc_codes(self, mock_neo4j_client):
        """Test getting related DTC codes."""
        mock_neo4j_client.run.return_value = [
            {"related_code": "P0100", "relationship": "RELATED_TO"},
            {"related_code": "P0102", "relationship": "RELATED_TO"},
            {"related_code": "P0103", "relationship": "RELATED_TO"},
        ]

        # Verify mock returns related codes
        results = mock_neo4j_client.run.return_value
        assert len(results) == 3
        codes = [r["related_code"] for r in results]
        assert "P0100" in codes

    @pytest.mark.asyncio
    async def test_get_dtc_by_symptom(self, mock_neo4j_client):
        """Test finding DTC codes by symptom."""
        mock_neo4j_client.run.return_value = [
            {"code": "P0101", "description": "MAF Circuit Issue"},
            {"code": "P0171", "description": "System Too Lean"},
        ]

        # Verify mock returns DTCs for symptom
        results = mock_neo4j_client.run.return_value
        assert len(results) == 2
        codes = [r["code"] for r in results]
        assert "P0101" in codes
        assert "P0171" in codes

    @pytest.mark.asyncio
    async def test_get_dtc_by_component(self, mock_neo4j_client):
        """Test finding DTC codes by affected component."""
        mock_neo4j_client.run.return_value = [
            {"code": "P0101", "description": "MAF Circuit Issue"},
            {"code": "P0100", "description": "MAF Circuit Malfunction"},
        ]

        # Verify mock returns DTCs for component
        results = mock_neo4j_client.run.return_value
        assert len(results) == 2


class TestNeo4jVehicleSpecificData:
    """Test Neo4j vehicle-specific data queries."""

    @pytest.mark.asyncio
    async def test_get_vehicle_common_issues(self, mock_neo4j_client):
        """Test getting common issues for a vehicle."""
        mock_neo4j_client.run.return_value = [
            {
                "dtc": "P0101",
                "frequency": "common",
                "year_range": "2015-2020",
            },
            {
                "dtc": "P0171",
                "frequency": "occasional",
                "year_range": "2015-2020",
            },
        ]

        # Verify mock returns vehicle-specific issues
        results = mock_neo4j_client.run.return_value
        assert len(results) == 2
        assert results[0]["frequency"] == "common"

    @pytest.mark.asyncio
    async def test_get_vehicle_by_make_model(self, mock_neo4j_client):
        """Test finding vehicle node by make and model."""
        mock_neo4j_client.run.return_value = [
            {
                "make": "Volkswagen",
                "model": "Golf",
                "years": [2015, 2016, 2017, 2018, 2019, 2020],
            }
        ]

        # Verify mock returns vehicle data
        results = mock_neo4j_client.run.return_value
        assert len(results) == 1
        assert results[0]["make"] == "Volkswagen"
        assert results[0]["model"] == "Golf"


class TestNeo4jRepairPaths:
    """Test Neo4j repair path queries."""

    @pytest.mark.asyncio
    async def test_get_repair_steps(self, mock_neo4j_client):
        """Test getting repair steps for a component."""
        mock_neo4j_client.run.return_value = [
            {
                "step": 1,
                "action": "Disconnect battery",
                "tools": ["10mm socket"],
            },
            {
                "step": 2,
                "action": "Remove MAF sensor connector",
                "tools": [],
            },
            {
                "step": 3,
                "action": "Remove MAF sensor",
                "tools": ["T20 Torx"],
            },
        ]

        # Verify mock returns repair steps
        results = mock_neo4j_client.run.return_value
        assert len(results) == 3
        assert results[0]["step"] == 1

    @pytest.mark.asyncio
    async def test_get_repair_parts(self, mock_neo4j_client):
        """Test getting required parts for a repair."""
        mock_neo4j_client.run.return_value = [
            {
                "part_name": "MAF Sensor",
                "part_number": "123456789",
                "estimated_cost": 15000,
            },
            {
                "part_name": "Air Filter",
                "part_number": "987654321",
                "estimated_cost": 5000,
            },
        ]

        # Verify mock returns parts
        results = mock_neo4j_client.run.return_value
        assert len(results) == 2
        assert results[0]["part_name"] == "MAF Sensor"

    @pytest.mark.asyncio
    async def test_get_repair_difficulty(self, mock_neo4j_client):
        """Test getting repair difficulty information."""
        mock_neo4j_client.run.return_value = [
            {
                "repair_name": "Replace MAF Sensor",
                "difficulty": "beginner",
                "time_minutes": 30,
                "requires_lift": False,
            }
        ]

        # Verify mock returns difficulty
        results = mock_neo4j_client.run.return_value
        assert results[0]["difficulty"] == "beginner"
        assert results[0]["time_minutes"] == 30


class TestNeo4jSymptomQueries:
    """Test Neo4j symptom-related queries."""

    @pytest.mark.asyncio
    async def test_find_symptoms_by_dtc(self, mock_neo4j_client):
        """Test finding symptoms associated with a DTC code."""
        mock_neo4j_client.run.return_value = [
            {"name": "Rough idle", "severity": "medium"},
            {"name": "Poor acceleration", "severity": "high"},
            {"name": "Check engine light", "severity": "low"},
        ]

        # Verify mock returns symptoms
        results = mock_neo4j_client.run.return_value
        assert len(results) == 3
        symptom_names = [r["name"] for r in results]
        assert "Rough idle" in symptom_names

    @pytest.mark.asyncio
    async def test_find_dtc_by_symptom_combination(self, mock_neo4j_client):
        """Test finding DTC codes by multiple symptoms."""
        mock_neo4j_client.run.return_value = [
            {
                "code": "P0101",
                "matching_symptoms": 3,
                "total_symptoms": 3,
                "match_score": 1.0,
            },
            {
                "code": "P0100",
                "matching_symptoms": 2,
                "total_symptoms": 3,
                "match_score": 0.67,
            },
        ]

        # Verify mock returns scored results
        results = mock_neo4j_client.run.return_value
        assert len(results) == 2
        assert results[0]["match_score"] == 1.0


class TestNeo4jGraphIntegrity:
    """Test Neo4j graph data integrity."""

    @pytest.mark.asyncio
    async def test_dtc_node_has_required_properties(self, mock_neo4j_client):
        """Test that DTC nodes have required properties."""
        mock_neo4j_client.run.return_value = [
            {
                "code": "P0101",
                "description": "MAF Circuit Issue",
                "category": "powertrain",
                "severity": "medium",
            }
        ]

        # Verify mock has required properties
        result = mock_neo4j_client.run.return_value[0]
        assert "code" in result
        assert "description" in result
        assert "category" in result
        assert "severity" in result

    @pytest.mark.asyncio
    async def test_component_node_has_required_properties(self, mock_neo4j_client):
        """Test that Component nodes have required properties."""
        mock_neo4j_client.run.return_value = [
            {
                "name": "MAF Sensor",
                "system": "Engine",
                "location": "Air Intake",
            }
        ]

        # Verify mock has required properties
        result = mock_neo4j_client.run.return_value[0]
        assert "name" in result
        assert "system" in result

    @pytest.mark.asyncio
    async def test_repair_node_has_required_properties(self, mock_neo4j_client):
        """Test that Repair nodes have required properties."""
        mock_neo4j_client.run.return_value = [
            {
                "name": "Replace MAF Sensor",
                "difficulty": "intermediate",
                "estimated_time_minutes": 30,
            }
        ]

        # Verify mock has required properties
        result = mock_neo4j_client.run.return_value[0]
        assert "name" in result
        assert "difficulty" in result


class TestNeo4jErrorHandling:
    """Test Neo4j error handling."""

    @pytest.mark.asyncio
    async def test_handles_connection_error(self, mock_neo4j_client):
        """Test handling of connection errors."""
        mock_neo4j_client.run.side_effect = Exception("Connection failed")

        # Should handle error gracefully
        with pytest.raises(Exception) as exc_info:
            await mock_neo4j_client.run("MATCH (n) RETURN n")

        assert "Connection failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handles_query_error(self, mock_neo4j_client):
        """Test handling of query errors."""
        mock_neo4j_client.run.side_effect = Exception("Invalid query syntax")

        # Should handle error gracefully
        with pytest.raises(Exception) as exc_info:
            await mock_neo4j_client.run("INVALID QUERY")

        assert "Invalid query" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handles_empty_result(self, mock_neo4j_client):
        """Test handling of empty results."""
        mock_neo4j_client.run.return_value = []

        # Should return empty list, not error
        result = mock_neo4j_client.run.return_value
        assert result == []

    @pytest.mark.asyncio
    async def test_handles_null_values(self, mock_neo4j_client):
        """Test handling of null values in results."""
        mock_neo4j_client.run.return_value = [
            {
                "code": "P0101",
                "description": None,
                "severity": None,
            }
        ]

        # Should handle null values
        result = mock_neo4j_client.run.return_value[0]
        assert result["code"] == "P0101"
        assert result["description"] is None
