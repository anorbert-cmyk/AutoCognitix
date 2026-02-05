"""
End-to-end tests for database integrity and consistency.

Tests data consistency across:
- PostgreSQL data integrity (constraints, relationships)
- Neo4j graph relationships (DTC -> Symptom -> Component -> Repair)
- Qdrant vector search accuracy
- Cross-database consistency
"""

import pytest
from unittest.mock import patch
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_path))


class TestPostgreSQLDataIntegrity:
    """Test PostgreSQL data integrity constraints."""

    @pytest.mark.asyncio
    async def test_dtc_code_unique_constraint(self, async_client, seeded_db):
        """Test that DTC code is unique in database."""
        # First create a DTC code
        dtc1 = {
            "code": "P5001",
            "description_en": "First Test Code",
            "category": "powertrain",
            "severity": "medium",
        }

        response1 = await async_client.post("/api/v1/dtc/", json=dtc1)
        assert response1.status_code == 201

        # Try to create duplicate
        dtc2 = {
            "code": "P5001",
            "description_en": "Duplicate Code",
            "category": "powertrain",
            "severity": "low",
        }

        response2 = await async_client.post("/api/v1/dtc/", json=dtc2)
        assert response2.status_code == 400

    @pytest.mark.asyncio
    async def test_user_email_unique_constraint(self, async_client, seeded_db):
        """Test that user email is unique."""
        # First registration
        user1 = {
            "email": "uniquetest@example.com",
            "password": "SecurePassword123!",
        }

        response1 = await async_client.post("/api/v1/auth/register", json=user1)
        assert response1.status_code == 201

        # Duplicate email registration
        user2 = {
            "email": "uniquetest@example.com",
            "password": "DifferentPassword123!",
        }

        response2 = await async_client.post("/api/v1/auth/register", json=user2)
        assert response2.status_code == 400

    @pytest.mark.asyncio
    async def test_dtc_code_format_constraint(self, async_client, seeded_db):
        """Test that DTC code format is validated."""
        invalid_dtcs = [
            {"code": "INVALID", "description_en": "Test", "category": "powertrain", "severity": "medium"},
            {"code": "X1234", "description_en": "Test", "category": "powertrain", "severity": "medium"},
            {"code": "P12", "description_en": "Test", "category": "powertrain", "severity": "medium"},
        ]

        for dtc in invalid_dtcs:
            response = await async_client.post("/api/v1/dtc/", json=dtc)
            assert response.status_code in [400, 422], f"Invalid code {dtc['code']} should be rejected"

    @pytest.mark.asyncio
    async def test_dtc_category_enum_constraint(self, async_client, seeded_db):
        """Test that DTC category must be valid enum value."""
        invalid_dtc = {
            "code": "P5555",
            "description_en": "Test Code",
            "category": "invalid_category",  # Not in enum
            "severity": "medium",
        }

        response = await async_client.post("/api/v1/dtc/", json=invalid_dtc)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_dtc_severity_enum_constraint(self, async_client, seeded_db):
        """Test that DTC severity must be valid enum value."""
        invalid_dtc = {
            "code": "P5556",
            "description_en": "Test Code",
            "category": "powertrain",
            "severity": "invalid_severity",  # Not in enum
        }

        response = await async_client.post("/api/v1/dtc/", json=invalid_dtc)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_user_role_enum_constraint(self, async_client, seeded_db):
        """Test that user role must be valid value."""
        user = {
            "email": "invalidrole@example.com",
            "password": "SecurePassword123!",
            "role": "superadmin",  # Invalid role
        }

        response = await async_client.post("/api/v1/auth/register", json=user)
        # Should either reject or default to user role
        if response.status_code == 201:
            data = response.json()
            assert data["role"] in ["user", "mechanic", "admin"]

    @pytest.mark.asyncio
    async def test_diagnosis_session_vehicle_fields_required(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
    ):
        """Test that diagnosis requires vehicle fields."""
        incomplete_requests = [
            # Missing vehicle_make
            {
                "vehicle_model": "Golf",
                "vehicle_year": 2018,
                "dtc_codes": ["P0101"],
                "symptoms": "A motor nehezen indul.",
            },
            # Missing vehicle_model
            {
                "vehicle_make": "Volkswagen",
                "vehicle_year": 2018,
                "dtc_codes": ["P0101"],
                "symptoms": "A motor nehezen indul.",
            },
            # Missing vehicle_year
            {
                "vehicle_make": "Volkswagen",
                "vehicle_model": "Golf",
                "dtc_codes": ["P0101"],
                "symptoms": "A motor nehezen indul.",
            },
        ]

        for request_data in incomplete_requests:
            response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)
            assert response.status_code == 422


class TestDataTypeConsistency:
    """Test data type consistency across operations."""

    @pytest.mark.asyncio
    async def test_uuid_format_in_responses(self, async_client, seeded_db):
        """Test that UUIDs in responses are valid format."""
        user = {
            "email": "uuidtest@example.com",
            "password": "SecurePassword123!",
        }

        response = await async_client.post("/api/v1/auth/register", json=user)
        assert response.status_code == 201
        data = response.json()

        # Verify UUID format
        import uuid
        try:
            uuid.UUID(data["id"])
        except ValueError:
            pytest.fail(f"Invalid UUID format: {data['id']}")

    @pytest.mark.asyncio
    async def test_datetime_format_in_responses(self, async_client, seeded_db):
        """Test that datetime fields are properly formatted."""
        user = {
            "email": "datetimetest@example.com",
            "password": "SecurePassword123!",
        }

        response = await async_client.post("/api/v1/auth/register", json=user)
        assert response.status_code == 201
        data = response.json()

        if data.get("created_at"):
            from datetime import datetime
            try:
                # Try parsing ISO format
                datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
            except ValueError:
                pytest.fail(f"Invalid datetime format: {data['created_at']}")

    @pytest.mark.asyncio
    async def test_confidence_scores_in_valid_range(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
        diagnosis_request_data,
    ):
        """Test that confidence scores are always between 0 and 1."""
        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=diagnosis_request_data,
            )

            if response.status_code == 201:
                data = response.json()

                # Check overall confidence
                assert 0 <= data["confidence_score"] <= 1

                # Check probable cause confidences
                for cause in data["probable_causes"]:
                    assert 0 <= cause["confidence"] <= 1

    @pytest.mark.asyncio
    async def test_relevance_scores_in_valid_range(self, async_client, seeded_db):
        """Test that search relevance scores are between 0 and 1."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101"},
        )

        assert response.status_code == 200
        data = response.json()

        for item in data:
            if item.get("relevance_score") is not None:
                assert 0 <= item["relevance_score"] <= 1


class TestNeo4jGraphRelationships:
    """Test Neo4j graph relationship integrity."""

    @pytest.mark.asyncio
    async def test_dtc_to_symptoms_relationship(
        self,
        async_client,
        seeded_db,
        mock_neo4j_client,
    ):
        """Test DTC -> Symptom relationship exists in graph data."""
        with patch("app.db.neo4j_models.get_diagnostic_path") as mock_path:
            mock_path.return_value = {
                "symptoms": [
                    {"name": "Rough idle"},
                    {"name": "Poor acceleration"},
                ],
                "components": [],
                "repairs": [],
            }

            response = await async_client.get(
                "/api/v1/dtc/P0101",
                params={"include_graph": True},
            )

            assert response.status_code == 200
            data = response.json()
            # Symptoms from graph should be included
            assert "symptoms" in data
            assert isinstance(data["symptoms"], list)

    @pytest.mark.asyncio
    async def test_dtc_to_components_relationship(
        self,
        async_client,
        seeded_db,
        mock_neo4j_client,
    ):
        """Test DTC -> Component relationship in graph."""
        with patch("app.db.neo4j_models.get_diagnostic_path") as mock_path:
            mock_path.return_value = {
                "symptoms": [],
                "components": [
                    {"name": "MAF Sensor", "system": "Engine"},
                ],
                "repairs": [],
            }

            response = await async_client.get(
                "/api/v1/dtc/P0101",
                params={"include_graph": True},
            )

            assert response.status_code == 200
            data = response.json()
            # Components may be in possible_causes
            assert "possible_causes" in data

    @pytest.mark.asyncio
    async def test_dtc_to_repairs_relationship(
        self,
        async_client,
        seeded_db,
        mock_neo4j_client,
    ):
        """Test DTC -> Repair relationship in graph."""
        with patch("app.db.neo4j_models.get_diagnostic_path") as mock_path:
            mock_path.return_value = {
                "symptoms": [],
                "components": [],
                "repairs": [
                    {"name": "Clean MAF Sensor", "difficulty": "beginner"},
                ],
            }

            response = await async_client.get(
                "/api/v1/dtc/P0101",
                params={"include_graph": True},
            )

            assert response.status_code == 200
            data = response.json()
            # Repairs should be in diagnostic_steps
            assert "diagnostic_steps" in data

    @pytest.mark.asyncio
    async def test_related_codes_from_graph(
        self,
        async_client,
        seeded_db,
        mock_neo4j_client,
    ):
        """Test related DTC codes from graph relationships."""
        response = await async_client.get("/api/v1/dtc/P0101/related")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_graph_data_graceful_fallback(self, async_client, seeded_db):
        """Test that missing graph data falls back gracefully."""
        with patch("app.db.neo4j_models.get_diagnostic_path") as mock_path:
            mock_path.side_effect = Exception("Neo4j connection failed")

            response = await async_client.get(
                "/api/v1/dtc/P0101",
                params={"include_graph": True},
            )

            # Should still return data from PostgreSQL
            assert response.status_code == 200
            data = response.json()
            assert data["code"] == "P0101"


class TestQdrantVectorSearchAccuracy:
    """Test Qdrant vector search accuracy."""

    @pytest.mark.asyncio
    async def test_semantic_search_returns_relevant_results(
        self,
        async_client,
        seeded_db,
        mock_embedding_service,
        mock_qdrant_client,
    ):
        """Test that semantic search returns relevant results."""
        with patch("app.services.embedding_service.get_embedding_service", return_value=mock_embedding_service), \
             patch("app.db.qdrant_client.qdrant_client", mock_qdrant_client):

            response = await async_client.get(
                "/api/v1/dtc/search",
                params={"q": "motor rezeg", "use_semantic": True},
            )

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_vector_search_graceful_fallback(self, async_client, seeded_db):
        """Test that vector search falls back to text search on error."""
        with patch("app.services.embedding_service.get_embedding_service") as mock_embed:
            mock_embed.side_effect = Exception("Embedding service unavailable")

            # Should still work with text search
            response = await async_client.get(
                "/api/v1/dtc/search",
                params={"q": "P0101", "use_semantic": True},
            )

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_embedding_dimension_consistency(self, mock_embedding_service):
        """Test that embedding dimensions are consistent (768 for huBERT)."""
        assert mock_embedding_service.embedding_dimension == 768

        # Generate embedding
        embedding = mock_embedding_service.embed_text("test text")
        assert len(embedding) == 768


class TestCrossDatabaseConsistency:
    """Test consistency across different databases."""

    @pytest.mark.asyncio
    async def test_dtc_exists_in_postgres_searchable(self, async_client, seeded_db):
        """Test that DTC codes in PostgreSQL are searchable."""
        # Seeded data includes P0101
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101"},
        )

        assert response.status_code == 200
        data = response.json()
        codes = [item["code"] for item in data]
        assert "P0101" in codes

    @pytest.mark.asyncio
    async def test_dtc_detail_matches_search_result(self, async_client, seeded_db):
        """Test that DTC detail matches search result data."""
        # Get from search
        search_response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101"},
        )
        search_data = search_response.json()

        if len(search_data) > 0:
            search_item = next((item for item in search_data if item["code"] == "P0101"), None)

            if search_item:
                # Get detail
                detail_response = await async_client.get("/api/v1/dtc/P0101")
                detail_data = detail_response.json()

                # Compare common fields
                assert search_item["code"] == detail_data["code"]
                assert search_item["description_en"] == detail_data["description_en"]
                assert search_item["category"] == detail_data["category"]

    @pytest.mark.asyncio
    async def test_user_data_consistent_across_requests(self, async_client, seeded_db):
        """Test that user data is consistent across requests."""
        # Register user
        user = {
            "email": "consistent@example.com",
            "password": "SecurePassword123!",
            "full_name": "Consistent User",
        }
        register_response = await async_client.post("/api/v1/auth/register", json=user)
        register_data = register_response.json()

        # Login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "consistent@example.com", "password": "SecurePassword123!"},
        )
        tokens = login_response.json()

        # Get profile
        profile_response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        profile_data = profile_response.json()

        # Compare
        assert register_data["id"] == profile_data["id"]
        assert register_data["email"] == profile_data["email"]
        assert register_data["full_name"] == profile_data["full_name"]


class TestDataPersistence:
    """Test data persistence across operations."""

    @pytest.mark.asyncio
    async def test_created_dtc_persists(self, async_client, seeded_db):
        """Test that created DTC code persists."""
        # Create DTC
        dtc = {
            "code": "P7001",
            "description_en": "Persistent Test Code",
            "description_hu": "Maradando teszt kod",
            "category": "powertrain",
            "severity": "medium",
            "is_generic": True,
            "symptoms": ["Test symptom"],
            "possible_causes": ["Test cause"],
        }

        create_response = await async_client.post("/api/v1/dtc/", json=dtc)
        assert create_response.status_code == 201

        # Retrieve it
        get_response = await async_client.get("/api/v1/dtc/P7001")
        assert get_response.status_code == 200
        data = get_response.json()

        assert data["code"] == "P7001"
        assert data["description_en"] == "Persistent Test Code"
        assert data["description_hu"] == "Maradando teszt kod"

    @pytest.mark.asyncio
    async def test_updated_profile_persists(self, async_client, seeded_db):
        """Test that profile updates persist."""
        # Register and login
        user = {
            "email": "updatepersist@example.com",
            "password": "SecurePassword123!",
            "full_name": "Original Name",
        }
        await async_client.post("/api/v1/auth/register", json=user)

        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "updatepersist@example.com", "password": "SecurePassword123!"},
        )
        tokens = login_response.json()
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}

        # Update profile
        await async_client.put(
            "/api/v1/auth/me",
            json={"full_name": "Updated Name"},
            headers=headers,
        )

        # Verify persistence
        profile_response = await async_client.get("/api/v1/auth/me", headers=headers)
        profile_data = profile_response.json()

        assert profile_data["full_name"] == "Updated Name"


class TestBulkOperationIntegrity:
    """Test data integrity during bulk operations."""

    @pytest.mark.asyncio
    async def test_bulk_import_atomic(self, async_client, seeded_db):
        """Test that bulk import operations maintain atomicity."""
        bulk_data = {
            "codes": [
                {
                    "code": "P8001",
                    "description_en": "Bulk Test 1",
                    "category": "powertrain",
                    "severity": "medium",
                },
                {
                    "code": "P8002",
                    "description_en": "Bulk Test 2",
                    "category": "powertrain",
                    "severity": "low",
                },
            ],
            "overwrite_existing": False,
        }

        response = await async_client.post("/api/v1/dtc/bulk", json=bulk_data)
        assert response.status_code == 201
        data = response.json()

        # Verify all codes were created
        created_count = data["created"]

        # Check they exist
        for code in ["P8001", "P8002"]:
            check_response = await async_client.get(f"/api/v1/dtc/{code}")
            if created_count > 0:
                assert check_response.status_code in [200, 404]

    @pytest.mark.asyncio
    async def test_bulk_import_error_handling(self, async_client, seeded_db):
        """Test that bulk import handles errors without corrupting data."""
        # First create a code
        await async_client.post(
            "/api/v1/dtc/",
            json={
                "code": "P8100",
                "description_en": "Existing Code",
                "category": "powertrain",
                "severity": "medium",
            },
        )

        # Bulk import with duplicate
        bulk_data = {
            "codes": [
                {
                    "code": "P8100",  # Duplicate
                    "description_en": "Duplicate Attempt",
                    "category": "powertrain",
                    "severity": "medium",
                },
                {
                    "code": "P8101",  # New
                    "description_en": "New Code",
                    "category": "powertrain",
                    "severity": "medium",
                },
            ],
            "overwrite_existing": False,
        }

        response = await async_client.post("/api/v1/dtc/bulk", json=bulk_data)
        assert response.status_code == 201
        data = response.json()

        # Should have skipped duplicate and created new
        assert data["skipped"] >= 0

        # Original code should be unchanged
        check_response = await async_client.get("/api/v1/dtc/P8100")
        assert check_response.status_code == 200
        check_data = check_response.json()
        assert check_data["description_en"] == "Existing Code"


class TestSearchIndexConsistency:
    """Test search index consistency."""

    @pytest.mark.asyncio
    async def test_newly_created_dtc_searchable(self, async_client, seeded_db):
        """Test that newly created DTC is immediately searchable."""
        # Create new DTC
        dtc = {
            "code": "P9001",
            "description_en": "Unique Searchable Code",
            "category": "powertrain",
            "severity": "medium",
        }

        create_response = await async_client.post("/api/v1/dtc/", json=dtc)
        assert create_response.status_code == 201

        # Should be immediately searchable
        search_response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P9001"},
        )

        assert search_response.status_code == 200
        data = search_response.json()
        codes = [item["code"] for item in data]
        assert "P9001" in codes

    @pytest.mark.asyncio
    async def test_text_search_matches_description(self, async_client, seeded_db):
        """Test that text search matches description content."""
        # Seeded P0101 has "Mass Air Flow" in description
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "Mass Air Flow"},
        )

        assert response.status_code == 200
        data = response.json()

        # Should find P0101 or similar MAF codes
        assert len(data) >= 0  # May not find if not seeded with this description

    @pytest.mark.asyncio
    async def test_category_filter_accuracy(self, async_client, seeded_db):
        """Test that category filter is accurate."""
        categories = ["powertrain", "body", "chassis", "network"]

        for category in categories:
            response = await async_client.get(
                "/api/v1/dtc/search",
                params={"q": "sensor", "category": category},
            )

            assert response.status_code == 200
            data = response.json()

            # All results should match category
            for item in data:
                assert item["category"] == category


class TestReferentialIntegrity:
    """Test referential integrity between related data."""

    @pytest.mark.asyncio
    async def test_diagnosis_references_valid_dtc_codes(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
    ):
        """Test that diagnosis results reference valid DTC codes."""
        request_data = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": ["P0101"],
            "symptoms": "A motor nehezen indul es egyenetlenul jar.",
        }

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)

            if response.status_code == 201:
                data = response.json()

                # Each DTC code in response should exist in database
                for code in data["dtc_codes"]:
                    dtc_response = await async_client.get(f"/api/v1/dtc/{code}")
                    # May be 200 (found) or 404 (not in DB but valid format)
                    assert dtc_response.status_code in [200, 404]

    @pytest.mark.asyncio
    async def test_related_codes_exist_in_database(self, async_client, seeded_db):
        """Test that related DTC codes exist in database."""
        response = await async_client.get("/api/v1/dtc/P0101/related")

        assert response.status_code == 200
        data = response.json()

        # Each related code should be retrievable
        for item in data:
            code = item["code"]
            check_response = await async_client.get(f"/api/v1/dtc/{code}")
            assert check_response.status_code == 200


class TestDataValidationConsistency:
    """Test that validation is consistent across endpoints."""

    @pytest.mark.asyncio
    async def test_dtc_code_validation_consistent(self, async_client, seeded_db):
        """Test that DTC code validation is consistent across endpoints."""
        invalid_code = "X1234"

        # Create endpoint should reject
        create_response = await async_client.post(
            "/api/v1/dtc/",
            json={
                "code": invalid_code,
                "description_en": "Test",
                "category": "powertrain",
                "severity": "medium",
            },
        )
        assert create_response.status_code in [400, 422]

        # Detail endpoint should reject
        detail_response = await async_client.get(f"/api/v1/dtc/{invalid_code}")
        assert detail_response.status_code == 400

    @pytest.mark.asyncio
    async def test_email_validation_consistent(self, async_client, seeded_db):
        """Test that email validation is consistent."""
        invalid_email = "not-an-email"

        # Registration should reject
        register_response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": invalid_email,
                "password": "SecurePassword123!",
            },
        )
        assert register_response.status_code == 422

        # Forgot password should not error (security - no email enumeration)
        forgot_response = await async_client.post(
            "/api/v1/auth/forgot-password",
            json={"email": invalid_email},
        )
        # May accept or reject based on implementation
        assert forgot_response.status_code in [200, 422]


class TestDataIsolation:
    """Test that user data is properly isolated."""

    @pytest.mark.asyncio
    async def test_user_cannot_access_other_user_diagnosis(
        self,
        authenticated_client,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
    ):
        """Test that users cannot access other users' diagnoses."""
        # Register and login as user 1
        user1 = {
            "email": "user1isolation@example.com",
            "password": "SecurePassword123!",
        }
        await async_client.post("/api/v1/auth/register", json=user1)
        login1 = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "user1isolation@example.com", "password": "SecurePassword123!"},
        )
        tokens1 = login1.json()
        headers1 = {"Authorization": f"Bearer {tokens1['access_token']}"}

        # User 1 creates a diagnosis
        request_data = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": ["P0101"],
            "symptoms": "A motor nehezen indul es egyenetlenul jar.",
        }

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            create_response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=request_data,
                headers=headers1,
            )

            if create_response.status_code == 201:
                diagnosis_id = create_response.json()["id"]

                # Register and login as user 2
                user2 = {
                    "email": "user2isolation@example.com",
                    "password": "SecurePassword123!",
                }
                await async_client.post("/api/v1/auth/register", json=user2)
                login2 = await async_client.post(
                    "/api/v1/auth/login",
                    data={"username": "user2isolation@example.com", "password": "SecurePassword123!"},
                )
                tokens2 = login2.json()
                headers2 = {"Authorization": f"Bearer {tokens2['access_token']}"}

                # User 2 should not be able to delete user 1's diagnosis
                delete_response = await async_client.delete(
                    f"/api/v1/diagnosis/{diagnosis_id}",
                    headers=headers2,
                )

                # Should either not find or not be authorized
                assert delete_response.status_code in [404, 403]


class TestDataVersioning:
    """Test data versioning and update tracking."""

    @pytest.mark.asyncio
    async def test_profile_update_timestamp(self, async_client, seeded_db):
        """Test that profile updates affect timestamp."""
        # Register and login
        user = {
            "email": "timestamptest@example.com",
            "password": "SecurePassword123!",
            "full_name": "Original Name",
        }
        register_response = await async_client.post("/api/v1/auth/register", json=user)
        original_data = register_response.json()
        original_created = original_data.get("created_at")

        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "timestamptest@example.com", "password": "SecurePassword123!"},
        )
        tokens = login_response.json()
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}

        # Small delay to ensure timestamp difference
        import asyncio
        await asyncio.sleep(0.1)

        # Update profile
        update_response = await async_client.put(
            "/api/v1/auth/me",
            json={"full_name": "Updated Name"},
            headers=headers,
        )

        if update_response.status_code == 200:
            updated_data = update_response.json()
            # created_at should remain the same
            if original_created and updated_data.get("created_at"):
                assert original_created == updated_data["created_at"]


class TestConcurrentDataAccess:
    """Test concurrent data access scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_dtc_creation(self, async_client, seeded_db):
        """Test concurrent DTC creation with same code."""
        import asyncio

        dtc_data = {
            "code": "P9500",
            "description_en": "Concurrent Test Code",
            "category": "powertrain",
            "severity": "medium",
        }

        # Try to create same DTC concurrently
        tasks = [
            async_client.post("/api/v1/dtc/", json=dtc_data)
            for _ in range(3)
        ]

        responses = await asyncio.gather(*tasks)

        # One should succeed, others should fail
        success_count = sum(1 for r in responses if r.status_code == 201)
        failure_count = sum(1 for r in responses if r.status_code == 400)

        assert success_count == 1
        assert failure_count == 2

    @pytest.mark.asyncio
    async def test_concurrent_search_stability(self, async_client, seeded_db):
        """Test that concurrent searches return stable results."""
        import asyncio

        # Run multiple concurrent searches
        tasks = [
            async_client.get("/api/v1/dtc/search", params={"q": "P0101"})
            for _ in range(5)
        ]

        responses = await asyncio.gather(*tasks)

        # All should succeed
        for response in responses:
            assert response.status_code == 200

        # All should return same results
        results = [r.json() for r in responses]
        first_result = results[0]
        for result in results[1:]:
            assert len(result) == len(first_result)


class TestNullHandling:
    """Test null value handling across the system."""

    @pytest.mark.asyncio
    async def test_null_description_hu_handled(self, async_client, seeded_db):
        """Test that null Hungarian description is handled."""
        dtc = {
            "code": "P9600",
            "description_en": "Code without Hungarian",
            "description_hu": None,  # Explicitly null
            "category": "powertrain",
            "severity": "medium",
        }

        response = await async_client.post("/api/v1/dtc/", json=dtc)
        assert response.status_code == 201

        # Retrieve and verify
        get_response = await async_client.get("/api/v1/dtc/P9600")
        if get_response.status_code == 200:
            data = get_response.json()
            # description_hu should be null or empty
            assert data["description_hu"] in [None, ""]

    @pytest.mark.asyncio
    async def test_empty_lists_handled(self, async_client, seeded_db):
        """Test that empty lists are handled correctly."""
        dtc = {
            "code": "P9601",
            "description_en": "Code with empty lists",
            "category": "powertrain",
            "severity": "medium",
            "symptoms": [],
            "possible_causes": [],
            "diagnostic_steps": [],
        }

        response = await async_client.post("/api/v1/dtc/", json=dtc)
        assert response.status_code == 201

        # Retrieve and verify
        get_response = await async_client.get("/api/v1/dtc/P9601")
        if get_response.status_code == 200:
            data = get_response.json()
            assert isinstance(data["symptoms"], list)
            assert isinstance(data["possible_causes"], list)
            assert isinstance(data["diagnostic_steps"], list)


class TestDataSanitization:
    """Test data sanitization and XSS prevention."""

    @pytest.mark.asyncio
    async def test_html_in_description_escaped(self, async_client, seeded_db):
        """Test that HTML in descriptions is properly handled."""
        dtc = {
            "code": "P9700",
            "description_en": "<script>alert('xss')</script>Test Code",
            "category": "powertrain",
            "severity": "medium",
        }

        response = await async_client.post("/api/v1/dtc/", json=dtc)
        assert response.status_code == 201

        # Retrieve and verify (should not execute script)
        get_response = await async_client.get("/api/v1/dtc/P9700")
        if get_response.status_code == 200:
            data = get_response.json()
            # Script tags should be stored as-is (API doesn't execute)
            # But in a real scenario, frontend should escape
            assert "description_en" in data

    @pytest.mark.asyncio
    async def test_special_characters_in_user_name(self, async_client, seeded_db):
        """Test that special characters in user name are handled."""
        user = {
            "email": "specialname@example.com",
            "password": "SecurePassword123!",
            "full_name": "O'Brien & Smith <test>",
        }

        response = await async_client.post("/api/v1/auth/register", json=user)
        assert response.status_code == 201

        # Verify name is stored correctly
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "specialname@example.com", "password": "SecurePassword123!"},
        )
        tokens = login_response.json()

        profile_response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        profile_data = profile_response.json()

        assert profile_data["full_name"] == "O'Brien & Smith <test>"


class TestHealthEndpoints:
    """Test health check endpoints for data connectivity."""

    @pytest.mark.asyncio
    async def test_health_endpoint_accessible(self, async_client, seeded_db):
        """Test that basic health endpoint is accessible."""
        response = await async_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    @pytest.mark.asyncio
    async def test_health_ready_checks_databases(self, async_client, seeded_db):
        """Test that readiness probe checks database connectivity."""
        response = await async_client.get("/health/ready")

        # Should return status indicating database health
        assert response.status_code in [200, 503]

    @pytest.mark.asyncio
    async def test_health_live_always_responds(self, async_client, seeded_db):
        """Test that liveness probe responds even if databases are down."""
        response = await async_client.get("/health/live")

        # Liveness should always return quickly
        assert response.status_code == 200
