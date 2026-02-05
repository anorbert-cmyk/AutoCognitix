"""
Pytest fixtures for integration tests.

Provides async database sessions, test clients, and mock services
for comprehensive integration testing.
"""

import asyncio
from datetime import datetime
from typing import AsyncGenerator, Generator
from uuid import uuid4
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_path))

from app.db.postgres.models import Base, DTCCode, User, DiagnosisSession, VehicleMake, VehicleModel


# =============================================================================
# Event Loop Configuration
# =============================================================================


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Database Fixtures
# =============================================================================


@pytest_asyncio.fixture(scope="function")
async def async_engine():
    """Create an async SQLite engine for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        echo=False,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a database session for testing."""
    session_factory = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    async with session_factory() as session:
        yield session
        await session.rollback()


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_dtc_codes():
    """Sample DTC codes for testing."""
    return [
        DTCCode(
            id=1,
            code="P0101",
            description_en="Mass Air Flow Circuit Range/Performance",
            description_hu="Levegotomeg-mero aramkor tartomany/teljesitmeny hiba",
            category="powertrain",
            severity="medium",
            is_generic=True,
            symptoms=["Rough idle", "Poor acceleration", "Check engine light"],
            possible_causes=["Dirty MAF sensor", "Air leak in intake", "Faulty MAF sensor"],
            diagnostic_steps=["Check for vacuum leaks", "Clean MAF sensor", "Test MAF signal"],
            related_codes=["P0100", "P0102", "P0103"],
        ),
        DTCCode(
            id=2,
            code="P0171",
            description_en="System Too Lean (Bank 1)",
            description_hu="Rendszer tul sovany (Bank 1)",
            category="powertrain",
            severity="medium",
            is_generic=True,
            symptoms=["Poor fuel economy", "Hesitation", "Rough idle"],
            possible_causes=["Vacuum leak", "Faulty fuel injector", "Low fuel pressure"],
            diagnostic_steps=["Check fuel pressure", "Inspect vacuum lines", "Check injectors"],
            related_codes=["P0172", "P0174", "P0175"],
        ),
        DTCCode(
            id=3,
            code="B1234",
            description_en="Airbag Sensor Circuit Malfunction",
            description_hu="Legzsak szenzor aramkor meghibasodas",
            category="body",
            severity="high",
            is_generic=True,
            symptoms=["Airbag warning light", "SRS system fault"],
            possible_causes=["Faulty sensor", "Wiring issue", "Connector corrosion"],
            diagnostic_steps=["Scan SRS system", "Check sensor wiring", "Inspect connectors"],
            related_codes=["B1235", "B1236"],
        ),
        DTCCode(
            id=4,
            code="C0035",
            description_en="Left Front Wheel Speed Sensor Circuit",
            description_hu="Bal elso kerek sebessegmero szenzor aramkor",
            category="chassis",
            severity="medium",
            is_generic=True,
            symptoms=["ABS warning light", "Traction control disabled"],
            possible_causes=["Faulty wheel speed sensor", "Damaged wiring", "Debris on sensor"],
            diagnostic_steps=["Check sensor resistance", "Inspect wiring", "Clean sensor"],
            related_codes=["C0036", "C0037"],
        ),
        DTCCode(
            id=5,
            code="U0100",
            description_en="Lost Communication With ECM/PCM",
            description_hu="Kommunikacio megszakadt az ECM/PCM-mel",
            category="network",
            severity="high",
            is_generic=True,
            symptoms=["No start condition", "Multiple warning lights"],
            possible_causes=["Faulty ECM", "CAN bus issue", "Power supply problem"],
            diagnostic_steps=["Check ECM power", "Test CAN bus", "Verify ground connections"],
            related_codes=["U0101", "U0102"],
        ),
    ]


@pytest_asyncio.fixture
async def seeded_db(db_session: AsyncSession, sample_dtc_codes):
    """Seed the database with test data."""
    for dtc in sample_dtc_codes:
        db_session.add(dtc)

    # Add a test user
    test_user = User(
        id=uuid4(),
        email="test@example.com",
        hashed_password="$2b$12$test_hashed_password",
        full_name="Test User",
        is_active=True,
        role="user",
    )
    db_session.add(test_user)

    # Add vehicle makes
    makes = [
        VehicleMake(id="volkswagen", name="Volkswagen", country="Germany"),
        VehicleMake(id="toyota", name="Toyota", country="Japan"),
        VehicleMake(id="ford", name="Ford", country="USA"),
    ]
    for make in makes:
        db_session.add(make)

    await db_session.commit()

    return {
        "dtc_codes": sample_dtc_codes,
        "user": test_user,
        "makes": makes,
    }


# =============================================================================
# Mock Service Fixtures
# =============================================================================


@pytest.fixture
def mock_nhtsa_service():
    """Mock NHTSA service for testing without external API calls."""
    from app.services.nhtsa_service import VINDecodeResult, Recall, Complaint

    mock_service = AsyncMock()

    # Mock VIN decode
    mock_service.decode_vin.return_value = VINDecodeResult(
        vin="WVWZZZ3CZWE123456",
        make="Volkswagen",
        model="Golf",
        model_year=2018,
        body_class="Hatchback",
        vehicle_type="Passenger Car",
        plant_country="Germany",
        engine_cylinders=4,
        engine_displacement_l=2.0,
        fuel_type_primary="Gasoline",
        transmission_style="Automatic",
        drive_type="FWD",
        raw_data={},
    )

    # Mock recalls
    mock_service.get_recalls.return_value = [
        Recall(
            campaign_number="20V123000",
            manufacturer="Volkswagen",
            make="Volkswagen",
            model="Golf",
            model_year=2018,
            recall_date="2020-03-15",
            component="FUEL SYSTEM, GASOLINE:FUEL PUMP",
            summary="The fuel pump may fail, causing the engine to stall.",
            consequence="Engine stall increases risk of crash.",
            remedy="Dealers will replace the fuel pump free of charge.",
        ),
    ]

    # Mock complaints
    mock_service.get_complaints.return_value = [
        Complaint(
            odinumber="12345678",
            manufacturer="Volkswagen",
            make="Volkswagen",
            model="Golf",
            model_year=2018,
            crash=False,
            fire=False,
            injuries=0,
            deaths=0,
            complaint_date="2019-06-01",
            components="ENGINE",
            summary="Engine hesitates during acceleration.",
        ),
    ]

    return mock_service


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for testing without loading ML models."""
    mock_service = MagicMock()

    # Return 768-dimensional zero vectors for simplicity
    mock_service.embed_text.return_value = [0.0] * 768
    mock_service.embed_batch.return_value = [[0.0] * 768]
    mock_service.preprocess_hungarian.return_value = "preprocessed text"
    mock_service.embedding_dimension = 768
    mock_service.is_model_loaded = True

    return mock_service


@pytest.fixture
def mock_rag_service():
    """Mock RAG service for testing without LLM calls."""
    mock_service = AsyncMock()

    mock_service.diagnose.return_value = {
        "probable_causes": [
            {
                "title": "Mass Air Flow Sensor Issue",
                "description": "A levegotomeg-mero szenzor hibaja valoszinu.",
                "confidence": 0.85,
                "related_dtc_codes": ["P0101"],
                "components": ["MAF Sensor"],
            },
        ],
        "recommended_repairs": [
            {
                "title": "MAF Sensor Cleaning",
                "description": "Tisztitsa meg a levegotomeg-mero szenzort.",
                "difficulty": "beginner",
                "estimated_cost_min": 0,
                "estimated_cost_max": 5000,
                "estimated_cost_currency": "HUF",
                "parts_needed": ["MAF cleaner spray"],
            },
        ],
        "confidence_score": 0.75,
        "sources": [
            {
                "type": "database",
                "title": "DTC Database - P0101",
                "url": None,
                "relevance_score": 0.85,
            },
        ],
    }

    return mock_service


@pytest.fixture
def mock_neo4j_client():
    """Mock Neo4j client for graph query testing."""
    mock_client = AsyncMock()

    mock_client.run.return_value = [
        {
            "dtc": {"code": "P0101", "description": "MAF Circuit Issue"},
            "symptoms": [{"name": "Rough idle"}, {"name": "Poor acceleration"}],
            "components": [{"name": "MAF Sensor", "system": "Engine"}],
            "repairs": [{"name": "Replace MAF Sensor", "difficulty": "beginner"}],
        }
    ]

    return mock_client


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for vector search testing."""
    mock_client = AsyncMock()

    mock_client.search.return_value = [
        {
            "id": "1",
            "score": 0.92,
            "payload": {
                "code": "P0101",
                "description_hu": "Levegotomeg-mero hiba",
                "category": "powertrain",
            },
        },
    ]

    mock_client.search_dtc.return_value = [
        {
            "id": "1",
            "score": 0.92,
            "payload": {
                "code": "P0101",
                "description_hu": "Levegotomeg-mero hiba",
            },
        },
    ]

    mock_client.search_similar_symptoms.return_value = [
        {
            "id": "2",
            "score": 0.85,
            "payload": {
                "description": "Motor nehezen indul",
                "related_dtc": ["P0101", "P0171"],
            },
        },
    ]

    return mock_client


# =============================================================================
# FastAPI Test Client Fixtures
# =============================================================================


@pytest.fixture
def app():
    """Create FastAPI application for testing."""
    from fastapi import FastAPI
    from fastapi.responses import ORJSONResponse

    test_app = FastAPI(default_response_class=ORJSONResponse)

    # Import routers
    from app.api.v1.endpoints.dtc_codes import router as dtc_router
    from app.api.v1.endpoints.diagnosis import router as diagnosis_router
    from app.api.v1.endpoints.vehicles import router as vehicles_router
    from app.api.v1.endpoints.auth import router as auth_router
    from app.api.v1.endpoints.health import router as health_router

    test_app.include_router(dtc_router, prefix="/api/v1/dtc", tags=["DTC"])
    test_app.include_router(diagnosis_router, prefix="/api/v1/diagnosis", tags=["Diagnosis"])
    test_app.include_router(vehicles_router, prefix="/api/v1/vehicles", tags=["Vehicles"])
    test_app.include_router(auth_router, prefix="/api/v1/auth", tags=["Auth"])
    test_app.include_router(health_router, prefix="/api/v1", tags=["Health"])

    return test_app


@pytest_asyncio.fixture
async def async_client(app, db_session) -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client for testing."""
    from app.db.postgres.session import get_db

    # Override the database dependency
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


# =============================================================================
# Request/Response Fixtures
# =============================================================================


@pytest.fixture
def diagnosis_request_data():
    """Sample diagnosis request data."""
    return {
        "vehicle_make": "Volkswagen",
        "vehicle_model": "Golf",
        "vehicle_year": 2018,
        "vehicle_engine": "2.0 TSI",
        "vin": "WVWZZZ3CZWE123456",
        "dtc_codes": ["P0101", "P0171"],
        "symptoms": "A motor nehezen indul hidegben, egyenetlenul jar alapjaraton.",
        "additional_context": "A problema telen rosszabb.",
    }


@pytest.fixture
def user_registration_data():
    """Sample user registration data."""
    return {
        "email": "newuser@example.com",
        "password": "SecurePassword123!",
        "full_name": "New Test User",
    }


@pytest.fixture
def user_login_data():
    """Sample user login data."""
    return {
        "username": "test@example.com",
        "password": "TestPassword123!",
    }


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def valid_vins():
    """List of valid VIN numbers for testing."""
    return [
        "WVWZZZ3CZWE123456",  # Volkswagen
        "JTDKN3DU0A0123456",  # Toyota
        "1FAHP3F29CL123456",  # Ford
        "WDB9634031L123456",  # Mercedes
        "JN1GANR35U0123456",  # Nissan
    ]


@pytest.fixture
def invalid_vins():
    """List of invalid VIN numbers for testing."""
    return [
        "INVALID",           # Too short
        "WVWZZZ3CZWE12345O", # Contains invalid character O
        "WVWZZZ3CZWE12345I", # Contains invalid character I
        "WVWZZZ3CZWE12345Q", # Contains invalid character Q
        "WVWZZZ3CZWE",       # Too short
        "WVWZZZ3CZWE123456789", # Too long
    ]


@pytest.fixture
def dtc_code_categories():
    """DTC code category prefixes and their meanings."""
    return {
        "P": {"name": "Powertrain", "name_hu": "Hajtaslancm"},
        "B": {"name": "Body", "name_hu": "Karosszeria"},
        "C": {"name": "Chassis", "name_hu": "Alvazmz"},
        "U": {"name": "Network", "name_hu": "Halozat"},
    }
