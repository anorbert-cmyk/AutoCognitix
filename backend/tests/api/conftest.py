"""
Pytest fixtures for API tests.

Provides comprehensive fixtures for testing all API endpoints including:
- Async database sessions with in-memory SQLite
- Test HTTP client with proper dependency overrides
- Test user creation with authentication tokens
- Mock services for Neo4j, Qdrant, NHTSA, embedding, and RAG
- Sample data fixtures for DTC codes, vehicles, diagnoses
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import AsyncGenerator, Generator
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.security import get_password_hash, create_access_token, create_refresh_token
from app.db.postgres.models import (
    Base,
    DTCCode,
    DiagnosisSession,
    User,
    VehicleMake,
    VehicleModel,
)


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
# Test User Fixtures
# =============================================================================


@pytest.fixture
def test_user_password() -> str:
    """Password for the test user."""
    return "TestPassword123!"


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession, test_user_password: str) -> User:
    """Create a test user in the database."""
    user = User(
        id=uuid4(),
        email="testuser@example.com",
        hashed_password=get_password_hash(test_user_password),
        full_name="Test User",
        is_active=True,
        role="user",
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def admin_user(db_session: AsyncSession, test_user_password: str) -> User:
    """Create an admin user in the database."""
    user = User(
        id=uuid4(),
        email="admin@example.com",
        hashed_password=get_password_hash(test_user_password),
        full_name="Admin User",
        is_active=True,
        role="admin",
        is_superuser=True,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def inactive_user(db_session: AsyncSession, test_user_password: str) -> User:
    """Create an inactive user in the database."""
    user = User(
        id=uuid4(),
        email="inactive@example.com",
        hashed_password=get_password_hash(test_user_password),
        full_name="Inactive User",
        is_active=False,
        role="user",
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
def user_access_token(test_user: User) -> str:
    """Generate an access token for the test user."""
    return create_access_token(
        subject=str(test_user.id),
        additional_claims={"role": test_user.role},
    )


@pytest.fixture
def user_refresh_token(test_user: User) -> str:
    """Generate a refresh token for the test user."""
    return create_refresh_token(subject=str(test_user.id))


@pytest.fixture
def admin_access_token(admin_user: User) -> str:
    """Generate an access token for the admin user."""
    return create_access_token(
        subject=str(admin_user.id),
        additional_claims={"role": admin_user.role},
    )


@pytest.fixture
def auth_headers(user_access_token: str) -> dict[str, str]:
    """Authorization headers for authenticated requests."""
    return {"Authorization": f"Bearer {user_access_token}"}


@pytest.fixture
def admin_auth_headers(admin_access_token: str) -> dict[str, str]:
    """Authorization headers for admin requests."""
    return {"Authorization": f"Bearer {admin_access_token}"}


# =============================================================================
# DTC Code Fixtures
# =============================================================================


@pytest.fixture
def sample_dtc_codes_data() -> list[dict]:
    """Sample DTC code data for creating test DTCs."""
    return [
        {
            "code": "P0101",
            "description_en": "Mass Air Flow Circuit Range/Performance",
            "description_hu": "Levegotomeg-mero aramkor tartomany/teljesitmeny hiba",
            "category": "powertrain",
            "severity": "medium",
            "is_generic": True,
            "system": "Fuel and Air Metering",
            "symptoms": ["Rough idle", "Poor acceleration", "Check engine light"],
            "possible_causes": ["Dirty MAF sensor", "Air leak in intake", "Faulty MAF sensor"],
            "diagnostic_steps": ["Check for vacuum leaks", "Clean MAF sensor", "Test MAF signal"],
            "related_codes": ["P0100", "P0102", "P0103"],
        },
        {
            "code": "P0171",
            "description_en": "System Too Lean (Bank 1)",
            "description_hu": "Rendszer tul sovany (Bank 1)",
            "category": "powertrain",
            "severity": "medium",
            "is_generic": True,
            "system": "Fuel and Air Metering",
            "symptoms": ["Poor fuel economy", "Hesitation", "Rough idle"],
            "possible_causes": ["Vacuum leak", "Faulty fuel injector", "Low fuel pressure"],
            "diagnostic_steps": ["Check fuel pressure", "Inspect vacuum lines", "Check injectors"],
            "related_codes": ["P0172", "P0174", "P0175"],
        },
        {
            "code": "B1234",
            "description_en": "Airbag Sensor Circuit Malfunction",
            "description_hu": "Legzsak szenzor aramkor meghibasodas",
            "category": "body",
            "severity": "high",
            "is_generic": True,
            "system": "Supplemental Restraint System",
            "symptoms": ["Airbag warning light", "SRS system fault"],
            "possible_causes": ["Faulty sensor", "Wiring issue", "Connector corrosion"],
            "diagnostic_steps": ["Scan SRS system", "Check sensor wiring", "Inspect connectors"],
            "related_codes": ["B1235", "B1236"],
        },
        {
            "code": "C0035",
            "description_en": "Left Front Wheel Speed Sensor Circuit",
            "description_hu": "Bal elso kerek sebessegmero szenzor aramkor",
            "category": "chassis",
            "severity": "medium",
            "is_generic": True,
            "system": "Anti-Lock Brake System",
            "symptoms": ["ABS warning light", "Traction control disabled"],
            "possible_causes": ["Faulty wheel speed sensor", "Damaged wiring", "Debris on sensor"],
            "diagnostic_steps": ["Check sensor resistance", "Inspect wiring", "Clean sensor"],
            "related_codes": ["C0036", "C0037"],
        },
        {
            "code": "U0100",
            "description_en": "Lost Communication With ECM/PCM",
            "description_hu": "Kommunikacio megszakadt az ECM/PCM-mel",
            "category": "network",
            "severity": "high",
            "is_generic": True,
            "system": "Communication Network",
            "symptoms": ["No start condition", "Multiple warning lights"],
            "possible_causes": ["Faulty ECM", "CAN bus issue", "Power supply problem"],
            "diagnostic_steps": ["Check ECM power", "Test CAN bus", "Verify ground connections"],
            "related_codes": ["U0101", "U0102"],
        },
    ]


@pytest_asyncio.fixture
async def sample_dtc_codes(
    db_session: AsyncSession, sample_dtc_codes_data: list[dict]
) -> list[DTCCode]:
    """Create sample DTC codes in the database."""
    dtc_codes = []
    for i, data in enumerate(sample_dtc_codes_data, start=1):
        dtc = DTCCode(id=i, **data)
        db_session.add(dtc)
        dtc_codes.append(dtc)

    await db_session.commit()
    return dtc_codes


# =============================================================================
# Vehicle Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def sample_vehicle_makes(db_session: AsyncSession) -> list[VehicleMake]:
    """Create sample vehicle makes in the database."""
    makes = [
        VehicleMake(id="volkswagen", name="Volkswagen", country="Germany"),
        VehicleMake(id="toyota", name="Toyota", country="Japan"),
        VehicleMake(id="ford", name="Ford", country="USA"),
        VehicleMake(id="bmw", name="BMW", country="Germany"),
        VehicleMake(id="audi", name="Audi", country="Germany"),
    ]
    for make in makes:
        db_session.add(make)
    await db_session.commit()
    return makes


@pytest_asyncio.fixture
async def sample_vehicle_models(
    db_session: AsyncSession, sample_vehicle_makes: list[VehicleMake]
) -> list[VehicleModel]:
    """Create sample vehicle models in the database."""
    models = [
        VehicleModel(id="golf", name="Golf", make_id="volkswagen", year_start=1974),
        VehicleModel(id="passat", name="Passat", make_id="volkswagen", year_start=1973),
        VehicleModel(id="camry", name="Camry", make_id="toyota", year_start=1982),
        VehicleModel(id="focus", name="Focus", make_id="ford", year_start=1998),
    ]
    for model in models:
        db_session.add(model)
    await db_session.commit()
    return models


# =============================================================================
# Diagnosis Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def sample_diagnosis_session(db_session: AsyncSession, test_user: User) -> DiagnosisSession:
    """Create a sample diagnosis session in the database."""
    session = DiagnosisSession(
        id=uuid4(),
        user_id=test_user.id,
        vehicle_make="Volkswagen",
        vehicle_model="Golf",
        vehicle_year=2018,
        vehicle_vin="WVWZZZ3CZWE123456",
        dtc_codes=["P0101", "P0171"],
        symptoms_text="A motor nehezen indul hidegben, egyenetlenul jar alapjaraton.",
        additional_context="A problema telen rosszabb.",
        diagnosis_result={
            "probable_causes": [
                {
                    "title": "MAF szenzor hiba",
                    "description": "A levegotomeg-mero szenzor hibas vagy szennyezett.",
                    "confidence": 0.85,
                    "related_dtc_codes": ["P0101"],
                    "components": ["MAF szenzor", "Levegoszuro"],
                }
            ],
            "recommended_repairs": [
                {
                    "title": "MAF szenzor tisztitasa/csereje",
                    "description": "Ellenorizze es tisztitsa meg a MAF szenzort.",
                    "estimated_cost_min": 5000,
                    "estimated_cost_max": 45000,
                    "estimated_cost_currency": "HUF",
                    "difficulty": "intermediate",
                    "parts_needed": ["MAF szenzor tisztito"],
                    "estimated_time_minutes": 30,
                }
            ],
            "sources": [
                {"type": "database", "title": "OBD-II DTC Database", "relevance_score": 0.95}
            ],
        },
        confidence_score=0.82,
        created_at=datetime.utcnow(),
    )
    db_session.add(session)
    await db_session.commit()
    await db_session.refresh(session)
    return session


@pytest_asyncio.fixture
async def multiple_diagnosis_sessions(
    db_session: AsyncSession, test_user: User
) -> list[DiagnosisSession]:
    """Create multiple diagnosis sessions for pagination and filter testing."""
    sessions = []
    vehicles = [
        ("Volkswagen", "Golf", 2018),
        ("Volkswagen", "Passat", 2020),
        ("Toyota", "Camry", 2019),
        ("BMW", "3 Series", 2021),
        ("Audi", "A4", 2020),
    ]

    for i, (make, model, year) in enumerate(vehicles):
        session = DiagnosisSession(
            id=uuid4(),
            user_id=test_user.id,
            vehicle_make=make,
            vehicle_model=model,
            vehicle_year=year,
            dtc_codes=["P0101"],
            symptoms_text=f"Teszt tunet a {make} {model} jarmuhoz.",
            diagnosis_result={
                "probable_causes": [],
                "recommended_repairs": [],
                "sources": [],
            },
            confidence_score=0.7 + (i * 0.05),
            created_at=datetime.utcnow(),
        )
        db_session.add(session)
        sessions.append(session)

    await db_session.commit()
    return sessions


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
                "estimated_time_minutes": 15,
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


@pytest.fixture
def mock_redis_cache():
    """Mock Redis cache for testing."""
    mock_cache = AsyncMock()

    mock_cache.get.return_value = None
    mock_cache.set.return_value = True
    mock_cache.delete.return_value = True
    mock_cache.delete_pattern.return_value = 0
    mock_cache.get_dtc_code.return_value = None
    mock_cache.set_dtc_code.return_value = True
    mock_cache.get_dtc_search_results.return_value = None
    mock_cache.set_dtc_search_results.return_value = True

    return mock_cache


# =============================================================================
# FastAPI Application and Client Fixtures
# =============================================================================


@pytest.fixture
def app(
    mock_embedding_service,
    mock_qdrant_client,
    mock_neo4j_client,
    mock_redis_cache,
):
    """Create FastAPI application for testing with mocked dependencies."""
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

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def authenticated_client(
    app, db_session, test_user, user_access_token
) -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client with authentication headers."""
    from app.db.postgres.session import get_db

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {user_access_token}"},
    ) as client:
        yield client

    app.dependency_overrides.clear()


# =============================================================================
# Request Data Fixtures
# =============================================================================


@pytest.fixture
def user_registration_data() -> dict:
    """Sample user registration data."""
    return {
        "email": "newuser@example.com",
        "password": "SecurePassword123!",
        "full_name": "New Test User",
    }


@pytest.fixture
def user_login_data() -> dict:
    """Sample user login data (OAuth2 form format)."""
    return {
        "username": "testuser@example.com",
        "password": "TestPassword123!",
    }


@pytest.fixture
def diagnosis_request_data() -> dict:
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
def dtc_create_data() -> dict:
    """Sample DTC creation data."""
    return {
        "code": "P9999",
        "description_en": "Test DTC Code for Testing",
        "description_hu": "Teszt DTC kod teszteleshez",
        "category": "powertrain",
        "severity": "medium",
        "is_generic": True,
        "system": "Test System",
        "symptoms": ["Test symptom 1", "Test symptom 2"],
        "possible_causes": ["Test cause 1", "Test cause 2"],
        "diagnostic_steps": ["Test step 1", "Test step 2"],
        "related_codes": ["P0101", "P0102"],
    }


# =============================================================================
# VIN Fixtures
# =============================================================================


@pytest.fixture
def valid_vins() -> list[str]:
    """List of valid VIN numbers for testing."""
    return [
        "WVWZZZ3CZWE123456",  # Volkswagen
        "JTDKN3DU0A0123456",  # Toyota
        "1FAHP3F29CL123456",  # Ford
        "WDB9634031L123456",  # Mercedes
        "JN1GANR35U0123456",  # Nissan
    ]


@pytest.fixture
def invalid_vins() -> list[str]:
    """List of invalid VIN numbers for testing."""
    return [
        "INVALID",  # Too short
        "WVWZZZ3CZWE12345O",  # Contains invalid character O
        "WVWZZZ3CZWE12345I",  # Contains invalid character I
        "WVWZZZ3CZWE12345Q",  # Contains invalid character Q
        "WVWZZZ3CZWE",  # Too short
        "WVWZZZ3CZWE123456789",  # Too long
    ]


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def dtc_categories() -> list[dict]:
    """DTC code categories with descriptions."""
    return [
        {
            "code": "P",
            "name": "Powertrain",
            "name_hu": "Hajtaslánc",
            "description": "Engine, transmission, and emission systems",
            "description_hu": "Motor, váltó és emissziós rendszerek",
        },
        {
            "code": "B",
            "name": "Body",
            "name_hu": "Karosszéria",
            "description": "Body systems including airbags, A/C, lighting",
            "description_hu": "Karosszéria rendszerek: légzsákok, klíma, világítás",
        },
        {
            "code": "C",
            "name": "Chassis",
            "name_hu": "Alváz",
            "description": "Chassis systems including ABS, steering, suspension",
            "description_hu": "Alváz rendszerek: ABS, kormányzás, felfüggesztés",
        },
        {
            "code": "U",
            "name": "Network",
            "name_hu": "Hálózat",
            "description": "Communication network and module systems",
            "description_hu": "Kommunikációs hálózat és vezérlő modulok",
        },
    ]
