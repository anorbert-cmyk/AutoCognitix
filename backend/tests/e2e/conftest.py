"""
Pytest fixtures for end-to-end tests.

Provides comprehensive fixtures for:
- Database session management (PostgreSQL, Neo4j, Qdrant)
- Test client setup with authentication support
- Mock data creation for realistic test scenarios
- Service mocking for external dependencies
"""

import asyncio
from datetime import datetime, timedelta
from typing import AsyncGenerator, Generator, Dict, Any, Optional
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
# Comprehensive Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_dtc_codes():
    """Comprehensive sample DTC codes covering all categories."""
    return [
        # Powertrain codes
        DTCCode(
            id=1,
            code="P0101",
            description_en="Mass Air Flow Circuit Range/Performance",
            description_hu="Levegotomeg-mero aramkor tartomany/teljesitmeny hiba",
            category="powertrain",
            severity="medium",
            is_generic=True,
            symptoms=["Rough idle", "Poor acceleration", "Check engine light", "Increased fuel consumption"],
            possible_causes=["Dirty MAF sensor", "Air leak in intake", "Faulty MAF sensor", "Vacuum leak"],
            diagnostic_steps=["Check for vacuum leaks", "Clean MAF sensor", "Test MAF signal", "Inspect air filter"],
            related_codes=["P0100", "P0102", "P0103", "P0104"],
        ),
        DTCCode(
            id=2,
            code="P0171",
            description_en="System Too Lean (Bank 1)",
            description_hu="Rendszer tul sovany (Bank 1)",
            category="powertrain",
            severity="medium",
            is_generic=True,
            symptoms=["Poor fuel economy", "Hesitation", "Rough idle", "Engine misfires"],
            possible_causes=["Vacuum leak", "Faulty fuel injector", "Low fuel pressure", "Clogged fuel filter"],
            diagnostic_steps=["Check fuel pressure", "Inspect vacuum lines", "Check injectors", "Test O2 sensors"],
            related_codes=["P0172", "P0174", "P0175"],
        ),
        DTCCode(
            id=3,
            code="P0300",
            description_en="Random/Multiple Cylinder Misfire Detected",
            description_hu="Veletlenszeru/tobbszoros hengergyujtasi hiba eszlelve",
            category="powertrain",
            severity="high",
            is_generic=True,
            symptoms=["Engine shaking", "Loss of power", "Check engine light flashing", "Poor fuel economy"],
            possible_causes=["Bad spark plugs", "Faulty ignition coils", "Fuel delivery issues", "Compression problems"],
            diagnostic_steps=["Check spark plugs", "Test ignition coils", "Check fuel injectors", "Perform compression test"],
            related_codes=["P0301", "P0302", "P0303", "P0304"],
        ),
        # Body codes
        DTCCode(
            id=4,
            code="B1234",
            description_en="Airbag Sensor Circuit Malfunction",
            description_hu="Legzsak szenzor aramkor meghibasodas",
            category="body",
            severity="high",
            is_generic=True,
            symptoms=["Airbag warning light", "SRS system fault", "Airbag may not deploy"],
            possible_causes=["Faulty sensor", "Wiring issue", "Connector corrosion", "Impact sensor damage"],
            diagnostic_steps=["Scan SRS system", "Check sensor wiring", "Inspect connectors", "Test sensor resistance"],
            related_codes=["B1235", "B1236", "B1237"],
        ),
        DTCCode(
            id=5,
            code="B0012",
            description_en="Driver Frontal Stage 1 Deployment Control",
            description_hu="Vezeto oldali elso 1. fazisu legzsak vezerles",
            category="body",
            severity="critical",
            is_generic=True,
            symptoms=["Airbag warning light", "Driver airbag fault"],
            possible_causes=["Faulty airbag module", "Wiring harness damage", "Clock spring failure"],
            diagnostic_steps=["Check clock spring", "Inspect wiring", "Test airbag module", "Clear codes and retest"],
            related_codes=["B0013", "B0014"],
        ),
        # Chassis codes
        DTCCode(
            id=6,
            code="C0035",
            description_en="Left Front Wheel Speed Sensor Circuit",
            description_hu="Bal elso kerek sebessegmero szenzor aramkor",
            category="chassis",
            severity="medium",
            is_generic=True,
            symptoms=["ABS warning light", "Traction control disabled", "Stability control fault"],
            possible_causes=["Faulty wheel speed sensor", "Damaged wiring", "Debris on sensor", "Wheel bearing damage"],
            diagnostic_steps=["Check sensor resistance", "Inspect wiring", "Clean sensor", "Test sensor output"],
            related_codes=["C0036", "C0037", "C0038"],
        ),
        DTCCode(
            id=7,
            code="C0051",
            description_en="Rear Wheel Speed Sensor Circuit Malfunction",
            description_hu="Hatso kerek sebessegerzekelo aramkor hiba",
            category="chassis",
            severity="medium",
            is_generic=True,
            symptoms=["ABS light on", "Stability control inactive"],
            possible_causes=["Faulty sensor", "Broken tone ring", "Wiring damage"],
            diagnostic_steps=["Inspect tone ring", "Test sensor", "Check wiring continuity"],
            related_codes=["C0052", "C0053"],
        ),
        # Network codes
        DTCCode(
            id=8,
            code="U0100",
            description_en="Lost Communication With ECM/PCM",
            description_hu="Kommunikacio megszakadt az ECM/PCM-mel",
            category="network",
            severity="high",
            is_generic=True,
            symptoms=["No start condition", "Multiple warning lights", "Engine runs rough", "Limp mode"],
            possible_causes=["Faulty ECM", "CAN bus issue", "Power supply problem", "Ground connection issue"],
            diagnostic_steps=["Check ECM power", "Test CAN bus", "Verify ground connections", "Check fuses"],
            related_codes=["U0101", "U0102", "U0103"],
        ),
        DTCCode(
            id=9,
            code="U0121",
            description_en="Lost Communication with Anti-Lock Brake System Module",
            description_hu="Kommunikacio megszakadt az ABS modullal",
            category="network",
            severity="high",
            is_generic=True,
            symptoms=["ABS warning light", "Brake system fault", "Communication errors"],
            possible_causes=["Faulty ABS module", "CAN bus wiring", "Module power issue"],
            diagnostic_steps=["Check ABS module power", "Test CAN communication", "Inspect wiring"],
            related_codes=["U0122", "U0123"],
        ),
        DTCCode(
            id=10,
            code="U0140",
            description_en="Lost Communication with Body Control Module",
            description_hu="Kommunikacio megszakadt a BCM-mel",
            category="network",
            severity="medium",
            is_generic=True,
            symptoms=["Multiple electrical faults", "Interior lights issues", "Door lock problems"],
            possible_causes=["Faulty BCM", "Wiring issues", "Power supply problem"],
            diagnostic_steps=["Check BCM power", "Scan all modules", "Test CAN bus"],
            related_codes=["U0141", "U0142"],
        ),
    ]


@pytest.fixture
def sample_users():
    """Sample users with different roles for testing."""
    return [
        {
            "id": uuid4(),
            "email": "testuser@example.com",
            "hashed_password": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNiGJz5.MVNS2",  # "TestPassword123!"
            "full_name": "Test User",
            "is_active": True,
            "role": "user",
        },
        {
            "id": uuid4(),
            "email": "mechanic@example.com",
            "hashed_password": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNiGJz5.MVNS2",
            "full_name": "Test Mechanic",
            "is_active": True,
            "role": "mechanic",
        },
        {
            "id": uuid4(),
            "email": "admin@example.com",
            "hashed_password": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNiGJz5.MVNS2",
            "full_name": "Test Admin",
            "is_active": True,
            "role": "admin",
        },
        {
            "id": uuid4(),
            "email": "inactive@example.com",
            "hashed_password": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNiGJz5.MVNS2",
            "full_name": "Inactive User",
            "is_active": False,
            "role": "user",
        },
    ]


@pytest.fixture
def sample_vehicle_makes():
    """Sample vehicle makes for testing."""
    return [
        VehicleMake(id="volkswagen", name="Volkswagen", country="Germany"),
        VehicleMake(id="toyota", name="Toyota", country="Japan"),
        VehicleMake(id="ford", name="Ford", country="USA"),
        VehicleMake(id="bmw", name="BMW", country="Germany"),
        VehicleMake(id="audi", name="Audi", country="Germany"),
        VehicleMake(id="mercedes", name="Mercedes-Benz", country="Germany"),
        VehicleMake(id="honda", name="Honda", country="Japan"),
        VehicleMake(id="nissan", name="Nissan", country="Japan"),
    ]


@pytest_asyncio.fixture
async def seeded_db(db_session: AsyncSession, sample_dtc_codes, sample_users, sample_vehicle_makes):
    """Seed the database with comprehensive test data."""
    # Add DTC codes
    for dtc in sample_dtc_codes:
        db_session.add(dtc)

    # Add users
    for user_data in sample_users:
        user = User(**user_data)
        db_session.add(user)

    # Add vehicle makes
    for make in sample_vehicle_makes:
        db_session.add(make)

    await db_session.commit()

    return {
        "dtc_codes": sample_dtc_codes,
        "users": sample_users,
        "makes": sample_vehicle_makes,
    }


# =============================================================================
# Mock Service Fixtures
# =============================================================================


@pytest.fixture
def mock_nhtsa_service():
    """Comprehensive mock NHTSA service for testing."""
    from app.services.nhtsa_service import VINDecodeResult, Recall, Complaint

    mock_service = AsyncMock()

    # Mock VIN decode - Volkswagen
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
        Recall(
            campaign_number="21V456000",
            manufacturer="Volkswagen",
            make="Volkswagen",
            model="Golf",
            model_year=2018,
            recall_date="2021-06-20",
            component="ELECTRICAL SYSTEM",
            summary="Software issue may cause intermittent electrical faults.",
            consequence="Various electrical systems may malfunction.",
            remedy="Software update available at dealerships.",
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
            summary="Engine hesitates during acceleration from stop.",
        ),
        Complaint(
            odinumber="12345679",
            manufacturer="Volkswagen",
            make="Volkswagen",
            model="Golf",
            model_year=2018,
            crash=False,
            fire=False,
            injuries=0,
            deaths=0,
            complaint_date="2019-08-15",
            components="ELECTRICAL SYSTEM",
            summary="Random warning lights appear on dashboard.",
        ),
    ]

    return mock_service


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for testing without loading ML models."""
    mock_service = MagicMock()

    # Return 768-dimensional vectors with slight variations for realistic testing
    import random

    def generate_embedding():
        return [random.uniform(-0.1, 0.1) for _ in range(768)]

    mock_service.embed_text.side_effect = lambda text, preprocess=True: generate_embedding()
    mock_service.embed_batch.side_effect = lambda texts, preprocess=True: [generate_embedding() for _ in texts]
    mock_service.preprocess_hungarian.return_value = "preprocessed text"
    mock_service.embedding_dimension = 768
    mock_service.is_model_loaded = True

    return mock_service


@pytest.fixture
def mock_rag_service():
    """Comprehensive mock RAG service for diagnosis testing."""
    mock_service = AsyncMock()

    mock_service.diagnose.return_value = {
        "probable_causes": [
            {
                "title": "Levegotomeg-mero (MAF) szenzor hiba",
                "description": "A levegotomeg-mero szenzor szennyezett vagy hibas. A szenzor pontatlan jelei miatt a motor vezerloegyseg nem tudja megfeleloen szabalyozni az uzemanyag-levego kevereket.",
                "confidence": 0.85,
                "related_dtc_codes": ["P0101"],
                "components": ["MAF szenzor", "Levegoszuro"],
            },
            {
                "title": "Vakuum szivargás",
                "description": "A szivocso vagy vakuumcsovek szivargatnak, ami sovany kevereket okoz es befolyasolja a motor alapjartat.",
                "confidence": 0.72,
                "related_dtc_codes": ["P0101", "P0171"],
                "components": ["Szivocso tomitesek", "Vakuumcsovek"],
            },
            {
                "title": "Levegoszuro eltomodes",
                "description": "Az eltomott levegoszuro korlatozza a levegoaramlast, ami befolyasolja a MAF szenzor olevasasait.",
                "confidence": 0.58,
                "related_dtc_codes": ["P0101"],
                "components": ["Levegoszuro"],
            },
        ],
        "recommended_repairs": [
            {
                "title": "MAF szenzor tisztitasa",
                "description": "Tisztitsa meg a MAF szenzort specialis MAF tisztito sprayvel. Soha ne erintse meg a szenzor elemeit.",
                "difficulty": "beginner",
                "estimated_cost_min": 1500,
                "estimated_cost_max": 5000,
                "estimated_cost_currency": "HUF",
                "parts_needed": ["MAF tisztito spray"],
                "estimated_time_minutes": 30,
            },
            {
                "title": "Vakuumcsovek ellenorzese es csereje",
                "description": "Vizsgalja meg az osszes vakuumcsot repedesek es szivargas szempontjabol. Csereljee a serult csoveket.",
                "difficulty": "beginner",
                "estimated_cost_min": 2000,
                "estimated_cost_max": 10000,
                "estimated_cost_currency": "HUF",
                "parts_needed": ["Vakuumcso keszlet"],
                "estimated_time_minutes": 45,
            },
            {
                "title": "MAF szenzor csere",
                "description": "Ha a tisztitas nem segit, csereje ki a MAF szenzort uj, eredeti minusegu alkatresre.",
                "difficulty": "intermediate",
                "estimated_cost_min": 25000,
                "estimated_cost_max": 75000,
                "estimated_cost_currency": "HUF",
                "parts_needed": ["MAF szenzor"],
                "estimated_time_minutes": 60,
            },
        ],
        "confidence_score": 0.78,
        "sources": [
            {
                "type": "database",
                "title": "OBD-II DTC Database - P0101",
                "url": None,
                "relevance_score": 0.95,
            },
            {
                "type": "tsb",
                "title": "Volkswagen TSB 2019-12",
                "url": "https://example.com/tsb/2019-12",
                "relevance_score": 0.82,
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
            "symptoms": [
                {"name": "Rough idle"},
                {"name": "Poor acceleration"},
                {"name": "Check engine light"},
            ],
            "components": [
                {"name": "MAF Sensor", "system": "Engine", "failure_mode": "Signal out of range"},
                {"name": "Air Filter", "system": "Intake", "failure_mode": "Clogged"},
            ],
            "repairs": [
                {"name": "Clean MAF Sensor", "difficulty": "beginner", "description": "Use MAF cleaner spray"},
                {"name": "Replace MAF Sensor", "difficulty": "intermediate", "description": "Install new sensor"},
            ],
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
                "description_hu": "Levegotomeg-mero aramkor hiba",
                "category": "powertrain",
            },
        },
        {
            "id": "2",
            "score": 0.85,
            "payload": {
                "code": "P0171",
                "description_hu": "Rendszer tul sovany",
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
                "description_hu": "Levegotomeg-mero aramkor hiba",
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


@pytest_asyncio.fixture
async def authenticated_client(app, db_session, seeded_db) -> AsyncGenerator[Dict[str, Any], None]:
    """Create authenticated client with user token."""
    from app.db.postgres.session import get_db
    from app.core.security import create_access_token, create_refresh_token

    # Override the database dependency
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    # Get the test user
    test_user = seeded_db["users"][0]  # Regular user

    # Create tokens
    access_token = create_access_token(
        subject=str(test_user["id"]),
        additional_claims={"role": test_user["role"]},
    )
    refresh_token = create_refresh_token(subject=str(test_user["id"]))

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield {
            "client": client,
            "user": test_user,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "headers": {"Authorization": f"Bearer {access_token}"},
        }

    app.dependency_overrides.clear()


# =============================================================================
# Request/Response Data Fixtures
# =============================================================================


@pytest.fixture
def diagnosis_request_data():
    """Sample diagnosis request data for testing."""
    return {
        "vehicle_make": "Volkswagen",
        "vehicle_model": "Golf",
        "vehicle_year": 2018,
        "vehicle_engine": "2.0 TSI",
        "vin": "WVWZZZ3CZWE123456",
        "dtc_codes": ["P0101", "P0171"],
        "symptoms": "A motor nehezen indul hidegben, egyenetlenul jar alapjaraton. A fogyasztas is megnovekedett az uttobbi idoben.",
        "additional_context": "A problema telen rosszabb, de nyaron is elofordul neha.",
    }


@pytest.fixture
def minimal_diagnosis_request():
    """Minimal valid diagnosis request data."""
    return {
        "vehicle_make": "Toyota",
        "vehicle_model": "Corolla",
        "vehicle_year": 2020,
        "dtc_codes": ["P0300"],
        "symptoms": "A motor rezeg es erot veszit gyorsitaskor.",
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
def user_login_credentials():
    """Sample user login credentials matching seeded user."""
    return {
        "username": "testuser@example.com",
        "password": "TestPassword123!",
    }


@pytest.fixture
def invalid_user_credentials():
    """Invalid login credentials for error testing."""
    return {
        "username": "nonexistent@example.com",
        "password": "WrongPassword123!",
    }


# =============================================================================
# Validation Fixtures
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
        "WBAPH5C55BA123456",  # BMW
        "WAUZZZ8V0BA123456",  # Audi
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
        "0000000000000000",  # All zeros
    ]


@pytest.fixture
def valid_dtc_codes():
    """Valid DTC codes for testing."""
    return [
        "P0101", "P0171", "P0300", "P0420", "P0442",
        "B1234", "B0012", "B1500",
        "C0035", "C0051", "C0200",
        "U0100", "U0121", "U0140",
    ]


@pytest.fixture
def invalid_dtc_codes():
    """Invalid DTC codes for testing."""
    return [
        "X1234",     # Invalid prefix
        "P123",      # Too short
        "P01234",    # Too long for generic
        "PABCD",     # Non-numeric suffix
        "P0OO1",     # Contains letters instead of numbers
        "",          # Empty
        "12345",     # No prefix
    ]


@pytest.fixture
def dtc_code_categories():
    """DTC code category information."""
    return {
        "P": {"name": "Powertrain", "name_hu": "Hajtaslancm"},
        "B": {"name": "Body", "name_hu": "Karosszeria"},
        "C": {"name": "Chassis", "name_hu": "Alvazmz"},
        "U": {"name": "Network", "name_hu": "Halozat"},
    }


# =============================================================================
# Utility Functions
# =============================================================================


def create_auth_header(token: str) -> Dict[str, str]:
    """Create authorization header with bearer token."""
    return {"Authorization": f"Bearer {token}"}


def assert_valid_uuid(uuid_string: str) -> bool:
    """Assert that a string is a valid UUID."""
    import uuid
    try:
        uuid.UUID(uuid_string)
        return True
    except ValueError:
        return False


def assert_diagnosis_response_structure(data: Dict[str, Any]) -> None:
    """Assert that diagnosis response has correct structure."""
    required_fields = [
        "id", "vehicle_make", "vehicle_model", "vehicle_year",
        "dtc_codes", "symptoms", "probable_causes",
        "recommended_repairs", "confidence_score", "created_at"
    ]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"

    assert 0 <= data["confidence_score"] <= 1
    assert isinstance(data["probable_causes"], list)
    assert isinstance(data["recommended_repairs"], list)


def assert_dtc_detail_structure(data: Dict[str, Any]) -> None:
    """Assert that DTC detail response has correct structure."""
    required_fields = [
        "code", "description_en", "category", "is_generic",
        "severity", "symptoms", "possible_causes", "diagnostic_steps"
    ]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"


def assert_token_response_structure(data: Dict[str, Any]) -> None:
    """Assert that token response has correct structure."""
    required_fields = ["access_token", "refresh_token", "token_type"]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
    assert data["token_type"] == "bearer"
    assert len(data["access_token"]) > 0
    assert len(data["refresh_token"]) > 0


def assert_user_response_structure(data: Dict[str, Any]) -> None:
    """Assert that user response has correct structure."""
    required_fields = ["id", "email", "is_active", "role"]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
    assert "password" not in data
    assert "hashed_password" not in data


# =============================================================================
# Additional Fixtures for Edge Cases
# =============================================================================


@pytest.fixture
def sample_manufacturer_dtc_codes():
    """Manufacturer-specific DTC codes for testing."""
    return [
        DTCCode(
            id=101,
            code="P2004",
            description_en="Intake Manifold Runner Control Stuck Open (Bank 1)",
            description_hu="Szivocso futoszelep nyitva ragadt (Bank 1)",
            category="powertrain",
            severity="medium",
            is_generic=False,
            manufacturer_code="VAG",
            symptoms=["Rough idle", "Loss of power", "Check engine light"],
            possible_causes=["Stuck IMRC valve", "Vacuum leak", "Control solenoid failure"],
            diagnostic_steps=["Test IMRC actuator", "Check vacuum lines", "Verify solenoid operation"],
            related_codes=["P2005", "P2006", "P2007"],
        ),
        DTCCode(
            id=102,
            code="P20BA",
            description_en="Reductant Heater A Control Circuit/Open",
            description_hu="AdBlue futoelemA vezerlo aramkor nyitott",
            category="powertrain",
            severity="high",
            is_generic=False,
            manufacturer_code="BMW",
            symptoms=["AdBlue warning", "Reduced power", "Limp mode"],
            possible_causes=["Faulty heater element", "Wiring damage", "Control module issue"],
            diagnostic_steps=["Test heater resistance", "Check wiring", "Verify module output"],
            related_codes=["P20BB", "P20BC"],
        ),
    ]


@pytest.fixture
def sample_hungarian_symptoms():
    """Hungarian symptom texts for testing NLP processing."""
    return [
        "A motor nehezen indul hidegben, egyenetlenul jar alapjaraton, es a fogyasztas jelentosen megnott.",
        "A kocsi remeg gyorsitaskor, es furcsa zaj jon a motor felourol.",
        "A fel lampa vilagit, az ABS mukodik, de a fekezesnel furcsa hang van.",
        "A klima nem hut eleg hatekonyyan, es futo levego jon belole.",
        "Az automatikus valtó nehezen kapcsol, es neha megcsusztatva valt.",
        "A kormany rezeg 80 km/h felett, es a kerekek kopnak egyenetlenul.",
        "Olajfogyasztas novekedes eszlelheto, kek fust jon a kipufogóból.",
        "A motor leall melegedés után, újraindításnál nehezen indul.",
    ]


@pytest.fixture
def sample_diagnosis_sessions():
    """Sample diagnosis sessions for history testing."""
    from app.db.postgres.models import DiagnosisSession
    from datetime import datetime, timedelta

    base_time = datetime.utcnow()

    return [
        {
            "id": uuid4(),
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "vehicle_vin": "WVWZZZ3CZWE123456",
            "dtc_codes": ["P0101", "P0171"],
            "symptoms_text": "A motor nehezen indul hidegben.",
            "confidence_score": 0.85,
            "created_at": base_time - timedelta(days=1),
        },
        {
            "id": uuid4(),
            "vehicle_make": "Toyota",
            "vehicle_model": "Corolla",
            "vehicle_year": 2020,
            "vehicle_vin": "JTDKN3DU0A0123456",
            "dtc_codes": ["P0300", "P0301"],
            "symptoms_text": "A motor rezeg es erot veszit.",
            "confidence_score": 0.78,
            "created_at": base_time - timedelta(days=7),
        },
        {
            "id": uuid4(),
            "vehicle_make": "BMW",
            "vehicle_model": "3 Series",
            "vehicle_year": 2019,
            "vehicle_vin": "WBAPH5C55BA123456",
            "dtc_codes": ["U0100"],
            "symptoms_text": "Kommunikacios hiba a muszerfalon.",
            "confidence_score": 0.92,
            "created_at": base_time - timedelta(days=30),
        },
    ]


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for RAG testing."""
    mock_service = AsyncMock()

    mock_service.generate.return_value = {
        "content": """
        # Diagnózis Eredmény

        ## Valószínű okok:
        1. MAF szenzor szennyeződés (85% valószínűség)
        2. Vakuum szivárgás (72% valószínűség)

        ## Javasolt javítások:
        1. MAF szenzor tisztítása
        2. Vakuumcsövek ellenőrzése
        """,
        "usage": {
            "prompt_tokens": 500,
            "completion_tokens": 200,
            "total_tokens": 700,
        },
    }

    return mock_service


@pytest.fixture
def mock_redis_cache():
    """Mock Redis cache for testing cache behavior."""
    cache_data = {}

    mock_cache = AsyncMock()

    async def mock_get(key):
        return cache_data.get(key)

    async def mock_set(key, value, ttl=None):
        cache_data[key] = value

    async def mock_delete(key):
        cache_data.pop(key, None)

    async def mock_delete_pattern(pattern):
        keys_to_delete = [k for k in cache_data if pattern.replace("*", "") in k]
        for k in keys_to_delete:
            cache_data.pop(k, None)

    mock_cache.get = mock_get
    mock_cache.set = mock_set
    mock_cache.delete = mock_delete
    mock_cache.delete_pattern = mock_delete_pattern
    mock_cache.get_dtc_code = mock_get
    mock_cache.set_dtc_code = mock_set
    mock_cache.get_dtc_search_results = mock_get
    mock_cache.set_dtc_search_results = mock_set
    mock_cache.get_stats.return_value = {"status": "connected", "hits": 0, "misses": 0}

    return mock_cache


@pytest.fixture
def edge_case_dtc_codes():
    """Edge case DTC codes for boundary testing."""
    return [
        "P0000",  # Minimum valid code
        "P9999",  # Maximum generic code
        "P1000",  # Manufacturer-specific start
        "P3FFF",  # Maximum manufacturer-specific (hex)
        "B0000",  # Body minimum
        "C0000",  # Chassis minimum
        "U0000",  # Network minimum
        "U3FFF",  # Network maximum
    ]


@pytest.fixture
def rate_limit_test_data():
    """Data for rate limiting tests."""
    return {
        "requests_per_minute": 60,
        "requests_per_hour": 1000,
        "burst_limit": 10,
    }


@pytest.fixture
def concurrent_request_count():
    """Number of concurrent requests for stress testing."""
    return 10


# =============================================================================
# Admin User Fixture
# =============================================================================


@pytest_asyncio.fixture
async def admin_client(app, db_session, seeded_db) -> AsyncGenerator[Dict[str, Any], None]:
    """Create authenticated client with admin token."""
    from app.db.postgres.session import get_db
    from app.core.security import create_access_token, create_refresh_token

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    # Get the admin user from seeded data
    admin_user = next(u for u in seeded_db["users"] if u["role"] == "admin")

    access_token = create_access_token(
        subject=str(admin_user["id"]),
        additional_claims={"role": admin_user["role"]},
    )
    refresh_token = create_refresh_token(subject=str(admin_user["id"]))

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield {
            "client": client,
            "user": admin_user,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "headers": {"Authorization": f"Bearer {access_token}"},
        }

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def mechanic_client(app, db_session, seeded_db) -> AsyncGenerator[Dict[str, Any], None]:
    """Create authenticated client with mechanic token."""
    from app.db.postgres.session import get_db
    from app.core.security import create_access_token, create_refresh_token

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    # Get the mechanic user from seeded data
    mechanic_user = next(u for u in seeded_db["users"] if u["role"] == "mechanic")

    access_token = create_access_token(
        subject=str(mechanic_user["id"]),
        additional_claims={"role": mechanic_user["role"]},
    )
    refresh_token = create_refresh_token(subject=str(mechanic_user["id"]))

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield {
            "client": client,
            "user": mechanic_user,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "headers": {"Authorization": f"Bearer {access_token}"},
        }

    app.dependency_overrides.clear()
