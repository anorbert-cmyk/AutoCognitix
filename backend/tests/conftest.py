"""
Pytest configuration and fixtures for AutoCognitix tests.
"""

import os
import sys
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_path))

# Set environment variables for testing BEFORE any imports
# These need to be set before the modules are imported
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test_db")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test_password")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("SECRET_KEY", "test_secret_key_for_testing_only")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_for_testing_only")
os.environ.setdefault("ENVIRONMENT", "testing")

import pytest
from unittest.mock import patch


@pytest.fixture
def mock_settings():
    """Mock settings for testing without external dependencies."""
    with patch("app.core.config.settings") as mock:
        mock.HUBERT_MODEL = "SZTAKI-HLT/hubert-base-cc"
        mock.EMBEDDING_DIMENSION = 768
        mock.HUSPACY_MODEL = "hu_core_news_lg"
        mock.PROJECT_NAME = "AutoCognitix"
        mock.API_V1_PREFIX = "/api/v1"
        mock.DATABASE_URL = "postgresql+asyncpg://test:test@localhost:5432/test"
        mock.NEO4J_URI = "bolt://localhost:7687"
        mock.QDRANT_HOST = "localhost"
        mock.QDRANT_PORT = 6333
        yield mock


@pytest.fixture
def sample_dtc_codes():
    """Sample DTC codes for testing."""
    return [
        {
            "code": "P0101",
            "description_en": "Mass Air Flow Circuit Range/Performance",
            "description_hu": "Levegotomeg-mero aramkor tartomany/teljesitmeny hiba",
            "category": "powertrain",
            "severity": "medium",
            "is_generic": True,
        },
        {
            "code": "P0171",
            "description_en": "System Too Lean (Bank 1)",
            "description_hu": "Rendszer tul sovany (Bank 1)",
            "category": "powertrain",
            "severity": "medium",
            "is_generic": True,
        },
        {
            "code": "B1234",
            "description_en": "Airbag Sensor Circuit Malfunction",
            "description_hu": "Legzsak szenzor aramkor meghibasodas",
            "category": "body",
            "severity": "high",
            "is_generic": True,
        },
        {
            "code": "C0035",
            "description_en": "Left Front Wheel Speed Sensor Circuit",
            "description_hu": "Bal elso kerek sebessegmero szenzor aramkor",
            "category": "chassis",
            "severity": "medium",
            "is_generic": True,
        },
        {
            "code": "U0100",
            "description_en": "Lost Communication With ECM/PCM",
            "description_hu": "Kommunikacio megszakadt az ECM/PCM-mel",
            "category": "network",
            "severity": "high",
            "is_generic": True,
        },
    ]
