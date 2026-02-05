"""
Application configuration using Pydantic Settings.
Loads configuration from environment variables and .env file.
Supports Railway deployment with automatic environment variable detection.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Union

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Application
    PROJECT_NAME: str = "AutoCognitix"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    API_V1_PREFIX: str = "/api/v1"

    # Railway-specific
    RAILWAY_ENVIRONMENT: Optional[str] = None
    PORT: int = 8000

    # Security - IMPORTANT: These MUST be set via environment variables in production
    # Generate with: openssl rand -hex 32
    SECRET_KEY: str = ""  # Required - will fail startup if not set
    JWT_SECRET_KEY: str = ""  # Required - will fail startup if not set
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS - Use Union type to handle both string (from env) and list formats
    BACKEND_CORS_ORIGINS: Union[List[str], str] = ["http://localhost:3000", "http://localhost:8000"]

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v):
        """Convert comma-separated string to list, or return list as-is."""
        if isinstance(v, str):
            # Handle comma-separated string format from environment variables
            if v.startswith("["):
                # JSON format - let pydantic handle it
                import json
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            # Comma-separated format
            return [i.strip() for i in v.split(",")]
        return v

    # PostgreSQL - Set via environment variables
    POSTGRES_USER: str = "autocognitix"
    POSTGRES_PASSWORD: str = ""  # Required in production
    POSTGRES_DB: str = "autocognitix"
    DATABASE_URL: str = ""  # Required - set via DATABASE_URL env var

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def convert_database_url(cls, v: str) -> str:
        """Convert Railway's DATABASE_URL to asyncpg format.

        Railway provides DATABASE_URL with postgresql:// prefix,
        but asyncpg requires postgresql+asyncpg:// prefix.
        """
        if v and v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    # Neo4j - Set via environment variables
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = ""  # Required in production

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_URL: Optional[str] = None  # For Qdrant Cloud: https://xxx.cloud.qdrant.io:6333
    QDRANT_API_KEY: Optional[str] = None  # For Qdrant Cloud authentication

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # External APIs
    NHTSA_API_BASE_URL: str = "https://vpic.nhtsa.dot.gov/api"
    YOUTUBE_API_KEY: Optional[str] = None
    CARMD_API_KEY: Optional[str] = None
    CARMD_PARTNER_TOKEN: Optional[str] = None
    CARAPI_API_KEY: Optional[str] = None

    # LLM Configuration
    LLM_PROVIDER: str = "anthropic"  # openai, anthropic, or ollama
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-3-5-sonnet-20241022"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama2"

    # Hungarian NLP
    HUBERT_MODEL: str = "SZTAKI-HLT/hubert-base-cc"
    EMBEDDING_DIMENSION: int = 768
    HUSPACY_MODEL: str = "hu_core_news_lg"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    SENTRY_DSN: Optional[str] = None

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
