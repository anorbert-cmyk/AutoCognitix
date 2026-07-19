"""
Application configuration using Pydantic Settings.
Loads configuration from environment variables and .env file.
Supports Railway deployment with automatic environment variable detection.
"""

from functools import lru_cache
from typing import List, Literal, Optional, Union

from pydantic import field_validator, model_validator
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

    # Cookie settings for httpOnly JWT storage
    COOKIE_DOMAIN: Optional[str] = None  # None = current domain only
    COOKIE_SECURE: bool = True  # Set to False for local HTTP development
    COOKIE_SAMESITE: Literal["lax", "strict", "none"] = "lax"  # CSRF protection

    @model_validator(mode="after")
    def validate_cookie_samesite_secure(self) -> "Settings":
        """
        Enforce the browser rule for cross-site cookies.

        Browsers reject a cookie with ``SameSite=None`` unless it is also
        marked ``Secure``. This combination is required in cross-site
        production (frontend and backend on different Railway domains), so
        guard against a misconfiguration that would silently drop auth cookies.

        Raises:
            ValueError: If COOKIE_SAMESITE is "none" but COOKIE_SECURE is False.
        """
        if self.COOKIE_SAMESITE == "none" and not self.COOKIE_SECURE:
            raise ValueError(
                "COOKIE_SAMESITE='none' requires COOKIE_SECURE=True. "
                "Browsers reject SameSite=None cookies without the Secure flag, "
                "which would drop auth cookies cross-site."
            )
        return self

    @field_validator("SECRET_KEY", "JWT_SECRET_KEY")
    @classmethod
    def validate_secrets(cls, v: str, info) -> str:
        """
        Validate that security secrets meet minimum requirements.

        Secrets must be:
        - Non-empty strings
        - At least 32 characters long for cryptographic security

        Raises:
            ValueError: If secret is invalid
        """
        if not v or len(v) < 32:
            raise ValueError(
                f"{info.field_name} must be at least 32 characters long. "
                f"Generate a secure secret with: openssl rand -hex 32"
            )
        return v

    # CORS - Use Union type to handle both string (from env) and list formats
    BACKEND_CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "https://autocognitix-landing-production.up.railway.app",
    ]

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
        if v and v.startswith("postgresql+asyncpg://"):
            return v
        if v and v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        if v and v.startswith("postgres://"):
            return v.replace("postgres://", "postgresql+asyncpg://", 1)
        return v

    # PostgreSQL Connection Pool Configuration
    DB_POOL_SIZE: int = 10  # Size of the connection pool
    DB_MAX_OVERFLOW: int = 10  # Maximum overflow connections
    DB_POOL_RECYCLE: int = 1800  # Recycle connections after 30 minutes
    DB_POOL_TIMEOUT: int = 30  # Timeout for acquiring a connection from the pool

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
    # Unified collection holding all huBERT vectors (DTC/complaint/recall) with a
    # type-discriminated payload. Env-overridable so a collection rename is a
    # Railway variable change, not a redeploy.
    QDRANT_UNIFIED_COLLECTION: str = "autocognitix"

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
    # Pin a specific revision so HuggingFace can't silently push a new model
    # under us. Override in prod with a verified commit hash; "main" means
    # "follow the branch tip" (acceptable for local dev, dangerous in prod).
    HUBERT_REVISION: str = "main"
    EMBEDDING_DIMENSION: int = 768
    HUSPACY_MODEL: str = "hu_core_news_lg"

    # Frontend URL (used for password reset links, etc.)
    FRONTEND_URL: str = "http://localhost:5173"

    # Email (n8n webhook or Resend API)
    N8N_WEBHOOK_URL: Optional[str] = None  # n8n base URL, e.g. https://your-n8n.app/webhook
    RESEND_API_KEY: Optional[str] = None
    EMAIL_FROM: str = "AutoCognitix <noreply@autocognitix.hu>"
    EMAIL_DEMO_MODE: bool = True  # True = csak logolás, nincs tényleges küldés

    # SMTP (optional, fallback email transport)
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None

    # Landing Page
    LANDING_PAGE_URL: str = "https://autocognitix-landing-production.up.railway.app"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    SENTRY_DSN: Optional[str] = None

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    TRUSTED_PROXY_COUNT: int = 1  # Number of trusted reverse proxies (Railway = 1)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
