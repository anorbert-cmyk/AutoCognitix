"""
Application configuration using Pydantic Settings.
Loads configuration from environment variables and .env file.
Supports Railway deployment with automatic environment variable detection.
"""

from functools import lru_cache
from typing import List, Literal, Optional, Union

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The one string that means "we do not know which build this is".
UNKNOWN_BUILD = "unknown"

# Values that are *present* but carry no information. Every one of these is
# something a real pipeline produces: an unexpanded `${GITHUB_SHA}` in a CI
# expression arrives as the empty string, a Docker ARG nobody passed arrives as
# whatever the Dockerfile defaulted it to, and a Python-formatted `None` arrives
# as the literal "None". They must all read as "unknown", never be echoed back
# as if they were a commit - a confident wrong answer here is exactly the thing
# that cost 2.5 hours of bundle-hash fingerprinting.
_UNINFORMATIVE_VALUES = frozenset({"", UNKNOWN_BUILD, "none", "null"})


def _known(value: Optional[str]) -> Optional[str]:
    """Return `value` stripped, or None if it carries no information.

    Kept as a module-level pure function rather than a Settings method so the
    "what counts as unknown" rule has exactly one definition and can be tested
    without constructing a whole Settings object.
    """
    cleaned = (value or "").strip()
    return cleaned if cleaned.lower() not in _UNINFORMATIVE_VALUES else None


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

    # --- Build identity --------------------------------------------------
    # Answers "which commit is actually serving?" without a login. Railway's
    # healthcheck-gated cutover means a FAILED deploy leaves the OLD container
    # running with a green healthcheck, so "the site responds" is not evidence
    # that the deploy landed - only the commit is.
    #
    # Railway injects the RAILWAY_* trio into the runtime environment of each
    # deployment, so on the normal push-to-deploy path this needs zero build
    # plumbing. They are read as ordinary settings fields (not os.environ at
    # request time) because a build identity cannot change while the process
    # lives; freezing it at construction is both correct and this file's idiom.
    RAILWAY_GIT_COMMIT_SHA: Optional[str] = None
    RAILWAY_GIT_BRANCH: Optional[str] = None
    RAILWAY_DEPLOYMENT_ID: Optional[str] = None
    # Fallback for images Railway did not build: the GHCR image from
    # .github/workflows/cd.yml already passes `--build-arg COMMIT_SHA`, and
    # Dockerfile.prod now declares it. Also covers local `docker build`.
    COMMIT_SHA: Optional[str] = None

    @property
    def build_commit_sha(self) -> str:
        """The git commit this process is running, or "unknown".

        Railway's runtime variable wins over the baked build arg: it is injected
        fresh by the platform at container start, whereas the ARG is frozen into
        an image layer that a cache hit could carry forward. When the platform
        tells us what it deployed, believe the platform.
        """
        return _known(self.RAILWAY_GIT_COMMIT_SHA) or _known(self.COMMIT_SHA) or UNKNOWN_BUILD

    @property
    def build_commit_source(self) -> str:
        """Where build_commit_sha came from: "railway", "build-arg" or "unknown".

        Exposed because the two sources have different failure modes, and an
        operator staring at an unexpected SHA needs to know which one they are
        arguing with before they can tell a bad pin from a bad deploy.
        """
        if _known(self.RAILWAY_GIT_COMMIT_SHA):
            return "railway"
        if _known(self.COMMIT_SHA):
            return "build-arg"
        return UNKNOWN_BUILD

    @property
    def build_branch(self) -> str:
        """Git branch of the running build, or "unknown" (Railway-only)."""
        return _known(self.RAILWAY_GIT_BRANCH) or UNKNOWN_BUILD

    @property
    def build_deployment_id(self) -> str:
        """Railway deployment id, or "unknown".

        The handle that turns "this is the wrong commit" into "here is the
        deploy log for the container that is actually up".
        """
        return _known(self.RAILWAY_DEPLOYMENT_ID) or UNKNOWN_BUILD

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
    # under us. This is the commit the ~54k indexed Qdrant vectors were produced
    # from (repo tip since 2024-10-24); the Dockerfile.prod ONNX export stage
    # uses the SAME value, so image and index can never drift apart.
    HUBERT_REVISION: str = "028baac7feb87a7b2f042bbdaa5deec6513c6060"
    EMBEDDING_DIMENSION: int = 768
    HUSPACY_MODEL: str = "hu_core_news_lg"

    # --- Embedding backend selection -------------------------------------
    # "auto"     -> ONNX Runtime if the exported graph is present, else torch.
    # "onnx"     -> ONNX Runtime only (production default via the image).
    # "torch"    -> torch/transformers only (local dev + offline indexer).
    # "disabled" -> no backend; embed calls raise EmbeddingUnavailableError.
    # No value ever yields a zero vector - a missing backend is an ERROR.
    EMBEDDING_BACKEND: str = "auto"
    # Paths are env-overridable so a wrong path is a Railway variable fix, not a
    # redeploy. Defaults match the Dockerfile.prod COPY target.
    HUBERT_ONNX_PATH: str = "/app/models/hubert_fp32.onnx"
    HUBERT_VOCAB_PATH: str = "/app/models/vocab.txt"
    # ONNX Runtime intra-op threads PER SESSION - and every gunicorn worker has
    # its own session. Explicit because ORT otherwise grabs every core. The
    # default is 1 because the total is a product, not a sum:
    #   WEB_CONCURRENCY (2) x embedding pool slots (2, embedding_service.py)
    #   x EMBEDDING_ORT_THREADS = 4 threads on a 2-vCPU Railway container.
    # Raise it as a Railway variable on a plan with more cores.
    EMBEDDING_ORT_THREADS: int = 1

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
