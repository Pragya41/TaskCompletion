import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    APP_NAME: str = "Palm Mind AI - RAG & Ingestion Engine"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = ["*"]

    # Database (Default async SQLite for zero-friction local run; PostgreSQL in Docker)
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./palm_mind.db",
        description="Async SQLAlchemy database connection URL (sqlite+aiosqlite or postgresql+asyncpg)",
    )

    # Redis Chat Memory
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for multi-turn conversational memory",
    )
    REDIS_CHAT_TTL_SECONDS: int = Field(
        default=86400,  # 24 hours
        description="TTL for chat session history in Redis (seconds)",
    )
    ALLOW_IN_MEMORY_REDIS_FALLBACK: bool = Field(
        default=True,
        description="If true, seamlessly falls back to fakeredis when no standalone Redis instance is running",
    )

    # Vector Database: Qdrant
    QDRANT_URL: Optional[str] = Field(
        default=None,
        description="Remote Qdrant URL (e.g. http://localhost:6333). If None or empty, uses local embedded storage",
    )
    QDRANT_API_KEY: Optional[str] = Field(
        default=None,
        description="API Key for Qdrant Cloud (if applicable)",
    )
    QDRANT_COLLECTION_NAME: str = Field(
        default="palm_mind_documents",
        description="Collection name in Qdrant for document embeddings",
    )
    QDRANT_EMBEDDED_PATH: str = Field(
        default="./qdrant_storage",
        description="Local directory for embedded Qdrant persistence when QDRANT_URL is not set",
    )
    VECTOR_DIMENSION: int = Field(
        default=1536,
        description="Embedding dimension (1536 for text-embedding-3-small)",
    )

    # LLM & Embedding Settings
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="OpenAI API key (or compatible service API key)",
    )
    OPENAI_BASE_URL: Optional[str] = Field(
        default=None,
        description="Custom base URL for OpenAI-compatible APIs (e.g. Ollama, OpenRouter)",
    )
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name",
    )
    LLM_MODEL: str = Field(
        default="gpt-4o-mini",
        description="Chat LLM model name",
    )
    MOCK_LLM_IF_NO_KEY: bool = Field(
        default=True,
        description="If True and OPENAI_API_KEY is not provided, provides deterministic intelligent mock responses for seamless local evaluation",
    )

    # Document Ingestion Defaults
    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 50
    MAX_FILE_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".txt"]


settings = Settings()
