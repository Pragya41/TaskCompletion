from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.core.logging import logger
from app.db.base import Base

# Create async SQLAlchemy engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
    # SQLite requires check_same_thread=False for async connection sharing
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
)

# Async session factory
async_session_maker = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency yielding an async database session."""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """Creates database tables if they do not exist."""
    logger.info("Initializing database schema...")
    # Import models here to ensure they are registered with Base metadata
    import app.models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema initialized successfully.")
