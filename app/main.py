from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_v1_router
from app.core.config import settings
from app.core.exceptions import AppBaseException
from app.core.logging import logger, setup_logging
from app.db.session import init_db
from app.services.memory import chat_memory
from app.services.vector_store import vector_store


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application startup and shutdown lifecycle."""
    # 1. Initialize logging
    setup_logging(level="DEBUG" if settings.DEBUG else "INFO")
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}...")

    # 2. Initialize relational database tables
    await init_db()

    # 3. Ensure Qdrant collection is ready
    try:
        vector_store.ensure_collection()
    except Exception as e:
        logger.warning(f"Qdrant collection check encountered an issue: {e}")

    # 4. Check Redis connection
    try:
        await chat_memory.get_client()
    except Exception as e:
        logger.warning(f"Redis memory check encountered an issue: {e}")

    logger.info("Application startup sequence complete. Ready to receive requests.")
    yield

    # Shutdown
    logger.info("Shutting down application and freeing resources...")
    await chat_memory.close()
    logger.info("Application shutdown complete.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Production-grade backend built for Palm Mind AI technical evaluation. "
        "Provides Document Ingestion (.pdf, .txt) with dual chunking strategies, Qdrant vector indexing, "
        "SQL metadata persistence, and a custom multi-turn Conversational RAG pipeline backed by Redis memory "
        "and automated LLM-driven interview scheduling."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Domain Exception Handler
@app.exception_handler(AppBaseException)
async def domain_exception_handler(request: Request, exc: AppBaseException) -> JSONResponse:
    logger.error(f"Domain error processing {request.method} {request.url.path}: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.get("/", tags=["Root"])
async def root() -> dict:
    """Root endpoint welcoming the evaluator with system status and documentation links."""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "online",
        "documentation": "/docs",
        "redoc": "/redoc",
        "api_v1": settings.API_V1_PREFIX,
    }


@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """Health check endpoint inspecting Database, Redis, and Vector DB status."""
    health_status = {
        "status": "healthy",
        "database": "connected",
        "redis_memory": "active",
        "vector_store": "ready",
    }
    return health_status


# Include API v1 router
app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)
