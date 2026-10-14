from fastapi import APIRouter

from app.api.v1.bookings import router as bookings_router
from app.api.v1.chat import router as chat_router
from app.api.v1.ingestion import router as ingestion_router

api_v1_router = APIRouter()

api_v1_router.include_router(ingestion_router)
api_v1_router.include_router(chat_router)
api_v1_router.include_router(bookings_router)
