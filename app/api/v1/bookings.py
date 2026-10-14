from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.booking import BookingCreate, BookingResponse
from app.services.booking.booking_service import booking_service

router = APIRouter(prefix="/bookings", tags=["Interview Bookings"])


@router.get(
    "",
    response_model=List[BookingResponse],
    summary="List all recorded interview bookings",
)
async def list_bookings(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
) -> List[BookingResponse]:
    records = await booking_service.list_bookings(db=db, limit=limit, offset=offset)
    return [BookingResponse.model_validate(b) for b in records]


@router.get(
    "/{booking_id}",
    response_model=BookingResponse,
    summary="Get details of a specific interview booking",
)
async def get_booking(
    booking_id: str,
    db: AsyncSession = Depends(get_db),
) -> BookingResponse:
    booking = await booking_service.get_booking_by_id(db=db, booking_id=booking_id)
    if not booking:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Interview booking with ID '{booking_id}' not found.",
        )
    return BookingResponse.model_validate(booking)


@router.post(
    "",
    response_model=BookingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Manually create an interview booking",
)
async def create_booking(
    booking_in: BookingCreate,
    db: AsyncSession = Depends(get_db),
) -> BookingResponse:
    record = await booking_service.create_booking(db=db, booking_in=booking_in)
    return BookingResponse.model_validate(record)
