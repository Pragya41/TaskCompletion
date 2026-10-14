from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import BookingError
from app.core.logging import logger
from app.models.booking import Booking
from app.schemas.booking import BookingCreate


class BookingService:
    """Handles interview booking persistence and retrieval."""

    @staticmethod
    async def create_booking(
        db: AsyncSession,
        booking_in: BookingCreate,
    ) -> Booking:
        """Persists a new interview booking in the database."""
        try:
            booking = Booking(
                session_id=booking_in.session_id,
                name=booking_in.name.strip(),
                email=str(booking_in.email).strip().lower(),
                date=booking_in.date.strip(),
                time=booking_in.time.strip(),
                status="CONFIRMED",
                notes=booking_in.notes,
            )
            db.add(booking)
            await db.commit()
            await db.refresh(booking)
            logger.info(
                f"Successfully created interview booking: ID={booking.id}, Candidate={booking.name}, Date={booking.date} {booking.time}"
            )
            return booking
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to create booking: {e}", exc_info=True)
            raise BookingError(f"Failed to store interview booking: {str(e)}") from e

    @staticmethod
    async def get_booking_by_id(
        db: AsyncSession,
        booking_id: str,
    ) -> Optional[Booking]:
        """Fetches a specific booking by its primary key ID."""
        result = await db.execute(select(Booking).where(Booking.id == booking_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def list_bookings(
        db: AsyncSession,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Booking]:
        """Retrieves a paginated list of all interview bookings."""
        stmt = (
            select(Booking)
            .order_by(Booking.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())


# Global singleton instance
booking_service = BookingService()
