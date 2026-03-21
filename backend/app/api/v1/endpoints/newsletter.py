"""
Newsletter subscription endpoints.

Handles subscribe, confirm, and unsubscribe for the landing page.
No authentication required - public endpoints.
"""

import secrets
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.postgres.models import NewsletterSubscriber
from app.db.postgres.session import get_db
from app.services.email_service import get_email_service

logger = get_logger(__name__)

router = APIRouter()


# =============================================================================
# Schemas
# =============================================================================


class SubscribeRequest(BaseModel):
    """Newsletter subscribe request."""

    email: EmailStr
    language: str = Field(default="hu", pattern="^(hu|en)$")
    source: str = Field(default="landing_page", max_length=50)


class SubscribeResponse(BaseModel):
    """Newsletter subscribe response."""

    success: bool
    message: str


class UnsubscribeResponse(BaseModel):
    """Newsletter unsubscribe response."""

    success: bool
    message: str


# =============================================================================
# Endpoints
# =============================================================================


@router.post(
    "/subscribe",
    response_model=SubscribeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Subscribe to newsletter",
    description="Public endpoint - no auth required. Subscribes email to the AutoCognitix newsletter.",
)
async def subscribe(
    payload: SubscribeRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> SubscribeResponse:
    """Subscribe to newsletter."""
    email = payload.email.lower().strip()

    # Check for existing subscriber
    result = await db.execute(
        select(NewsletterSubscriber).where(NewsletterSubscriber.email == email)
    )
    existing = result.scalar_one_or_none()

    if existing:
        if existing.status == "confirmed":
            return SubscribeResponse(
                success=True,
                message="Már feliratkoztál!" if payload.language == "hu" else "Already subscribed!",
            )
        if existing.status == "unsubscribed":
            # Re-subscribe
            existing.status = "pending"
            existing.confirm_token = secrets.token_urlsafe(32)
            existing.unsubscribed_at = None
            await db.flush()

            await _send_confirm_email(email, existing.confirm_token, payload.language)

            return SubscribeResponse(
                success=True,
                message=(
                    "Újra feliratkoztál! Erősítsd meg az email címed."
                    if payload.language == "hu"
                    else "Re-subscribed! Please confirm your email."
                ),
            )
        # Status is pending - resend confirmation
        if existing.confirm_token:
            await _send_confirm_email(email, existing.confirm_token, payload.language)
        return SubscribeResponse(
            success=True,
            message=(
                "Már feliratkoztál - nézd meg az email fiókodat a megerősítő linkért!"
                if payload.language == "hu"
                else "Already signed up - check your inbox for the confirmation link!"
            ),
        )

    # New subscriber
    confirm_token = secrets.token_urlsafe(32)
    unsubscribe_token = secrets.token_urlsafe(32)

    # Get client IP (Railway uses X-Forwarded-For)
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else None)
    if ip and "," in ip:
        ip = ip.split(",")[0].strip()

    subscriber = NewsletterSubscriber(
        email=email,
        status="pending",
        confirm_token=confirm_token,
        unsubscribe_token=unsubscribe_token,
        source=payload.source,
        language=payload.language,
        ip_address=ip,
    )
    db.add(subscriber)
    await db.flush()

    logger.info(f"New newsletter subscriber: {email} (source: {payload.source})")

    # Send confirmation email
    await _send_confirm_email(email, confirm_token, payload.language)

    return SubscribeResponse(
        success=True,
        message=(
            "Sikeres feliratkozás! Nézd meg az email fiókodat a megerősítő linkért."
            if payload.language == "hu"
            else "Subscribed! Check your inbox to confirm your email."
        ),
    )


@router.get(
    "/confirm/{token}",
    response_model=SubscribeResponse,
    summary="Confirm newsletter subscription",
    description="Confirms email address via token sent in confirmation email.",
)
async def confirm(
    token: str,
    db: AsyncSession = Depends(get_db),
) -> SubscribeResponse:
    """Confirm newsletter subscription via email token."""
    result = await db.execute(
        select(NewsletterSubscriber).where(NewsletterSubscriber.confirm_token == token)
    )
    subscriber = result.scalar_one_or_none()

    if not subscriber:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Érvénytelen vagy lejárt megerősítő link.",
        )

    if subscriber.status == "confirmed":
        return SubscribeResponse(success=True, message="Már megerősítve!")

    subscriber.status = "confirmed"
    subscriber.confirmed_at = datetime.now(timezone.utc)
    subscriber.confirm_token = None  # Invalidate token
    await db.flush()

    logger.info(f"Newsletter subscription confirmed: {subscriber.email}")

    return SubscribeResponse(
        success=True,
        message="Email cím megerősítve! Köszönjük a feliratkozást.",
    )


@router.get(
    "/unsubscribe/{token}",
    response_model=UnsubscribeResponse,
    summary="Unsubscribe from newsletter",
    description="Unsubscribe via unique token. GDPR compliant.",
)
async def unsubscribe(
    token: str,
    db: AsyncSession = Depends(get_db),
) -> UnsubscribeResponse:
    """Unsubscribe from newsletter."""
    result = await db.execute(
        select(NewsletterSubscriber).where(NewsletterSubscriber.unsubscribe_token == token)
    )
    subscriber = result.scalar_one_or_none()

    if not subscriber:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Érvénytelen leiratkozási link.",
        )

    if subscriber.status == "unsubscribed":
        return UnsubscribeResponse(success=True, message="Már leiratkoztál.")

    subscriber.status = "unsubscribed"
    subscriber.unsubscribed_at = datetime.now(timezone.utc)
    await db.flush()

    logger.info(f"Newsletter unsubscribe: {subscriber.email}")

    return UnsubscribeResponse(
        success=True,
        message="Sikeresen leiratkoztál. Sajnáljuk, hogy elmész!",
    )


# =============================================================================
# Helpers
# =============================================================================


async def _send_confirm_email(email: str, token: str, language: str = "hu") -> None:
    """Send confirmation email with double opt-in link."""
    service = get_email_service()

    # Build confirm URL (uses the backend API which will redirect or show success)
    from app.core.config import settings

    base_url = getattr(settings, "LANDING_PAGE_URL", "https://autocognitix-landing-production.up.railway.app")
    confirm_url = f"{base_url}/{language}/confirm.html?token={token}"

    if language == "hu":
        subject = "AutoCognitix - Erősítsd meg a feliratkozásod"
        text_content = (
            f"Kedves Feliratkozó!\n\n"
            f"Köszönjük, hogy feliratkoztál az AutoCognitix hírlevelére.\n\n"
            f"A feliratkozás megerősítéséhez kattints ide:\n{confirm_url}\n\n"
            f"Ha nem te iratkoztál fel, hagyd figyelmen kívül ezt az emailt.\n\n"
            f"Üdvözlettel,\nAz AutoCognitix Csapat"
        )
    else:
        subject = "AutoCognitix - Confirm your subscription"
        text_content = (
            f"Hello!\n\n"
            f"Thank you for subscribing to the AutoCognitix newsletter.\n\n"
            f"Please confirm your subscription by clicking here:\n{confirm_url}\n\n"
            f"If you didn't subscribe, please ignore this email.\n\n"
            f"Best regards,\nThe AutoCognitix Team"
        )

    await service._send_email(
        to_email=email,
        subject=subject,
        text_content=text_content,
    )
