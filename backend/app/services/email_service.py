"""
Email Service for AutoCognitix.

Email kuldes Resend API-val vagy demo modban logolassal.

Features:
- Jelszo visszaallitas email
- Udvozlo email regisztracio utan
- Demo mod (csak logolas, nincs tenyleges kuldes)
- Async mukodes

Author: AutoCognitix Team
"""

import asyncio
import logging
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Email Templates
# =============================================================================

PASSWORD_RESET_TEMPLATE_HU = """
Kedves {name}!

Jelszo visszaallitasi kerelmet kaptunk az AutoCognitix fiokjahoz.

A jelszo visszaallitasahoz kattintson az alabbi linkre:
{reset_link}

Ez a link 1 oran belul lejar.

Ha nem On kerte a jelszo visszaallitast, kerjuk hagyja figyelmen kivul ezt az emailt.

Udvozlettel,
Az AutoCognitix Csapat
"""

PASSWORD_RESET_TEMPLATE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #1a56db; color: white; padding: 20px; text-align: center; }}
        .content {{ padding: 20px; background-color: #f9fafb; }}
        .button {{ display: inline-block; background-color: #1a56db; color: white;
                   padding: 12px 24px; text-decoration: none; border-radius: 4px; margin: 20px 0; }}
        .footer {{ padding: 20px; text-align: center; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>AutoCognitix</h1>
        </div>
        <div class="content">
            <p>Kedves {name}!</p>
            <p>Jelszo visszaallitasi kerelmet kaptunk az AutoCognitix fiokjahoz.</p>
            <p>A jelszo visszaallitasahoz kattintson az alabbi gombra:</p>
            <p style="text-align: center;">
                <a href="{reset_link}" class="button">Jelszo visszaallitasa</a>
            </p>
            <p><small>Ez a link 1 oran belul lejar.</small></p>
            <p>Ha nem On kerte a jelszo visszaallitast, kerjuk hagyja figyelmen kivul ezt az emailt.</p>
        </div>
        <div class="footer">
            <p>Udvozlettel,<br>Az AutoCognitix Csapat</p>
            <p>&copy; 2024 AutoCognitix. Minden jog fenntartva.</p>
        </div>
    </div>
</body>
</html>
"""

WELCOME_TEMPLATE_HU = """
Kedves {name}!

Koszonjtuk az AutoCognitix platformon!

Sikeresen regisztralt a gepjarmu diagnosztikai rendszerunkbe.

Funkciok, amelyek elerhetoek az On szamara:
- DTC hibakod kereses es elemzes
- Magyar nyelvu diagnosztika
- Javitasi koltsegbecsles
- Visszahivas es panasz adatok

Kezdje el a hasznalat:
{login_link}

Ha barmilyen kerdese van, forduljon hozzank bizalommal.

Udvozlettel,
Az AutoCognitix Csapat
"""

WELCOME_TEMPLATE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #1a56db; color: white; padding: 20px; text-align: center; }}
        .content {{ padding: 20px; background-color: #f9fafb; }}
        .features {{ background-color: #fff; padding: 15px; border-radius: 4px; margin: 15px 0; }}
        .features ul {{ margin: 0; padding-left: 20px; }}
        .button {{ display: inline-block; background-color: #1a56db; color: white;
                   padding: 12px 24px; text-decoration: none; border-radius: 4px; margin: 20px 0; }}
        .footer {{ padding: 20px; text-align: center; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Udvozoljuk az AutoCognitix-en!</h1>
        </div>
        <div class="content">
            <p>Kedves {name}!</p>
            <p>Koszonjtuk az AutoCognitix platformon! Sikeresen regisztralt a gepjarmu diagnosztikai rendszerunkbe.</p>

            <div class="features">
                <p><strong>Funkciok, amelyek elerhetoek:</strong></p>
                <ul>
                    <li>DTC hibakod kereses es elemzes</li>
                    <li>Magyar nyelvu diagnosztika</li>
                    <li>Javitasi koltsegbecsles</li>
                    <li>Visszahivas es panasz adatok</li>
                </ul>
            </div>

            <p style="text-align: center;">
                <a href="{login_link}" class="button">Bejelentkezes</a>
            </p>
        </div>
        <div class="footer">
            <p>Udvozlettel,<br>Az AutoCognitix Csapat</p>
            <p>&copy; 2024 AutoCognitix. Minden jog fenntartva.</p>
        </div>
    </div>
</body>
</html>
"""


# =============================================================================
# Email Service
# =============================================================================


class EmailService:
    """
    Email kuldes szolgaltatas.

    Demo modban (EMAIL_DEMO_MODE=True) csak logol, nem kuld tenyleges emailt.
    Production modban Resend API-t hasznal.
    """

    _instance: Optional["EmailService"] = None
    _initialized: bool = False

    def __new__(cls) -> "EmailService":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize service."""
        if self._initialized:
            return

        self._initialized = True
        self._resend_client = None
        self._demo_mode = getattr(settings, "EMAIL_DEMO_MODE", True)
        self._from_email = getattr(settings, "EMAIL_FROM", "noreply@autocognitix.com")

        if self._demo_mode:
            logger.info("EmailService: DEMO mod - emailek csak logolasra kerulnek")
        else:
            self._init_resend()

    def _init_resend(self) -> None:
        """Initialize Resend client."""
        api_key = getattr(settings, "RESEND_API_KEY", None)
        if not api_key:
            logger.warning(
                "RESEND_API_KEY nincs beallitva, email kuldes nem lehetseges. Demo mod aktivalasa."
            )
            self._demo_mode = True
            return

        try:
            import resend

            resend.api_key = api_key
            self._resend_client = resend
            logger.info("EmailService: Resend kliens inicializalva")
        except ImportError:
            logger.warning("resend csomag nem telepitett, demo mod aktivalasa")
            self._demo_mode = True
        except Exception as e:
            logger.error(f"Resend inicializalasi hiba: {e}, demo mod aktivalasa")
            self._demo_mode = True

    async def send_password_reset(
        self,
        to_email: str,
        name: str,
        reset_link: str,
    ) -> bool:
        """
        Send password reset email.

        Args:
            to_email: Recipient email
            name: User name
            reset_link: Password reset link

        Returns:
            True if successful
        """
        subject = "AutoCognitix - Jelszo visszaallitas"

        text_content = PASSWORD_RESET_TEMPLATE_HU.format(
            name=name,
            reset_link=reset_link,
        )

        html_content = PASSWORD_RESET_TEMPLATE_HTML.format(
            name=name,
            reset_link=reset_link,
        )

        return await self._send_email(
            to_email=to_email,
            subject=subject,
            text_content=text_content,
            html_content=html_content,
        )

    async def send_welcome(
        self,
        to_email: str,
        name: str,
        login_link: str,
    ) -> bool:
        """
        Send welcome email after registration.

        Args:
            to_email: Recipient email
            name: User name
            login_link: Login link

        Returns:
            True if successful
        """
        subject = "Udvozoljuk az AutoCognitix-en!"

        text_content = WELCOME_TEMPLATE_HU.format(
            name=name,
            login_link=login_link,
        )

        html_content = WELCOME_TEMPLATE_HTML.format(
            name=name,
            login_link=login_link,
        )

        return await self._send_email(
            to_email=to_email,
            subject=subject,
            text_content=text_content,
            html_content=html_content,
        )

    async def _send_email(
        self,
        to_email: str,
        subject: str,
        text_content: str,
        html_content: Optional[str] = None,
    ) -> bool:
        """
        Internal email sending implementation.

        In demo mode, just logs instead of sending.

        Args:
            to_email: Recipient
            subject: Subject
            text_content: Plain text content
            html_content: HTML content (optional)

        Returns:
            True if successful
        """
        if self._demo_mode:
            logger.info(
                f"[DEMO] Email kuldese:\n"
                f"  Cimzett: {to_email}\n"
                f"  Targy: {subject}\n"
                f"  Tartalom (elso 200 karakter): {text_content[:200]}..."
            )
            return True

        if not self._resend_client:
            logger.error("Resend kliens nem elerheto")
            return False

        try:
            params = {
                "from": self._from_email,
                "to": [to_email],
                "subject": subject,
                "text": text_content,
            }

            if html_content:
                params["html"] = html_content

            # Resend is synchronous, run in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._resend_client.Emails.send(params),
            )

            logger.info(f"Email sikeresen elkuldve: {to_email}, id: {result.get('id', 'N/A')}")
            return True

        except Exception as e:
            logger.error(f"Email kuldesi hiba ({to_email}): {e}")
            return False

    @property
    def is_demo_mode(self) -> bool:
        """Check if running in demo mode."""
        return bool(self._demo_mode)


# =============================================================================
# Module-level Functions
# =============================================================================

_service_instance: Optional[EmailService] = None


def get_email_service() -> EmailService:
    """
    Get singleton service instance.

    Returns:
        EmailService instance
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = EmailService()
    return _service_instance


async def send_password_reset_email(
    to_email: str,
    name: str,
    reset_link: str,
) -> bool:
    """
    Convenience function for sending password reset email.

    Args:
        to_email: Recipient email
        name: User name
        reset_link: Reset link

    Returns:
        True if successful
    """
    service = get_email_service()
    return await service.send_password_reset(to_email, name, reset_link)


async def send_welcome_email(
    to_email: str,
    name: str,
    login_link: str,
) -> bool:
    """
    Convenience function for sending welcome email.

    Args:
        to_email: Recipient email
        name: User name
        login_link: Login link

    Returns:
        True if successful
    """
    service = get_email_service()
    return await service.send_welcome(to_email, name, login_link)
