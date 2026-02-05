"""
Authentication endpoints.

Provides user registration, login, token refresh, logout, profile management,
and password reset endpoints. All tokens are JWTs with configurable expiration times.
"""

import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.schemas.auth import (
    ForgotPasswordRequest,
    ForgotPasswordResponse,
    LogoutRequest,
    LogoutResponse,
    ResetPasswordRequest,
    ResetPasswordResponse,
    Token,
    TokenRefresh,
    UserCreate,
    UserPasswordUpdate,
    UserResponse,
    UserUpdate,
)
from app.core.security import (
    blacklist_token,
    create_access_token,
    create_password_reset_token,
    create_refresh_token,
    decode_token,
    get_password_hash,
    verify_password,
)
from app.db.postgres.models import User
from app.db.postgres.repositories import UserRepository
from app.db.postgres.session import get_db

router = APIRouter()
logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


# =============================================================================
# Dependencies
# =============================================================================


async def get_current_user_from_token(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Dependency to get the current authenticated user from JWT token.

    Args:
        token: JWT access token from Authorization header
        db: Database session

    Returns:
        User model instance

    Raises:
        401: Invalid or expired token
        401: User not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Érvénytelen vagy lejárt token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = decode_token(token)

    if not payload:
        logger.warning("Invalid token provided")
        raise credentials_exception

    if payload.get("type") != "access":
        logger.warning("Token is not an access token")
        raise credentials_exception

    user_id = payload.get("sub")
    if not user_id:
        logger.warning("Token has no subject")
        raise credentials_exception

    # Fetch user from database
    repository = UserRepository(db)
    try:
        user = await repository.get(UUID(user_id))
    except ValueError:
        # Invalid UUID format
        raise credentials_exception

    if not user:
        logger.warning(f"User {user_id} not found")
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Felhasználói fiók inaktív",
        )

    return user


async def get_optional_current_user(
    token: str | None = Depends(
        OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)
    ),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """
    Optional dependency to get current user (returns None if not authenticated).

    Useful for endpoints that work differently for authenticated vs anonymous users.
    """
    if not token:
        return None

    try:
        return await get_current_user_from_token(token, db)
    except HTTPException:
        return None


def require_role(*roles: str):
    """
    Dependency factory to require specific user roles.

    Args:
        roles: Allowed roles

    Returns:
        Dependency function that checks user role
    """

    async def role_checker(
        current_user: User = Depends(get_current_user_from_token),
    ) -> User:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Nincs megfelelő jogosultság",
            )
        return current_user

    return role_checker


# =============================================================================
# OpenAPI Response Examples
# =============================================================================

REGISTER_RESPONSES: dict[int, dict[str, Any]] = {
    201: {
        "description": "Felhasználó sikeresen regisztrálva",
        "content": {
            "application/json": {
                "example": {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "email": "user@example.com",
                    "full_name": "Kovács János",
                    "is_active": True,
                    "role": "user",
                }
            }
        },
    },
    400: {
        "description": "Email cím már regisztrálva van",
        "content": {
            "application/json": {"example": {"detail": "Ez az email cím már regisztrálva van"}}
        },
    },
    422: {
        "description": "Validációs hiba",
        "content": {
            "application/json": {
                "example": {
                    "detail": [
                        {
                            "loc": ["body", "password"],
                            "msg": "A jelszónak legalább 8 karakter hosszúnak kell lennie",
                            "type": "value_error",
                        }
                    ]
                }
            }
        },
    },
}

LOGIN_RESPONSES: dict[int, dict[str, Any]] = {
    200: {
        "description": "Sikeres bejelentkezés",
        "content": {
            "application/json": {
                "example": {
                    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                    "token_type": "bearer",
                }
            }
        },
    },
    401: {
        "description": "Hibás bejelentkezési adatok",
        "content": {
            "application/json": {"example": {"detail": "Hibás email cím vagy jelszó"}}
        },
    },
    423: {
        "description": "Fiók zárolva",
        "content": {
            "application/json": {
                "example": {
                    "detail": "Fiók zárolva túl sok sikertelen bejelentkezési kísérlet miatt"
                }
            }
        },
    },
}

REFRESH_RESPONSES: dict[int, dict[str, Any]] = {
    200: {
        "description": "Tokenek sikeresen frissítve",
        "content": {
            "application/json": {
                "example": {
                    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                    "token_type": "bearer",
                }
            }
        },
    },
    401: {
        "description": "Érvénytelen vagy lejárt refresh token",
        "content": {
            "application/json": {"example": {"detail": "Érvénytelen refresh token"}}
        },
    },
}

ME_RESPONSES: dict[int, dict[str, Any]] = {
    200: {
        "description": "Aktuális felhasználó adatai",
        "content": {
            "application/json": {
                "example": {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "email": "user@example.com",
                    "full_name": "Kovács János",
                    "is_active": True,
                    "role": "user",
                }
            }
        },
    },
    401: {
        "description": "Érvénytelen vagy lejárt access token",
        "content": {
            "application/json": {"example": {"detail": "Érvénytelen access token"}}
        },
    },
}


# =============================================================================
# Endpoints
# =============================================================================


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    responses=REGISTER_RESPONSES,
    summary="Új felhasználó regisztrálása",
    description="""
Új felhasználói fiók regisztrálása.

**Kérés törzs:**
- `email`: Érvényes email cím (kötelező)
- `password`: Jelszó 8-100 karakter, tartalmaznia kell nagybetűt, kisbetűt, számot és speciális karaktert (kötelező)
- `full_name`: Teljes név (opcionális)

**Visszatérési érték:** A létrehozott felhasználó adatai jelszó nélkül.
    """,
)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """
    Register a new user.

    Creates a new user account with the provided email and password.
    Password is securely hashed before storage using bcrypt.

    Args:
        user_data: User registration data (email, password, optional full name)
        db: Database session

    Returns:
        Created user information (without password)

    Raises:
        400: Email already registered
    """
    repository = UserRepository(db)

    # Check if email already exists
    existing_user = await repository.get_by_email(user_data.email.lower())
    if existing_user:
        logger.warning(f"Registration attempt with existing email: {user_data.email}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ez az email cím már regisztrálva van",
        )

    # Hash the password
    hashed_password = get_password_hash(user_data.password)

    # Prevent self-registration as admin
    role = user_data.role if user_data.role != "admin" else "user"

    # Create the user
    user = await repository.create(
        {
            "email": user_data.email.lower(),
            "hashed_password": hashed_password,
            "full_name": user_data.full_name,
            "is_active": True,
            "role": role,
        }
    )

    await db.commit()

    logger.info(f"User registered: {user.email}")

    return UserResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        role=user.role,
        created_at=user.created_at.isoformat() if user.created_at else None,
    )


@router.post(
    "/login",
    response_model=Token,
    responses=LOGIN_RESPONSES,
    summary="Felhasználó bejelentkezése",
    description="""
Felhasználó hitelesítése és JWT tokenek kiadása.

**Kérés törzs:** `application/x-www-form-urlencoded`
- `username`: Felhasználó email címe
- `password`: Felhasználó jelszava

**Visszatérési érték:**
- `access_token`: JWT token API hozzáféréshez (30 percig érvényes)
- `refresh_token`: JWT token az access token frissítéséhez (7 napig érvényes)
- `token_type`: Mindig "bearer"

**Használat:**
Az access tokent a következő kérésekbe kell beilleszteni:
```
Authorization: Bearer <access_token>
```
    """,
)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
) -> Token:
    """
    Authenticate user and return tokens.

    Validates email and password, then returns JWT access and refresh tokens.
    Uses OAuth2 password flow for compatibility with standard clients.

    Args:
        form_data: OAuth2 password request form (username=email, password)
        db: Database session

    Returns:
        Access and refresh tokens

    Raises:
        401: Invalid email or password
        423: Account is locked
    """
    repository = UserRepository(db)

    # Find user by email (form_data.username is the email in OAuth2 flow)
    user = await repository.get_by_email(form_data.username.lower())

    if not user:
        logger.warning(f"Login attempt with unknown email: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Hibás email cím vagy jelszó",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if account is locked
    if await repository.is_account_locked(user):
        logger.warning(f"Login attempt on locked account: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail="Fiók zárolva túl sok sikertelen bejelentkezési kísérlet miatt. Próbálja újra később.",
        )

    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Felhasználói fiók inaktív",
        )

    # Verify password
    if not verify_password(form_data.password, user.hashed_password):
        logger.warning(f"Login attempt with wrong password: {form_data.username}")
        # Record failed attempt
        is_locked = await repository.record_failed_login(user)
        await db.commit()

        if is_locked:
            logger.warning(f"Account locked due to failed attempts: {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Fiók zárolva túl sok sikertelen bejelentkezési kísérlet miatt. Próbálja újra később.",
            )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Hibás email cím vagy jelszó",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Record successful login
    await repository.record_successful_login(user)
    await db.commit()

    # Create tokens with role claim
    access_token = create_access_token(
        subject=str(user.id),
        additional_claims={"role": user.role},
    )
    refresh_token = create_refresh_token(subject=str(user.id))

    logger.info(f"User logged in: {user.email}")

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
    )


@router.post(
    "/refresh",
    response_model=Token,
    responses=REFRESH_RESPONSES,
    summary="Tokenek frissítése",
    description="""
Access token frissítése érvényes refresh tokennel.

Használja ezt a végpontot, amikor az access token lejár, hogy újakat kapjon
anélkül, hogy a felhasználónak újra be kellene jelentkeznie.

**Kérés törzs:**
- `refresh_token`: Érvényes refresh token a bejelentkezésből

**Visszatérési érték:** Új access és refresh tokenek.

**Megjegyzés:** A frissítés után mindkét régi token érvénytelenné válik.
    """,
)
async def refresh_tokens(
    token_data: TokenRefresh,
    db: AsyncSession = Depends(get_db),
) -> Token:
    """
    Refresh access token using refresh token.

    Validates the refresh token and issues new access and refresh tokens.
    The old refresh token should be discarded after this call.

    Args:
        token_data: Refresh token data
        db: Database session

    Returns:
        New access and refresh tokens

    Raises:
        401: Invalid refresh token
    """
    payload = decode_token(token_data.refresh_token)

    if not payload or payload.get("type") != "refresh":
        logger.warning("Invalid refresh token provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Érvénytelen refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")

    # Verify user still exists and is active
    repository = UserRepository(db)
    try:
        user = await repository.get(UUID(user_id))
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Érvénytelen refresh token",
        )

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Felhasználó nem található vagy inaktív",
        )

    # Blacklist old refresh token
    blacklist_token(token_data.refresh_token)

    # Create new tokens
    access_token = create_access_token(
        subject=user_id,
        additional_claims={"role": user.role},
    )
    new_refresh_token = create_refresh_token(subject=user_id)

    logger.info(f"Tokens refreshed for user: {user.email}")

    return Token(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
    )


@router.post(
    "/logout",
    response_model=LogoutResponse,
    summary="Kijelentkezés",
    description="""
Felhasználó kijelentkeztetése és tokenek érvénytelenítése.

**Kérés törzs:**
- `refresh_token`: Opcionális refresh token az érvénytelenítéshez

**Visszatérési érték:** Sikeres kijelentkezés üzenet.
    """,
)
async def logout(
    logout_data: LogoutRequest | None = None,
    token: str = Depends(oauth2_scheme),
) -> LogoutResponse:
    """
    Logout user and invalidate tokens.

    Args:
        logout_data: Optional logout request with refresh token
        token: Access token from header

    Returns:
        Logout confirmation
    """
    # Blacklist access token
    blacklist_token(token)

    # Blacklist refresh token if provided
    if logout_data and logout_data.refresh_token:
        blacklist_token(logout_data.refresh_token)

    logger.info("User logged out")

    return LogoutResponse(message="Sikeres kijelentkezés")


@router.get(
    "/me",
    response_model=UserResponse,
    responses=ME_RESPONSES,
    summary="Aktuális felhasználó lekérése",
    description="""
Az aktuálisan bejelentkezett felhasználó adatainak lekérése.

**Visszatérési érték:** A felhasználó adatai.
    """,
)
async def get_me(
    current_user: User = Depends(get_current_user_from_token),
) -> UserResponse:
    """
    Get current authenticated user.

    Returns the profile information for the currently authenticated user.

    Args:
        current_user: Current user from JWT token

    Returns:
        Current user information
    """
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        role=current_user.role,
        created_at=current_user.created_at.isoformat() if current_user.created_at else None,
    )


@router.put(
    "/me",
    response_model=UserResponse,
    summary="Profil frissítése",
    description="""
Az aktuális felhasználó profiljának frissítése.

**Kérés törzs:**
- `full_name`: Új teljes név (opcionális)
- `email`: Új email cím (opcionális)

**Visszatérési érték:** A frissített felhasználó adatai.
    """,
)
async def update_me(
    update_data: UserUpdate,
    current_user: User = Depends(get_current_user_from_token),
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """
    Update current user profile.

    Args:
        update_data: Profile update data
        current_user: Current user from token
        db: Database session

    Returns:
        Updated user information
    """
    repository = UserRepository(db)

    # Check if new email is already taken
    if update_data.email and update_data.email.lower() != current_user.email:
        existing = await repository.get_by_email(update_data.email.lower())
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Ez az email cím már használatban van",
            )

    # Build update dict
    update_dict: dict[str, Any] = {}
    if update_data.full_name is not None:
        update_dict["full_name"] = update_data.full_name
    if update_data.email is not None:
        update_dict["email"] = update_data.email.lower()

    if update_dict:
        user = await repository.update(current_user.id, update_dict)
        await db.commit()
        if user:
            logger.info(f"User profile updated: {user.email}")
            return UserResponse(
                id=str(user.id),
                email=user.email,
                full_name=user.full_name,
                is_active=user.is_active,
                role=user.role,
                created_at=user.created_at.isoformat() if user.created_at else None,
            )

    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        role=current_user.role,
        created_at=current_user.created_at.isoformat() if current_user.created_at else None,
    )


@router.put(
    "/me/password",
    response_model=ResetPasswordResponse,
    summary="Jelszó megváltoztatása",
    description="""
Az aktuális felhasználó jelszavának megváltoztatása.

**Kérés törzs:**
- `current_password`: Jelenlegi jelszó
- `new_password`: Új jelszó (ugyanazoknak a követelményeknek kell megfelelnie, mint a regisztrációnál)

**Visszatérési érték:** Sikeres jelszóváltoztatás üzenet.
    """,
)
async def change_password(
    password_data: UserPasswordUpdate,
    current_user: User = Depends(get_current_user_from_token),
    db: AsyncSession = Depends(get_db),
) -> ResetPasswordResponse:
    """
    Change current user's password.

    Args:
        password_data: Password update data
        current_user: Current user from token
        db: Database session

    Returns:
        Success message
    """
    repository = UserRepository(db)

    # Verify current password
    if not verify_password(password_data.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Hibás jelenlegi jelszó",
        )

    # Update password
    hashed_password = get_password_hash(password_data.new_password)
    await repository.update_password(current_user, hashed_password)
    await db.commit()

    logger.info(f"User changed password: {current_user.email}")

    return ResetPasswordResponse(message="A jelszó sikeresen megváltozott")


@router.post(
    "/forgot-password",
    response_model=ForgotPasswordResponse,
    summary="Elfelejtett jelszó",
    description="""
Jelszó visszaállítási token igénylése email címhez.

**Kérés törzs:**
- `email`: Regisztrált email cím

**Visszatérési érték:** Általános üzenet (biztonsági okokból nem jelzi, hogy létezik-e az email).

**Megjegyzés:** Éles környezetben itt email küldés történne a visszaállítási linkkel.
    """,
)
async def forgot_password(
    request_data: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_db),
) -> ForgotPasswordResponse:
    """
    Request password reset token.

    Args:
        request_data: Email address for password reset
        db: Database session

    Returns:
        Generic message (does not reveal if email exists)
    """
    repository = UserRepository(db)
    user = await repository.get_by_email(request_data.email.lower())

    if user and user.is_active:
        # Generate password reset token
        reset_token = create_password_reset_token(user.email)
        await repository.set_password_reset_token(user, reset_token)
        await db.commit()

        # TODO: Send email with reset link
        # In production, this would send an email like:
        # reset_link = f"{settings.FRONTEND_URL}/reset-password?token={reset_token}"
        # await send_password_reset_email(user.email, reset_link)

        logger.info(f"Password reset requested for: {user.email}")

    # Always return same response for security (prevents email enumeration)
    return ForgotPasswordResponse()


@router.post(
    "/reset-password",
    response_model=ResetPasswordResponse,
    summary="Jelszó visszaállítása",
    description="""
Új jelszó beállítása a visszaállítási tokennel.

**Kérés törzs:**
- `token`: Jelszó visszaállítási token (emailben kapott)
- `new_password`: Új jelszó (ugyanazoknak a követelményeknek kell megfelelnie, mint a regisztrációnál)

**Visszatérési érték:** Sikeres jelszóváltoztatás üzenet.
    """,
)
async def reset_password(
    request_data: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db),
) -> ResetPasswordResponse:
    """
    Reset password using reset token.

    Args:
        request_data: Reset token and new password
        db: Database session

    Returns:
        Success message
    """
    # Verify token
    payload = decode_token(request_data.token)

    if not payload or payload.get("type") != "password_reset":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Érvénytelen vagy lejárt visszaállítási token",
        )

    email = payload.get("sub")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Érvénytelen token",
        )

    repository = UserRepository(db)
    user = await repository.get_by_email(email.lower())

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Felhasználó nem található",
        )

    # Update password
    hashed_password = get_password_hash(request_data.new_password)
    await repository.update_password(user, hashed_password)
    await db.commit()

    # Blacklist the used reset token
    blacklist_token(request_data.token)

    logger.info(f"Password reset completed for: {user.email}")

    return ResetPasswordResponse(message="A jelszó sikeresen megváltozott")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "get_current_user_from_token",
    "get_optional_current_user",
    "require_role",
]
