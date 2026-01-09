"""Email provider connection endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from rl_emails.api.auth.dependencies import CurrentUser
from rl_emails.providers import (
    ConnectionService,
    ConnectionState,
    ProviderNotFoundError,
    ProviderType,
)

if TYPE_CHECKING:
    from rl_emails.api.auth.clerk import ClerkUser

router = APIRouter(prefix="/connections", tags=["connections"])
logger = structlog.get_logger(__name__)

# Dependency for connection service - will be overridden in app setup
_connection_service: ConnectionService | None = None


def get_connection_service() -> ConnectionService:
    """Get the connection service instance.

    Returns:
        ConnectionService instance.

    Raises:
        HTTPException: If service is not configured.
    """
    if _connection_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Connection service not configured",
        )
    return _connection_service


def set_connection_service(service: ConnectionService) -> None:
    """Set the connection service instance.

    Args:
        service: ConnectionService to use.
    """
    global _connection_service
    _connection_service = service


ConnectionServiceDep = Annotated[ConnectionService, Depends(get_connection_service)]


class ProviderStatusResponse(BaseModel):
    """Provider connection status response."""

    provider: str = Field(description="Provider type identifier")
    state: str = Field(description="Connection state")
    is_connected: bool = Field(description="Whether provider is connected")
    email: str | None = Field(default=None, description="Connected email address")
    connected_at: str | None = Field(default=None, description="When connection was established")
    last_sync: str | None = Field(default=None, description="When last sync occurred")
    error: str | None = Field(default=None, description="Error message if any")


class AllConnectionsResponse(BaseModel):
    """Response containing all provider statuses."""

    providers: list[ProviderStatusResponse] = Field(description="Status of all providers")
    connected_count: int = Field(description="Number of connected providers")


class AuthUrlResponse(BaseModel):
    """OAuth authorization URL response."""

    auth_url: str = Field(description="URL to redirect user for authorization")
    provider: str = Field(description="Provider being authorized")


class ConnectResponse(BaseModel):
    """Connection completion response."""

    provider: str = Field(description="Connected provider")
    state: str = Field(description="Connection state")
    message: str = Field(description="Status message")


class DisconnectResponse(BaseModel):
    """Disconnection response."""

    provider: str = Field(description="Disconnected provider")
    disconnected: bool = Field(description="Whether disconnection was successful")
    message: str = Field(description="Status message")


class SyncProgressResponse(BaseModel):
    """Sync progress response."""

    provider: str = Field(description="Provider being synced")
    in_progress: bool = Field(description="Whether sync is in progress")
    processed: int | None = Field(default=None, description="Messages processed")
    total: int | None = Field(default=None, description="Total messages to process")
    phase: str | None = Field(default=None, description="Current sync phase")


def _get_user_uuid(user: ClerkUser) -> UUID:
    """Convert Clerk user ID to UUID.

    Args:
        user: Clerk user.

    Returns:
        UUID derived from user ID.

    Note:
        Uses UUID5 with DNS namespace to create deterministic UUID from Clerk ID.
    """
    import uuid

    return uuid.uuid5(uuid.NAMESPACE_DNS, user.id)


@router.get(
    "/available",
    response_model=list[str],
    summary="List available providers",
    description="List all available (registered) email providers.",
)
async def list_available_providers(
    service: ConnectionServiceDep,
) -> list[str]:
    """List all available providers.

    Returns list of provider identifiers that can be connected.

    Args:
        service: Connection service.

    Returns:
        List of provider type strings.
    """
    providers = service.list_available_providers()
    return [p.value for p in providers]


@router.get(
    "",
    response_model=AllConnectionsResponse,
    summary="List all connections",
    description="Get connection status for all available email providers.",
)
async def list_connections(
    user: CurrentUser,
    service: ConnectionServiceDep,
) -> AllConnectionsResponse:
    """Get connection status for all providers.

    Returns status of all registered providers for the authenticated user.

    Args:
        user: Current authenticated user.
        service: Connection service.

    Returns:
        AllConnectionsResponse with status of each provider.
    """
    user_uuid = _get_user_uuid(user)

    statuses = await service.get_all_statuses(user_uuid)

    providers = []
    connected_count = 0

    for provider_type, connection_status in statuses.items():
        if connection_status.is_connected:
            connected_count += 1

        providers.append(
            ProviderStatusResponse(
                provider=provider_type.value,
                state=connection_status.state.value,
                is_connected=connection_status.is_connected,
                email=connection_status.email,
                connected_at=(
                    connection_status.connected_at.isoformat()
                    if connection_status.connected_at
                    else None
                ),
                last_sync=(
                    connection_status.last_sync.isoformat() if connection_status.last_sync else None
                ),
                error=connection_status.error,
            )
        )

    await logger.ainfo(
        "connections_listed",
        user_id=user.id,
        connected_count=connected_count,
    )

    return AllConnectionsResponse(
        providers=providers,
        connected_count=connected_count,
    )


@router.get(
    "/{provider}",
    response_model=ProviderStatusResponse,
    summary="Get provider status",
    description="Get connection status for a specific provider.",
)
async def get_connection_status(
    provider: str,
    user: CurrentUser,
    service: ConnectionServiceDep,
) -> ProviderStatusResponse:
    """Get connection status for a specific provider.

    Args:
        provider: Provider identifier (e.g., 'gmail').
        user: Current authenticated user.
        service: Connection service.

    Returns:
        ProviderStatusResponse with provider status.

    Raises:
        HTTPException: If provider is not found.
    """
    try:
        provider_type = ProviderType(provider)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown provider: {provider}",
        ) from None

    user_uuid = _get_user_uuid(user)

    try:
        connection_status = await service.get_status(user_uuid, provider_type)
    except ProviderNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider not available: {provider}",
        ) from None

    return ProviderStatusResponse(
        provider=provider_type.value,
        state=connection_status.state.value,
        is_connected=connection_status.is_connected,
        email=connection_status.email,
        connected_at=(
            connection_status.connected_at.isoformat() if connection_status.connected_at else None
        ),
        last_sync=(
            connection_status.last_sync.isoformat() if connection_status.last_sync else None
        ),
        error=connection_status.error,
    )


@router.post(
    "/{provider}/connect",
    response_model=AuthUrlResponse,
    summary="Start OAuth flow",
    description="Get authorization URL to connect a provider.",
)
async def start_connection(
    provider: str,
    user: CurrentUser,
    service: ConnectionServiceDep,
    state: Annotated[str | None, Query(description="CSRF state parameter")] = None,
) -> AuthUrlResponse:
    """Start OAuth authorization flow.

    Returns URL to redirect user to for provider authorization.

    Args:
        provider: Provider identifier (e.g., 'gmail').
        user: Current authenticated user.
        service: Connection service.
        state: Optional CSRF protection state.

    Returns:
        AuthUrlResponse with authorization URL.

    Raises:
        HTTPException: If provider is not found.
    """
    try:
        provider_type = ProviderType(provider)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown provider: {provider}",
        ) from None

    try:
        auth_url = await service.get_auth_url(provider_type, state=state)
    except ProviderNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider not available: {provider}",
        ) from None

    await logger.ainfo(
        "auth_flow_started",
        user_id=user.id,
        provider=provider,
    )

    return AuthUrlResponse(
        auth_url=auth_url,
        provider=provider,
    )


@router.post(
    "/{provider}/callback",
    response_model=ConnectResponse,
    summary="Complete OAuth flow",
    description="Complete OAuth authorization with the callback code.",
)
async def complete_connection(
    provider: str,
    code: Annotated[str, Query(description="Authorization code from OAuth callback")],
    user: CurrentUser,
    service: ConnectionServiceDep,
) -> ConnectResponse:
    """Complete OAuth authorization flow.

    Exchanges authorization code for tokens and establishes connection.

    Args:
        provider: Provider identifier (e.g., 'gmail').
        code: Authorization code from OAuth callback.
        user: Current authenticated user.
        service: Connection service.

    Returns:
        ConnectResponse with connection result.

    Raises:
        HTTPException: If authorization fails.
    """
    try:
        provider_type = ProviderType(provider)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown provider: {provider}",
        ) from None

    user_uuid = _get_user_uuid(user)

    try:
        connection_status = await service.complete_auth(user_uuid, provider_type, code)
    except ProviderNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider not available: {provider}",
        ) from None
    except Exception as e:
        await logger.aerror(
            "auth_completion_failed",
            user_id=user.id,
            provider=provider,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Authorization failed: {e}",
        ) from None

    await logger.ainfo(
        "connection_established",
        user_id=user.id,
        provider=provider,
        state=connection_status.state.value,
    )

    message = (
        "Successfully connected"
        if connection_status.state == ConnectionState.CONNECTED
        else f"Connection state: {connection_status.state.value}"
    )

    return ConnectResponse(
        provider=provider,
        state=connection_status.state.value,
        message=message,
    )


@router.delete(
    "/{provider}",
    response_model=DisconnectResponse,
    summary="Disconnect provider",
    description="Disconnect and revoke access for a provider.",
)
async def disconnect_provider(
    provider: str,
    user: CurrentUser,
    service: ConnectionServiceDep,
) -> DisconnectResponse:
    """Disconnect from a provider.

    Revokes OAuth tokens and removes connection.

    Args:
        provider: Provider identifier (e.g., 'gmail').
        user: Current authenticated user.
        service: Connection service.

    Returns:
        DisconnectResponse with disconnection result.

    Raises:
        HTTPException: If provider is not found.
    """
    try:
        provider_type = ProviderType(provider)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown provider: {provider}",
        ) from None

    user_uuid = _get_user_uuid(user)

    try:
        disconnected = await service.disconnect(user_uuid, provider_type)
    except ProviderNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider not available: {provider}",
        ) from None

    message = "Disconnected successfully" if disconnected else "Was not connected"

    await logger.ainfo(
        "provider_disconnected",
        user_id=user.id,
        provider=provider,
        disconnected=disconnected,
    )

    return DisconnectResponse(
        provider=provider,
        disconnected=disconnected,
        message=message,
    )


@router.get(
    "/{provider}/sync/progress",
    response_model=SyncProgressResponse,
    summary="Get sync progress",
    description="Get current sync progress for a provider.",
)
async def get_sync_progress(
    provider: str,
    user: CurrentUser,
    service: ConnectionServiceDep,
) -> SyncProgressResponse:
    """Get sync progress for a provider.

    Args:
        provider: Provider identifier (e.g., 'gmail').
        user: Current authenticated user.
        service: Connection service.

    Returns:
        SyncProgressResponse with sync progress.

    Raises:
        HTTPException: If provider is not found.
    """
    try:
        provider_type = ProviderType(provider)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown provider: {provider}",
        ) from None

    user_uuid = _get_user_uuid(user)

    try:
        progress = await service.get_sync_progress(user_uuid, provider_type)
    except ProviderNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider not available: {provider}",
        ) from None

    if progress is None:
        return SyncProgressResponse(
            provider=provider,
            in_progress=False,
        )

    return SyncProgressResponse(
        provider=provider,
        in_progress=True,
        processed=progress.processed,
        total=progress.total,
        phase=progress.current_phase,
    )
