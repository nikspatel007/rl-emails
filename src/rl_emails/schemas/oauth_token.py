"""Pydantic schemas for OAuth token operations."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class OAuthTokenCreate(BaseModel):
    """Schema for creating an OAuth token."""

    access_token: str
    refresh_token: str
    expires_at: datetime
    scopes: list[str] | None = None
    provider: str = "google"


class OAuthTokenUpdate(BaseModel):
    """Schema for updating an OAuth token."""

    access_token: str | None = None
    refresh_token: str | None = None
    expires_at: datetime | None = None
    scopes: list[str] | None = None


class OAuthTokenResponse(BaseModel):
    """Schema for OAuth token response."""

    id: UUID
    user_id: UUID
    provider: str
    expires_at: datetime
    scopes: list[str] | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class OAuthTokenStatus(BaseModel):
    """Schema for OAuth token status check."""

    connected: bool
    provider: str | None = None
    expires_at: datetime | None = None
    is_expired: bool = False
