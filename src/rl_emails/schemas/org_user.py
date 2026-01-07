"""Organization user Pydantic schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class OrgUserCreate(BaseModel):
    """Schema for creating an organization user."""

    email: EmailStr
    name: str | None = None
    role: Literal["admin", "member"] = "member"


class OrgUserUpdate(BaseModel):
    """Schema for updating an organization user."""

    name: str | None = Field(None, min_length=1, max_length=255)
    role: Literal["admin", "member"] | None = None


class OrgUserResponse(BaseModel):
    """Schema for organization user response."""

    id: UUID
    org_id: UUID
    email: str
    name: str | None
    role: str
    gmail_connected: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
