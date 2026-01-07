"""Organization Pydantic schemas."""

from __future__ import annotations

import re
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OrganizationCreate(BaseModel):
    """Schema for creating an organization."""

    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=63)
    settings: dict[str, object] = Field(default_factory=dict)

    @field_validator("slug")
    @classmethod
    def validate_slug(cls, v: str) -> str:
        """Validate slug format."""
        if not re.match(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$", v):
            msg = (
                "Slug must be lowercase alphanumeric with hyphens, not starting/ending with hyphen"
            )
            raise ValueError(msg)
        return v


class OrganizationUpdate(BaseModel):
    """Schema for updating an organization."""

    name: str | None = Field(None, min_length=1, max_length=255)
    settings: dict[str, object] | None = None


class OrganizationResponse(BaseModel):
    """Schema for organization response."""

    id: UUID
    name: str
    slug: str
    settings: dict[str, object]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
