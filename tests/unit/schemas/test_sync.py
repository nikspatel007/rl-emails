"""Tests for sync state schemas."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rl_emails.schemas.sync import SyncStateResponse, SyncStateUpdate


class TestSyncStateUpdate:
    """Tests for SyncStateUpdate schema."""

    def test_sync_update_empty(self) -> None:
        """Test creating an empty update."""
        data = SyncStateUpdate()
        assert data.sync_status is None
        assert data.last_history_id is None
        assert data.error_message is None
        assert data.emails_synced is None

    def test_sync_update_status(self) -> None:
        """Test updating status."""
        data = SyncStateUpdate(sync_status="syncing")
        assert data.sync_status == "syncing"

    def test_sync_update_history_id(self) -> None:
        """Test updating history ID."""
        data = SyncStateUpdate(last_history_id="12345")
        assert data.last_history_id == "12345"

    def test_sync_update_error(self) -> None:
        """Test updating error message."""
        data = SyncStateUpdate(sync_status="error", error_message="Connection failed")
        assert data.sync_status == "error"
        assert data.error_message == "Connection failed"

    def test_sync_update_emails_synced(self) -> None:
        """Test updating emails synced count."""
        data = SyncStateUpdate(emails_synced=100)
        assert data.emails_synced == 100

    def test_sync_update_validates_status(self) -> None:
        """Test that status is validated."""
        with pytest.raises(ValidationError):
            SyncStateUpdate(sync_status="invalid")  # type: ignore[arg-type]

    def test_sync_update_valid_statuses(self) -> None:
        """Test all valid status values."""
        assert SyncStateUpdate(sync_status="idle").sync_status == "idle"
        assert SyncStateUpdate(sync_status="syncing").sync_status == "syncing"
        assert SyncStateUpdate(sync_status="error").sync_status == "error"


class TestSyncStateResponse:
    """Tests for SyncStateResponse schema."""

    def test_sync_response_from_dict(self) -> None:
        """Test creating response from dict."""
        state_id = uuid.uuid4()
        user_id = uuid.uuid4()
        now = datetime.now(UTC)
        data = SyncStateResponse(
            id=state_id,
            user_id=user_id,
            last_history_id="12345",
            last_sync_at=now,
            sync_status="idle",
            error_message=None,
            emails_synced=500,
            created_at=now,
            updated_at=now,
        )
        assert data.id == state_id
        assert data.user_id == user_id
        assert data.last_history_id == "12345"
        assert data.sync_status == "idle"
        assert data.emails_synced == 500

    def test_sync_response_with_nullable_fields(self) -> None:
        """Test response with nullable fields."""
        state_id = uuid.uuid4()
        user_id = uuid.uuid4()
        now = datetime.now(UTC)
        data = SyncStateResponse(
            id=state_id,
            user_id=user_id,
            last_history_id=None,
            last_sync_at=None,
            sync_status="idle",
            error_message=None,
            emails_synced=0,
            created_at=now,
            updated_at=now,
        )
        assert data.last_history_id is None
        assert data.last_sync_at is None
        assert data.error_message is None

    def test_sync_response_from_attributes(self) -> None:
        """Test that from_attributes is enabled."""
        assert SyncStateResponse.model_config.get("from_attributes") is True
