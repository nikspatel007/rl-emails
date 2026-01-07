"""Tests for rl_emails.core.types."""

from __future__ import annotations

from typing import get_type_hints

from rl_emails.core.types import (
    ClusterAssignment,
    ClusterMetadataData,
    EmailData,
    EmailFeatures,
    LLMFlags,
    OAuthTokenData,
    OrganizationData,
    OrgUserData,
    PriorityScores,
    SyncStateData,
)


class TestEmailData:
    """Tests for EmailData TypedDict."""

    def test_email_data_has_expected_keys(self) -> None:
        """Test EmailData has all expected keys."""
        hints = get_type_hints(EmailData)
        expected_keys = {
            "message_id",
            "from_email",
            "from_name",
            "to_emails",
            "cc_emails",
            "bcc_emails",
            "subject",
            "date_str",
            "body_text",
            "body_html",
            "headers",
            "labels",
            "in_reply_to",
            "references",
        }
        assert set(hints.keys()) == expected_keys

    def test_email_data_can_be_created(self) -> None:
        """Test EmailData can be instantiated."""
        data: EmailData = {
            "message_id": "test@example.com",
            "from_email": "sender@example.com",
            "body_text": "Hello",
        }
        assert data["message_id"] == "test@example.com"


class TestEmailFeatures:
    """Tests for EmailFeatures TypedDict."""

    def test_email_features_has_expected_keys(self) -> None:
        """Test EmailFeatures has all expected keys."""
        hints = get_type_hints(EmailFeatures)
        expected_keys = {
            "email_id",
            "relationship_strength",
            "urgency_score",
            "is_service_email",
            "service_type",
            "service_importance",
        }
        assert set(hints.keys()) == expected_keys


class TestPriorityScores:
    """Tests for PriorityScores TypedDict."""

    def test_priority_scores_has_expected_keys(self) -> None:
        """Test PriorityScores has all expected keys."""
        hints = get_type_hints(PriorityScores)
        expected_keys = {
            "feature_score",
            "replied_similarity",
            "cluster_novelty",
            "sender_novelty",
            "priority_score",
        }
        assert set(hints.keys()) == expected_keys


class TestLLMFlags:
    """Tests for LLMFlags TypedDict."""

    def test_llm_flags_has_expected_keys(self) -> None:
        """Test LLMFlags has all expected keys."""
        hints = get_type_hints(LLMFlags)
        expected_keys = {"needs_llm", "reason"}
        assert set(hints.keys()) == expected_keys


class TestClusterAssignment:
    """Tests for ClusterAssignment TypedDict."""

    def test_cluster_assignment_has_expected_keys(self) -> None:
        """Test ClusterAssignment has all expected keys."""
        hints = get_type_hints(ClusterAssignment)
        expected_keys = {
            "email_id",
            "people_cluster_id",
            "content_cluster_id",
            "behavior_cluster_id",
            "service_cluster_id",
            "temporal_cluster_id",
        }
        assert set(hints.keys()) == expected_keys


class TestOrganizationData:
    """Tests for OrganizationData TypedDict."""

    def test_organization_data_has_expected_keys(self) -> None:
        """Test OrganizationData has all expected keys."""
        hints = get_type_hints(OrganizationData)
        expected_keys = {
            "id",
            "name",
            "slug",
            "settings",
            "created_at",
            "updated_at",
        }
        assert set(hints.keys()) == expected_keys

    def test_organization_data_can_be_created(self) -> None:
        """Test OrganizationData can be instantiated."""
        data: OrganizationData = {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "name": "Acme Corp",
            "slug": "acme",
        }
        assert data["name"] == "Acme Corp"
        assert data["slug"] == "acme"


class TestOrgUserData:
    """Tests for OrgUserData TypedDict."""

    def test_org_user_data_has_expected_keys(self) -> None:
        """Test OrgUserData has all expected keys."""
        hints = get_type_hints(OrgUserData)
        expected_keys = {
            "id",
            "org_id",
            "email",
            "name",
            "role",
            "gmail_connected",
            "created_at",
            "updated_at",
        }
        assert set(hints.keys()) == expected_keys

    def test_org_user_data_can_be_created(self) -> None:
        """Test OrgUserData can be instantiated."""
        data: OrgUserData = {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "org_id": "987e6543-e21b-12d3-a456-426614174000",
            "email": "user@example.com",
            "role": "admin",
        }
        assert data["email"] == "user@example.com"
        assert data["role"] == "admin"


class TestOAuthTokenData:
    """Tests for OAuthTokenData TypedDict."""

    def test_oauth_token_data_has_expected_keys(self) -> None:
        """Test OAuthTokenData has all expected keys."""
        hints = get_type_hints(OAuthTokenData)
        expected_keys = {
            "id",
            "user_id",
            "provider",
            "access_token",
            "refresh_token",
            "expires_at",
            "scopes",
            "created_at",
            "updated_at",
        }
        assert set(hints.keys()) == expected_keys

    def test_oauth_token_data_can_be_created(self) -> None:
        """Test OAuthTokenData can be instantiated."""
        data: OAuthTokenData = {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "user_id": "987e6543-e21b-12d3-a456-426614174000",
            "provider": "google",
            "access_token": "ya29.test",
            "refresh_token": "1//test",
            "expires_at": "2026-01-07T15:00:00Z",
            "scopes": ["gmail.readonly"],
        }
        assert data["provider"] == "google"
        assert data["scopes"] == ["gmail.readonly"]


class TestSyncStateData:
    """Tests for SyncStateData TypedDict."""

    def test_sync_state_data_has_expected_keys(self) -> None:
        """Test SyncStateData has all expected keys."""
        hints = get_type_hints(SyncStateData)
        expected_keys = {
            "id",
            "user_id",
            "last_history_id",
            "last_sync_at",
            "sync_status",
            "error_message",
            "emails_synced",
            "created_at",
            "updated_at",
        }
        assert set(hints.keys()) == expected_keys

    def test_sync_state_data_can_be_created(self) -> None:
        """Test SyncStateData can be instantiated."""
        data: SyncStateData = {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "user_id": "987e6543-e21b-12d3-a456-426614174000",
            "sync_status": "idle",
            "emails_synced": 0,
        }
        assert data["sync_status"] == "idle"


class TestClusterMetadataData:
    """Tests for ClusterMetadataData TypedDict."""

    def test_cluster_metadata_data_has_expected_keys(self) -> None:
        """Test ClusterMetadataData has all expected keys."""
        hints = get_type_hints(ClusterMetadataData)
        expected_keys = {
            "id",
            "org_id",
            "user_id",
            "dimension",
            "cluster_id",
            "size",
            "representative_email_id",
            "auto_label",
            "pct_replied",
            "avg_response_time_hours",
            "avg_relationship_strength",
            "is_project",
            "project_status",
            "last_activity_at",
            "created_at",
        }
        assert set(hints.keys()) == expected_keys

    def test_cluster_metadata_data_can_be_created(self) -> None:
        """Test ClusterMetadataData can be instantiated."""
        data: ClusterMetadataData = {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "dimension": "content",
            "cluster_id": 5,
            "size": 42,
            "is_project": True,
            "project_status": "active",
        }
        assert data["dimension"] == "content"
        assert data["is_project"] is True
