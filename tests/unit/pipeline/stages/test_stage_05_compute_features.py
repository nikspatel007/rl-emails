"""Tests for Stage 5: Compute ML features."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rl_emails.core.config import Config
from rl_emails.pipeline.stages import stage_05_compute_features
from rl_emails.pipeline.stages.base import StageResult


class TestExtractPlainText:
    """Tests for extract_plain_text function."""

    def test_none_input(self) -> None:
        """Test None input returns empty string."""
        assert stage_05_compute_features.extract_plain_text(None) == ""

    def test_empty_input(self) -> None:
        """Test empty input returns empty string."""
        assert stage_05_compute_features.extract_plain_text("") == ""

    def test_extracts_text(self) -> None:
        """Test HTML text extraction."""
        html = "<html><body><p>Hello World</p></body></html>"
        result = stage_05_compute_features.extract_plain_text(html)
        assert "Hello World" in result

    def test_removes_script_and_style(self) -> None:
        """Test script and style elements are removed."""
        html = "<html><script>alert('x')</script><style>body{}</style><p>Text</p></html>"
        result = stage_05_compute_features.extract_plain_text(html)
        assert "alert" not in result
        assert "body" not in result
        assert "Text" in result


class TestCountWords:
    """Tests for count_words function."""

    def test_none_input(self) -> None:
        """Test None input returns 0."""
        assert stage_05_compute_features.count_words(None) == 0

    def test_empty_input(self) -> None:
        """Test empty input returns 0."""
        assert stage_05_compute_features.count_words("") == 0

    def test_counts_words(self) -> None:
        """Test word counting."""
        assert stage_05_compute_features.count_words("Hello World") == 2


class TestDetectServiceEmail:
    """Tests for detect_service_email function."""

    def test_not_service(self) -> None:
        """Test non-service email."""
        is_service, conf, stype = stage_05_compute_features.detect_service_email(
            "friend@example.org", "Hello there", "How are you doing today?"
        )
        assert is_service is False
        assert conf == 0.0
        assert stype is None

    def test_service_domain(self) -> None:
        """Test service domain detection."""
        is_service, conf, stype = stage_05_compute_features.detect_service_email(
            "noreply@company.com", "Order Update", "Your order shipped"
        )
        assert is_service is True
        assert conf > 0

    def test_unsubscribe_detection(self) -> None:
        """Test unsubscribe link detection."""
        is_service, conf, stype = stage_05_compute_features.detect_service_email(
            "news@company.com", "Newsletter", "Click here to unsubscribe"
        )
        assert is_service is True

    def test_newsletter_type(self) -> None:
        """Test newsletter type detection."""
        is_service, conf, stype = stage_05_compute_features.detect_service_email(
            "newsletter@company.com", "Weekly Newsletter", "Content"
        )
        assert stype == "newsletter"

    def test_transactional_type(self) -> None:
        """Test transactional type detection."""
        is_service, conf, stype = stage_05_compute_features.detect_service_email(
            "noreply@shop.com", "Order Confirmation", "Your order is confirmed"
        )
        assert stype == "transactional"


class TestGetTimeBucket:
    """Tests for get_time_bucket function."""

    def test_morning(self) -> None:
        """Test morning bucket (5-12)."""
        assert stage_05_compute_features.get_time_bucket(8) == "morning"

    def test_afternoon(self) -> None:
        """Test afternoon bucket (12-17)."""
        assert stage_05_compute_features.get_time_bucket(14) == "afternoon"

    def test_evening(self) -> None:
        """Test evening bucket (17-21)."""
        assert stage_05_compute_features.get_time_bucket(19) == "evening"

    def test_night(self) -> None:
        """Test night bucket (21-5)."""
        assert stage_05_compute_features.get_time_bucket(2) == "night"
        assert stage_05_compute_features.get_time_bucket(22) == "night"


class TestComputeServiceImportance:
    """Tests for compute_service_importance function."""

    def test_not_service(self) -> None:
        """Test non-service email returns 0."""
        result = stage_05_compute_features.compute_service_importance(
            "friend@gmail.com", "Hello", is_service=False
        )
        assert result == 0.0

    def test_important_keywords(self) -> None:
        """Test important keyword detection."""
        result = stage_05_compute_features.compute_service_importance(
            "noreply@bank.com", "Security Alert", is_service=True
        )
        assert result > 0.5

    def test_low_importance_keywords(self) -> None:
        """Test low importance keyword detection."""
        result = stage_05_compute_features.compute_service_importance(
            "newsletter@shop.com", "Exclusive Offer Sale", is_service=True
        )
        assert result < 0.5

    def test_mixed_keywords(self) -> None:
        """Test mixed important and low keywords."""
        result = stage_05_compute_features.compute_service_importance(
            "noreply@shop.com", "Order Confirmation Exclusive", is_service=True
        )
        # important > low, so should be moderate-high
        assert 0.3 < result < 0.8

    def test_more_low_than_important(self) -> None:
        """Test more low importance keywords than important (both present)."""
        # Must have BOTH important and low keywords, with low > important
        # "order" is important, "sale deal offer" are low importance
        result = stage_05_compute_features.compute_service_importance(
            "newsletter@shop.com", "Order Sale Deal Offer", is_service=True
        )
        # low (3) > important (1), should be lower
        assert result < 0.5

    def test_sender_patterns(self) -> None:
        """Test important sender pattern detection."""
        result = stage_05_compute_features.compute_service_importance(
            "security@bank.com", "Notification", is_service=True
        )
        assert result > 0.4


class TestIsBusinessHours:
    """Tests for is_business_hours function."""

    def test_business_hours(self) -> None:
        """Test during business hours."""
        assert stage_05_compute_features.is_business_hours(10, 1) is True  # Tue 10am

    def test_after_hours(self) -> None:
        """Test after business hours."""
        assert stage_05_compute_features.is_business_hours(20, 1) is False  # Tue 8pm

    def test_weekend(self) -> None:
        """Test weekend."""
        assert stage_05_compute_features.is_business_hours(10, 5) is False  # Sat 10am


class TestComputeRelationshipStrength:
    """Tests for compute_relationship_strength function."""

    def test_strong_relationship(self) -> None:
        """Test strong relationship scores high."""
        result = stage_05_compute_features.compute_relationship_strength(
            emails_30d=10,
            user_replied_rate=0.8,
            sender_reply_rate=0.7,
            days_since_interaction=1,
            user_initiated_ratio=0.5,
        )
        assert result > 0.6

    def test_weak_relationship(self) -> None:
        """Test weak relationship scores low."""
        result = stage_05_compute_features.compute_relationship_strength(
            emails_30d=1,
            user_replied_rate=0.0,
            sender_reply_rate=0.0,
            days_since_interaction=90,
            user_initiated_ratio=0.0,
        )
        assert result < 0.2

    def test_capped_at_1(self) -> None:
        """Test result is capped at 1.0."""
        result = stage_05_compute_features.compute_relationship_strength(
            emails_30d=100,
            user_replied_rate=1.0,
            sender_reply_rate=1.0,
            days_since_interaction=0,
            user_initiated_ratio=1.0,
        )
        assert result <= 1.0


class TestComputeUrgencyScore:
    """Tests for compute_urgency_score function."""

    def test_high_urgency(self) -> None:
        """Test high urgency email."""
        result = stage_05_compute_features.compute_urgency_score(
            relationship_strength=0.9,
            days_since_received=0,
            is_business_hours_flag=True,
            has_attachments=True,
        )
        assert result > 0.7

    def test_low_urgency(self) -> None:
        """Test low urgency email."""
        result = stage_05_compute_features.compute_urgency_score(
            relationship_strength=0.1,
            days_since_received=14,
            is_business_hours_flag=False,
            has_attachments=False,
        )
        assert result < 0.3


class TestCreateEmailFeaturesTable:
    """Tests for create_email_features_table function."""

    @pytest.mark.asyncio
    async def test_creates_table(self) -> None:
        """Test table creation."""
        conn = AsyncMock()

        await stage_05_compute_features.create_email_features_table(conn)

        assert conn.execute.call_count == 2  # CREATE TABLE + CREATE INDEX


class TestComputeSenderStats:
    """Tests for compute_sender_stats function."""

    @pytest.mark.asyncio
    async def test_computes_stats(self) -> None:
        """Test sender stats computation."""
        conn = AsyncMock()
        conn.fetchval.return_value = datetime(2024, 1, 15)
        conn.fetch.return_value = [
            {
                "sender": "test@example.com",
                "emails_all": 10,
                "emails_7d": 2,
                "emails_30d": 5,
                "emails_90d": 8,
                "last_received": datetime(2024, 1, 14),
                "user_replied_count": 3,
                "avg_response_hours": 2.5,
                "user_initiated_count": 2,
                "sender_reply_rate": 0.5,
            }
        ]

        stats, ref_date = await stage_05_compute_features.compute_sender_stats(conn)

        assert "test@example.com" in stats
        assert stats["test@example.com"]["emails_all"] == 10

    @pytest.mark.asyncio
    async def test_handles_null_date(self) -> None:
        """Test handles NULL reference date."""
        conn = AsyncMock()
        conn.fetchval.return_value = None
        conn.fetch.return_value = []

        stats, ref_date = await stage_05_compute_features.compute_sender_stats(conn)

        assert isinstance(ref_date, datetime)


class TestComputeFeaturesBatch:
    """Tests for compute_features_batch function."""

    @pytest.mark.asyncio
    async def test_computes_features(self) -> None:
        """Test feature computation for batch."""
        conn = AsyncMock()
        conn.fetch.return_value = [
            {
                "id": 1,
                "from_email": "sender@example.com",
                "subject": "Test Subject",
                "body_text": "Hello world",
                "date_parsed": datetime(2024, 1, 10, 14, 30),
                "to_emails": ["recipient@example.com"],
                "cc_emails": None,
                "labels": None,
            }
        ]

        sender_stats: dict[str, dict[str, Any]] = {
            "sender@example.com": {
                "emails_all": 5,
                "emails_7d": 2,
                "emails_30d": 3,
                "emails_90d": 4,
                "user_replied_count": 2,
                "avg_response_hours": 1.0,
                "user_initiated_count": 1,
                "sender_reply_rate": 0.5,
                "last_received": datetime(2024, 1, 9),
            }
        }

        features = await stage_05_compute_features.compute_features_batch(
            conn, [1], sender_stats, datetime(2024, 1, 15)
        )

        assert len(features) == 1
        assert features[0]["email_id"] == 1
        assert features[0]["emails_from_sender_30d"] == 3

    @pytest.mark.asyncio
    async def test_handles_missing_sender(self) -> None:
        """Test handles sender not in stats."""
        conn = AsyncMock()
        conn.fetch.return_value = [
            {
                "id": 1,
                "from_email": "unknown@example.com",
                "subject": "Test",
                "body_text": "Hello",
                "date_parsed": datetime(2024, 1, 10),
                "to_emails": None,
                "cc_emails": None,
                "labels": None,
            }
        ]

        features = await stage_05_compute_features.compute_features_batch(
            conn, [1], {}, datetime(2024, 1, 15)
        )

        assert len(features) == 1
        assert features[0]["emails_from_sender_all"] == 0

    @pytest.mark.asyncio
    async def test_handles_null_date(self) -> None:
        """Test handles NULL date_parsed."""
        conn = AsyncMock()
        conn.fetch.return_value = [
            {
                "id": 1,
                "from_email": "sender@example.com",
                "subject": "Test",
                "body_text": "Hello",
                "date_parsed": None,
                "to_emails": None,
                "cc_emails": None,
                "labels": None,
            }
        ]

        features = await stage_05_compute_features.compute_features_batch(
            conn, [1], {}, datetime(2024, 1, 15)
        )

        assert len(features) == 1
        assert features[0]["hour_of_day"] == 0

    @pytest.mark.asyncio
    async def test_detects_attachments(self) -> None:
        """Test attachment detection from labels."""
        conn = AsyncMock()
        conn.fetch.return_value = [
            {
                "id": 1,
                "from_email": "sender@example.com",
                "subject": "Test",
                "body_text": "Hello",
                "date_parsed": datetime(2024, 1, 10),
                "to_emails": None,
                "cc_emails": None,
                "labels": ["Has attachment"],
            }
        ]

        features = await stage_05_compute_features.compute_features_batch(
            conn, [1], {}, datetime(2024, 1, 15)
        )

        assert features[0]["has_attachments"] is True


class TestStoreFeaturesBatch:
    """Tests for store_features_batch function."""

    @pytest.mark.asyncio
    async def test_empty_list(self) -> None:
        """Test empty features list."""
        conn = AsyncMock()

        result = await stage_05_compute_features.store_features_batch(conn, [])

        assert result == 0
        conn.transaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_stores_features(self) -> None:
        """Test storing features."""
        conn = AsyncMock()
        # Mock the transaction context manager properly
        conn.transaction = MagicMock(return_value=AsyncMock())

        features = [
            {
                "email_id": 1,
                "emails_from_sender_7d": 1,
                "emails_from_sender_30d": 2,
                "emails_from_sender_90d": 3,
                "emails_from_sender_all": 4,
                "user_replied_to_sender_count": 1,
                "user_replied_to_sender_rate": 0.5,
                "avg_response_time_hours": 1.0,
                "user_initiated_ratio": 0.3,
                "days_since_last_interaction": 5,
                "sender_replies_to_you_rate": 0.4,
                "relationship_strength": 0.6,
                "is_service_email": False,
                "service_confidence": 0.0,
                "service_type": None,
                "service_importance": 0.0,
                "has_unsubscribe_link": False,
                "has_list_unsubscribe_header": False,
                "from_common_service_domain": False,
                "hour_of_day": 10,
                "day_of_week": 1,
                "is_weekend": False,
                "is_business_hours": True,
                "days_since_received": 2,
                "is_recent": True,
                "time_bucket": "morning",
                "urgency_score": 0.5,
                "subject_word_count": 3,
                "body_word_count": 10,
                "has_attachments": False,
                "attachment_count": 0,
                "recipient_count": 1,
            }
        ]

        result = await stage_05_compute_features.store_features_batch(conn, features)

        assert result == 1


class TestConvertDbUrl:
    """Tests for _convert_db_url function."""

    def test_postgresql_converted(self) -> None:
        """Test postgresql:// is converted."""
        result = stage_05_compute_features._convert_db_url("postgresql://host/db")
        assert result == "postgres://host/db"

    def test_postgres_unchanged(self) -> None:
        """Test postgres:// is unchanged."""
        result = stage_05_compute_features._convert_db_url("postgres://host/db")
        assert result == "postgres://host/db"


class TestRunAsync:
    """Tests for run_async function."""

    @pytest.mark.asyncio
    @patch("rl_emails.pipeline.stages.stage_05_compute_features.asyncpg.connect")
    async def test_runs_pipeline_empty(self, mock_connect: MagicMock) -> None:
        """Test run_async with no emails."""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = datetime(2024, 1, 15)
        mock_conn.fetch.side_effect = [
            [],  # compute_sender_stats
            [],  # get email IDs
        ]
        mock_connect.return_value = mock_conn

        stats = await stage_05_compute_features.run_async("postgres://test", batch_size=100)

        assert stats["processed"] == 0
        mock_conn.close.assert_called_once()

    @pytest.mark.asyncio
    @patch("rl_emails.pipeline.stages.stage_05_compute_features.asyncpg.connect")
    async def test_runs_pipeline_with_emails(self, mock_connect: MagicMock) -> None:
        """Test run_async with emails to process."""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = datetime(2024, 1, 15)
        # Mock the transaction context manager
        mock_conn.transaction = MagicMock(return_value=AsyncMock())

        mock_conn.fetch.side_effect = [
            [],  # compute_sender_stats
            [{"id": 1}, {"id": 2}],  # get email IDs
            # compute_features_batch for batch 1
            [
                {
                    "id": 1,
                    "from_email": "test@example.com",
                    "subject": "Test",
                    "body_text": "Hello",
                    "date_parsed": datetime(2024, 1, 10),
                    "to_emails": None,
                    "cc_emails": None,
                    "labels": None,
                },
                {
                    "id": 2,
                    "from_email": "test@example.com",
                    "subject": "Test 2",
                    "body_text": "World",
                    "date_parsed": datetime(2024, 1, 11),
                    "to_emails": None,
                    "cc_emails": None,
                    "labels": None,
                },
            ],
        ]
        mock_connect.return_value = mock_conn

        stats = await stage_05_compute_features.run_async("postgres://test", batch_size=100)

        assert stats["processed"] == 2
        mock_conn.close.assert_called_once()


class TestRun:
    """Tests for run function."""

    @patch("rl_emails.pipeline.stages.stage_05_compute_features.asyncio.run")
    def test_run_success(self, mock_asyncio_run: MagicMock) -> None:
        """Test successful run."""
        mock_asyncio_run.return_value = {
            "total_emails": 1000,
            "processed": 1000,
            "senders": 100,
        }

        config = Config(database_url="postgresql://test")
        result = stage_05_compute_features.run(config, batch_size=500)

        assert isinstance(result, StageResult)
        assert result.success is True
        assert result.records_processed == 1000
