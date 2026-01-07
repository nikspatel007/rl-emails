"""Tests for Stage 7: Rule-based AI Handleability Classification."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from rl_emails.core.config import Config
from rl_emails.pipeline.stages import stage_07_classify_handleability
from rl_emails.pipeline.stages.base import StageResult


class TestDetectPatterns:
    """Tests for detect_patterns function."""

    def test_detects_question(self) -> None:
        """Test question mark detection in body."""
        patterns = stage_07_classify_handleability.detect_patterns(
            body="can you help me?", subject="help"
        )
        assert patterns["has_question"] is True

    def test_detects_question_in_subject(self) -> None:
        """Test question mark detection in subject."""
        patterns = stage_07_classify_handleability.detect_patterns(
            body="no question here", subject="are you available?"
        )
        assert patterns["has_question"] is True

    def test_no_question(self) -> None:
        """Test no question mark."""
        patterns = stage_07_classify_handleability.detect_patterns(
            body="this is a statement", subject="fyi"
        )
        assert patterns["has_question"] is False

    def test_detects_request(self) -> None:
        """Test request pattern detection."""
        patterns = stage_07_classify_handleability.detect_patterns(
            body="please send me the file", subject="request"
        )
        assert patterns["has_request"] is True

    def test_detects_scheduling(self) -> None:
        """Test scheduling pattern detection."""
        patterns = stage_07_classify_handleability.detect_patterns(
            body="let's schedule a meeting", subject="meeting request"
        )
        assert patterns["has_scheduling"] is True

    def test_detects_deadline(self) -> None:
        """Test deadline pattern detection."""
        patterns = stage_07_classify_handleability.detect_patterns(
            body="need this by friday", subject="urgent task"
        )
        assert patterns["has_deadline"] is True

    def test_detects_approval(self) -> None:
        """Test approval pattern detection."""
        patterns = stage_07_classify_handleability.detect_patterns(
            body="please approve this request", subject="approval needed"
        )
        assert patterns["has_approval"] is True

    def test_detects_newsletter(self) -> None:
        """Test newsletter pattern detection."""
        patterns = stage_07_classify_handleability.detect_patterns(
            body="click here to unsubscribe", subject="weekly news"
        )
        assert patterns["is_newsletter"] is True

    def test_detects_calendar_response(self) -> None:
        """Test calendar response detection."""
        patterns = stage_07_classify_handleability.detect_patterns(
            body="event details", subject="accepted: team meeting"
        )
        assert patterns["is_calendar_response"] is True

    def test_detects_auto_reply(self) -> None:
        """Test auto-reply detection."""
        patterns = stage_07_classify_handleability.detect_patterns(
            body="i am currently out of office", subject="out of office"
        )
        assert patterns["is_auto_reply"] is True

    def test_detects_fyi(self) -> None:
        """Test FYI pattern detection."""
        patterns = stage_07_classify_handleability.detect_patterns(
            body="fyi - here's the update", subject="update"
        )
        assert patterns["is_fyi"] is True


class TestClassifyEmail:
    """Tests for classify_email function."""

    def test_service_marketing_returns_ai_full(self) -> None:
        """Test marketing service emails are classified as ai_full."""
        email: dict[str, Any] = {"body_text": "special offer", "subject": "sale"}
        features: dict[str, Any] = {
            "is_service_email": True,
            "service_type": "marketing",
            "relationship_strength": 0,
        }

        handleability, reason, metadata = stage_07_classify_handleability.classify_email(
            email, features
        )

        assert handleability == "ai_full"
        assert reason == "service_marketing"
        assert metadata["action"] == "FILE_TO_FOLDER"
        assert metadata["folder"] == "Promotions"

    def test_service_newsletter_returns_ai_full(self) -> None:
        """Test newsletter service emails are classified as ai_full."""
        email: dict[str, Any] = {"body_text": "weekly digest", "subject": "newsletter"}
        features: dict[str, Any] = {
            "is_service_email": True,
            "service_type": "newsletter",
            "relationship_strength": 0,
        }

        handleability, reason, metadata = stage_07_classify_handleability.classify_email(
            email, features
        )

        assert handleability == "ai_full"
        assert reason == "service_newsletter"
        assert metadata["folder"] == "Updates"

    def test_calendar_response_returns_ai_full(self) -> None:
        """Test calendar responses are classified as ai_full."""
        email: dict[str, Any] = {"body_text": "event details", "subject": "Accepted: Meeting"}
        features: dict[str, Any] = {"is_service_email": False, "relationship_strength": 0.5}

        handleability, reason, metadata = stage_07_classify_handleability.classify_email(
            email, features
        )

        assert handleability == "ai_full"
        assert reason == "calendar_response"

    def test_auto_reply_returns_ai_full(self) -> None:
        """Test auto-replies are classified as ai_full."""
        email: dict[str, Any] = {
            "body_text": "i will respond when i return",
            "subject": "Out of Office",
        }
        features: dict[str, Any] = {"is_service_email": False, "relationship_strength": 0.3}

        handleability, reason, metadata = stage_07_classify_handleability.classify_email(
            email, features
        )

        assert handleability == "ai_full"
        assert reason == "auto_reply"

    def test_newsletter_pattern_low_relationship_returns_ai_full(self) -> None:
        """Test newsletter patterns with low relationship are ai_full."""
        email: dict[str, Any] = {
            "body_text": "click to unsubscribe from this list",
            "subject": "updates",
        }
        features: dict[str, Any] = {"is_service_email": False, "relationship_strength": 0.1}

        handleability, reason, metadata = stage_07_classify_handleability.classify_email(
            email, features
        )

        assert handleability == "ai_full"
        assert reason == "newsletter_pattern"

    def test_transactional_order_returns_ai_full(self) -> None:
        """Test transactional order emails are ai_full."""
        email: dict[str, Any] = {
            "body_text": "your order has shipped tracking number",
            "subject": "order confirmation",
        }
        features: dict[str, Any] = {
            "is_service_email": True,
            "service_type": "transactional",
            "relationship_strength": 0,
        }

        handleability, reason, metadata = stage_07_classify_handleability.classify_email(
            email, features
        )

        assert handleability == "ai_full"
        assert reason == "transactional"
        assert metadata["track_delivery"] is True

    def test_approval_request_returns_human_required(self) -> None:
        """Test approval requests need human attention."""
        email: dict[str, Any] = {
            "body_text": "please approve this budget request",
            "subject": "approval needed",
        }
        features: dict[str, Any] = {"is_service_email": False, "relationship_strength": 0.3}

        handleability, reason, metadata = stage_07_classify_handleability.classify_email(
            email, features
        )

        assert handleability == "human_required"
        assert reason == "approval_request"

    def test_high_relationship_returns_human_required(self) -> None:
        """Test high relationship emails need human attention."""
        email: dict[str, Any] = {"body_text": "hey how are you", "subject": "hi"}
        features: dict[str, Any] = {"is_service_email": False, "relationship_strength": 0.8}

        handleability, reason, metadata = stage_07_classify_handleability.classify_email(
            email, features
        )

        assert handleability == "human_required"
        assert reason == "important_relationship"

    def test_explicit_reply_request_returns_human_required(self) -> None:
        """Test explicit reply requests from known contacts need human."""
        email: dict[str, Any] = {
            "body_text": "please let me know your thoughts",
            "subject": "question",
        }
        features: dict[str, Any] = {"is_service_email": False, "relationship_strength": 0.4}

        handleability, reason, metadata = stage_07_classify_handleability.classify_email(
            email, features
        )

        assert handleability == "human_required"
        assert reason == "explicit_reply_request"

    def test_deadline_request_returns_human_required(self) -> None:
        """Test deadline requests need human attention."""
        email: dict[str, Any] = {
            "body_text": "please send the report by friday asap",
            "subject": "urgent",
        }
        features: dict[str, Any] = {"is_service_email": False, "relationship_strength": 0.2}

        handleability, reason, metadata = stage_07_classify_handleability.classify_email(
            email, features
        )

        assert handleability == "human_required"
        assert reason == "deadline_request"

    def test_scheduling_returns_ai_partial(self) -> None:
        """Test scheduling requests from weak relationships are ai_partial."""
        email: dict[str, Any] = {"body_text": "can we schedule a call", "subject": "meeting"}
        features: dict[str, Any] = {"is_service_email": False, "relationship_strength": 0.3}

        handleability, reason, metadata = stage_07_classify_handleability.classify_email(
            email, features
        )

        assert handleability == "ai_partial"
        assert reason == "scheduling_request"

    def test_fyi_from_contact_returns_ai_partial(self) -> None:
        """Test FYI emails from contacts are ai_partial."""
        email: dict[str, Any] = {"body_text": "fyi here is the update", "subject": "update"}
        features: dict[str, Any] = {"is_service_email": False, "relationship_strength": 0.2}

        handleability, reason, metadata = stage_07_classify_handleability.classify_email(
            email, features
        )

        assert handleability == "ai_partial"
        assert reason == "fyi_from_contact"

    def test_attachment_reference_returns_ai_partial(self) -> None:
        """Test attachment references are ai_partial."""
        email: dict[str, Any] = {
            "body_text": "please find attached the report",
            "subject": "report",
        }
        features: dict[str, Any] = {"is_service_email": False, "relationship_strength": 0.1}

        handleability, reason, metadata = stage_07_classify_handleability.classify_email(
            email, features
        )

        assert handleability == "ai_partial"
        assert reason == "has_attachment"

    def test_ambiguous_returns_needs_llm(self) -> None:
        """Test ambiguous emails return needs_llm."""
        email: dict[str, Any] = {"body_text": "thanks for the update", "subject": "re: project"}
        features: dict[str, Any] = {"is_service_email": False, "relationship_strength": 0.3}

        handleability, reason, metadata = stage_07_classify_handleability.classify_email(
            email, features
        )

        assert handleability == "needs_llm"
        assert reason == "ambiguous"
        assert "llm_priority" in metadata

    def test_llm_priority_high_for_important_person(self) -> None:
        """Test LLM priority is high for important relationships."""
        email: dict[str, Any] = {"body_text": "just checking in", "subject": "hi"}
        features: dict[str, Any] = {"is_service_email": False, "relationship_strength": 0.5}

        handleability, _, metadata = stage_07_classify_handleability.classify_email(email, features)

        assert handleability == "needs_llm"
        assert metadata["llm_priority"] == 1

    def test_llm_priority_low_for_service(self) -> None:
        """Test LLM priority is low for service emails."""
        email: dict[str, Any] = {"body_text": "system notification", "subject": "alert"}
        features: dict[str, Any] = {
            "is_service_email": True,
            "service_type": "other",
            "relationship_strength": 0,
        }

        handleability, _, metadata = stage_07_classify_handleability.classify_email(email, features)

        assert handleability == "needs_llm"
        assert metadata["llm_priority"] == 4

    def test_handles_none_values(self) -> None:
        """Test handling of None values in email and features."""
        email: dict[str, Any] = {"body_text": None, "subject": None}
        features: dict[str, Any] = {
            "is_service_email": None,
            "service_type": None,
            "relationship_strength": None,
        }

        handleability, reason, metadata = stage_07_classify_handleability.classify_email(
            email, features
        )

        # Should not raise, returns needs_llm
        assert handleability == "needs_llm"
        assert reason == "ambiguous"

    def test_transactional_without_order_keywords_falls_through(self) -> None:
        """Test transactional service without order keywords falls through."""
        email: dict[str, Any] = {"body_text": "account update info", "subject": "update"}
        features: dict[str, Any] = {
            "is_service_email": True,
            "service_type": "transactional",
            "relationship_strength": 0,
        }

        handleability, reason, _ = stage_07_classify_handleability.classify_email(email, features)

        # Falls through to needs_llm since no other rules match
        assert handleability == "needs_llm"

    def test_scheduling_high_relationship_hits_important(self) -> None:
        """Test scheduling from very high relationship hits important_relationship."""
        email: dict[str, Any] = {"body_text": "let's schedule a meeting", "subject": "meeting"}
        features: dict[str, Any] = {"is_service_email": False, "relationship_strength": 0.7}

        # relationship > 0.6 triggers human_required first
        handleability, reason, _ = stage_07_classify_handleability.classify_email(email, features)

        assert handleability == "human_required"
        assert reason == "important_relationship"

    def test_scheduling_medium_relationship_falls_through(self) -> None:
        """Test scheduling from medium relationship (0.5) falls through."""
        email: dict[str, Any] = {"body_text": "let's schedule a meeting", "subject": "sync"}
        features: dict[str, Any] = {"is_service_email": False, "relationship_strength": 0.5}

        # Has scheduling but relationship == 0.5 (not < 0.5), falls through
        handleability, reason, metadata = stage_07_classify_handleability.classify_email(
            email, features
        )

        # Falls through to needs_llm with high priority (relationship > 0.4)
        assert handleability == "needs_llm"
        assert metadata["llm_priority"] == 1

    def test_confirm_request_low_relationship(self) -> None:
        """Test confirm request with low relationship returns ai_partial."""
        email: dict[str, Any] = {"body_text": "please confirm receipt", "subject": "info"}
        features: dict[str, Any] = {"is_service_email": False, "relationship_strength": 0.2}

        handleability, reason, _ = stage_07_classify_handleability.classify_email(email, features)

        assert handleability == "ai_partial"
        assert reason == "confirm_request"

    def test_llm_priority_medium_high_for_request(self) -> None:
        """Test LLM priority is medium-high for request pattern."""
        email: dict[str, Any] = {"body_text": "please review this", "subject": "review"}
        features: dict[str, Any] = {"is_service_email": False, "relationship_strength": 0.3}

        handleability, _, metadata = stage_07_classify_handleability.classify_email(email, features)

        assert handleability == "needs_llm"
        assert metadata["llm_priority"] == 2

    def test_llm_priority_lowest_for_unknown_sender(self) -> None:
        """Test LLM priority is lowest for unknown sender."""
        email: dict[str, Any] = {"body_text": "hello there", "subject": "hi"}
        features: dict[str, Any] = {"is_service_email": False, "relationship_strength": 0.05}

        handleability, _, metadata = stage_07_classify_handleability.classify_email(email, features)

        assert handleability == "needs_llm"
        assert metadata["llm_priority"] == 5


class TestRun:
    """Tests for run function."""

    @patch("rl_emails.pipeline.stages.stage_07_classify_handleability.get_connection")
    def test_run_returns_stage_result(self, mock_get_connection: MagicMock) -> None:
        """Test run returns proper StageResult."""
        mock_conn = MagicMock()
        mock_get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_connection.return_value.__exit__ = MagicMock(return_value=False)

        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        # Mock no emails
        mock_cursor.fetchall.return_value = []

        config = Config(database_url="postgresql://test")
        result = stage_07_classify_handleability.run(config)

        assert isinstance(result, StageResult)
        assert result.success is True
        assert result.records_processed == 0

    @patch("rl_emails.pipeline.stages.stage_07_classify_handleability.execute_values")
    @patch("rl_emails.pipeline.stages.stage_07_classify_handleability.get_connection")
    def test_run_processes_emails(
        self, mock_get_connection: MagicMock, mock_execute_values: MagicMock
    ) -> None:
        """Test run processes emails and inserts results."""
        mock_conn = MagicMock()
        mock_get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_connection.return_value.__exit__ = MagicMock(return_value=False)

        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        # Mock emails
        mock_cursor.fetchall.return_value = [
            (1, "Test subject", "Test body", "test@example.com", 0.5, False, None, 0.3),
            (2, "Newsletter", "unsubscribe link", "news@example.com", 0.0, True, "newsletter", 0.1),
        ]

        config = Config(database_url="postgresql://test")
        result = stage_07_classify_handleability.run(config)

        assert result.success is True
        assert result.records_processed == 2
        assert mock_execute_values.called

    @patch("rl_emails.pipeline.stages.stage_07_classify_handleability.get_connection")
    def test_run_clears_table_first(self, mock_get_connection: MagicMock) -> None:
        """Test run clears classification table before inserting."""
        mock_conn = MagicMock()
        mock_get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_connection.return_value.__exit__ = MagicMock(return_value=False)

        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []

        config = Config(database_url="postgresql://test")
        stage_07_classify_handleability.run(config)

        # Check that DELETE was called
        calls = [str(c) for c in mock_cursor.execute.call_args_list]
        assert any("DELETE FROM email_ai_classification" in str(c) for c in calls)

    @patch("rl_emails.pipeline.stages.stage_07_classify_handleability.execute_values")
    @patch("rl_emails.pipeline.stages.stage_07_classify_handleability.get_connection")
    def test_run_returns_category_counts_in_metadata(
        self, mock_get_connection: MagicMock, mock_execute_values: MagicMock
    ) -> None:
        """Test run returns category counts in metadata."""
        mock_conn = MagicMock()
        mock_get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_connection.return_value.__exit__ = MagicMock(return_value=False)

        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        # Mock emails - one needs_llm, one ai_full (newsletter)
        mock_cursor.fetchall.return_value = [
            (1, "Test", "body text", "test@example.com", 0.3, False, None, 0.3),
            (2, "News", "unsubscribe here", "news@example.com", 0.0, True, "newsletter", 0.1),
        ]

        config = Config(database_url="postgresql://test")
        result = stage_07_classify_handleability.run(config)

        assert result.metadata is not None
        assert "category_counts" in result.metadata
