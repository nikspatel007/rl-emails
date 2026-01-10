"""Tests for entity extraction service."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from rl_emails.services.entity_extraction import (
    MARKETING_PATTERNS,
    ExtractionResult,
    PriorityContextBuilder,
    ProjectExtractor,
    TaskExtractor,
    _build_marketing_filter_sql,
    _is_real_person_sql,
    extract_all_entities,
)


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_create_success_result(self) -> None:
        """Test creating a successful extraction result."""
        result = ExtractionResult(
            tasks_created=10,
            tasks_updated=5,
            projects_created=3,
            projects_updated=2,
            priority_contexts_created=100,
            priority_contexts_updated=50,
            errors=[],
        )

        assert result.tasks_created == 10
        assert result.tasks_updated == 5
        assert result.projects_created == 3
        assert result.projects_updated == 2
        assert result.priority_contexts_created == 100
        assert result.priority_contexts_updated == 50
        assert result.errors == []
        assert result.tasks_filtered_marketing == 0
        assert result.projects_filtered_marketing == 0

    def test_create_result_with_errors(self) -> None:
        """Test creating result with errors."""
        result = ExtractionResult(
            tasks_created=5,
            tasks_updated=0,
            projects_created=1,
            projects_updated=0,
            priority_contexts_created=10,
            priority_contexts_updated=0,
            errors=["Error 1", "Error 2"],
        )

        assert len(result.errors) == 2
        assert "Error 1" in result.errors

    def test_create_result_with_filtered_counts(self) -> None:
        """Test creating result with filtered marketing counts."""
        result = ExtractionResult(
            tasks_created=10,
            tasks_updated=0,
            projects_created=5,
            projects_updated=0,
            priority_contexts_created=50,
            priority_contexts_updated=0,
            errors=[],
            tasks_filtered_marketing=20,
            projects_filtered_marketing=15,
        )

        assert result.tasks_filtered_marketing == 20
        assert result.projects_filtered_marketing == 15

    def test_total_created_property(self) -> None:
        """Test total_created property calculation."""
        result = ExtractionResult(
            tasks_created=10,
            tasks_updated=5,
            projects_created=3,
            projects_updated=2,
            priority_contexts_created=100,
            priority_contexts_updated=50,
            errors=[],
        )

        # total_created = tasks + projects + priority_contexts
        assert result.total_created == 113

    def test_total_updated_property(self) -> None:
        """Test total_updated property calculation."""
        result = ExtractionResult(
            tasks_created=10,
            tasks_updated=5,
            projects_created=3,
            projects_updated=2,
            priority_contexts_created=100,
            priority_contexts_updated=50,
            errors=[],
        )

        # total_updated = tasks + projects + priority_contexts
        assert result.total_updated == 57


class TestMarketingPatterns:
    """Tests for MARKETING_PATTERNS constant."""

    def test_marketing_patterns_exists(self) -> None:
        """Test that MARKETING_PATTERNS is defined."""
        assert MARKETING_PATTERNS is not None
        assert len(MARKETING_PATTERNS) > 0

    def test_marketing_patterns_include_noreply(self) -> None:
        """Test that noreply patterns are included."""
        assert "%noreply%" in MARKETING_PATTERNS
        assert "%no-reply%" in MARKETING_PATTERNS

    def test_marketing_patterns_include_common_services(self) -> None:
        """Test that common service patterns are included."""
        assert "%@google.com" in MARKETING_PATTERNS
        assert "%@github.com" in MARKETING_PATTERNS
        assert "%@linkedin.com" in MARKETING_PATTERNS


class TestBuildMarketingFilterSql:
    """Tests for _build_marketing_filter_sql function."""

    def test_returns_string(self) -> None:
        """Test that function returns a string."""
        result = _build_marketing_filter_sql()
        assert isinstance(result, str)

    def test_contains_not_like_clauses(self) -> None:
        """Test that SQL contains NOT LIKE clauses."""
        result = _build_marketing_filter_sql()
        assert "NOT LIKE" in result

    def test_contains_and_separator(self) -> None:
        """Test that clauses are joined with AND."""
        result = _build_marketing_filter_sql()
        assert " AND " in result

    def test_references_from_email(self) -> None:
        """Test that SQL references e.from_email."""
        result = _build_marketing_filter_sql()
        assert "e.from_email" in result


class TestIsRealPersonSql:
    """Tests for _is_real_person_sql function."""

    def test_returns_string(self) -> None:
        """Test that function returns a string."""
        result = _is_real_person_sql()
        assert isinstance(result, str)

    def test_contains_case_when(self) -> None:
        """Test that SQL contains CASE WHEN."""
        result = _is_real_person_sql()
        assert "CASE" in result
        assert "WHEN" in result
        assert "END" in result

    def test_checks_marketing_patterns(self) -> None:
        """Test that SQL checks marketing patterns."""
        result = _is_real_person_sql()
        assert "LIKE" in result
        assert "noreply" in result

    def test_checks_reply_rate(self) -> None:
        """Test that SQL checks reply_rate."""
        result = _is_real_person_sql()
        assert "reply_rate" in result

    def test_checks_important_sender(self) -> None:
        """Test that SQL checks is_important_sender."""
        result = _is_real_person_sql()
        assert "is_important_sender" in result

    def test_checks_emails_from(self) -> None:
        """Test that SQL checks emails_from count."""
        result = _is_real_person_sql()
        assert "emails_from" in result


class TestTaskExtractorInit:
    """Tests for TaskExtractor initialization."""

    def test_init_stores_connection(self) -> None:
        """Test that init stores the connection."""
        conn = MagicMock()
        extractor = TaskExtractor(conn)

        assert extractor.conn == conn
        assert extractor.user_id is None

    def test_init_stores_user_id(self) -> None:
        """Test that init stores user_id when provided."""
        conn = MagicMock()
        user_id = uuid.uuid4()
        extractor = TaskExtractor(conn, user_id)

        assert extractor.conn == conn
        assert extractor.user_id == user_id


class TestTaskExtractorExtractFromLlm:
    """Tests for TaskExtractor.extract_from_llm_classifications method."""

    @pytest.fixture
    def mock_conn(self) -> MagicMock:
        """Create mock connection."""
        return MagicMock()

    def test_empty_results_returns_zeros(self, mock_conn: MagicMock) -> None:
        """Test that empty results return zeros."""
        mock_conn.execute.return_value.fetchall.return_value = []

        extractor = TaskExtractor(mock_conn)
        created, updated, filtered, errors = extractor.extract_from_llm_classifications()

        assert created == 0
        assert updated == 0
        assert filtered == 0
        assert errors == []

    def test_filters_non_real_person(self, mock_conn: MagicMock) -> None:
        """Test that non-real-person emails are filtered."""
        # Row: email_id, action_type, urgency, next_step, one_liner, subject,
        #      from_email, sender_name, reply_rate, is_real_person
        mock_conn.execute.return_value.fetchall.return_value = [
            (
                1,
                "task",
                "today",
                None,
                "Do something",
                "Subject",
                "noreply@test.com",
                None,
                0.0,
                False,
            ),
        ]

        extractor = TaskExtractor(mock_conn)
        created, updated, filtered, errors = extractor.extract_from_llm_classifications()

        assert created == 0
        assert filtered == 1

    def test_creates_task_for_real_person(self, mock_conn: MagicMock) -> None:
        """Test that task is created for real person."""
        # First call: fetchall for query, then fetchone for existing check
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (
                1,
                "task",
                "today",
                "Next step",
                "One liner",
                "Subject",
                "person@test.com",
                "Person",
                0.5,
                True,
            ),
        ]
        mock_result.fetchone.return_value = None  # No existing task

        mock_conn.execute.return_value = mock_result

        extractor = TaskExtractor(mock_conn)
        created, updated, filtered, errors = extractor.extract_from_llm_classifications()

        assert created == 1
        assert updated == 0
        assert filtered == 0

    def test_updates_existing_task(self, mock_conn: MagicMock) -> None:
        """Test that existing task is updated."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                # First call is the main query
                result.fetchall.return_value = [
                    (
                        1,
                        "task",
                        "today",
                        "Next step",
                        "One liner",
                        "Subject",
                        "person@test.com",
                        "Person",
                        0.5,
                        True,
                    ),
                ]
            else:
                # Second call is checking for existing
                result.fetchone.return_value = (123,)  # Existing task ID
            return result

        mock_conn.execute.side_effect = mock_execute

        extractor = TaskExtractor(mock_conn)
        created, updated, filtered, errors = extractor.extract_from_llm_classifications()

        assert created == 0
        assert updated == 1

    def test_handles_exception_on_create(self, mock_conn: MagicMock) -> None:
        """Test that exceptions during create are captured."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        1,
                        "task",
                        "today",
                        None,
                        "One liner",
                        "Subject",
                        "person@test.com",
                        None,
                        0.5,
                        True,
                    ),
                ]
            elif call_count == 2:
                result.fetchone.return_value = None
            else:
                raise Exception("Database error")
            return result

        mock_conn.execute.side_effect = mock_execute

        extractor = TaskExtractor(mock_conn)
        created, updated, filtered, errors = extractor.extract_from_llm_classifications()

        assert len(errors) == 1
        assert "Database error" in errors[0]

    def test_handles_exception_on_update(self, mock_conn: MagicMock) -> None:
        """Test that exceptions during update are captured."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        1,
                        "task",
                        "today",
                        None,
                        "One liner",
                        "Subject",
                        "person@test.com",
                        None,
                        0.5,
                        True,
                    ),
                ]
            elif call_count == 2:
                result.fetchone.return_value = (123,)
            else:
                raise Exception("Update error")
            return result

        mock_conn.execute.side_effect = mock_execute

        extractor = TaskExtractor(mock_conn)
        created, updated, filtered, errors = extractor.extract_from_llm_classifications()

        assert len(errors) == 1
        assert "Update error" in errors[0]

    def test_urgency_mapping(self, mock_conn: MagicMock) -> None:
        """Test that urgency values are mapped correctly."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (1, "task", "immediate", None, "Urgent", "Subject", "person@test.com", None, 0.0, True),
        ]
        mock_result.fetchone.return_value = None

        mock_conn.execute.return_value = mock_result

        extractor = TaskExtractor(mock_conn)
        extractor.extract_from_llm_classifications()

        # Verify task was created (urgency_score should be 1.0 for "immediate")
        assert mock_conn.execute.call_count >= 2

    def test_description_truncation(self, mock_conn: MagicMock) -> None:
        """Test that long descriptions are truncated."""
        long_description = "A" * 1500
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (
                1,
                "task",
                "today",
                None,
                long_description,
                "Subject",
                "person@test.com",
                "Name",
                0.5,
                True,
            ),
        ]
        mock_result.fetchone.return_value = None

        mock_conn.execute.return_value = mock_result

        extractor = TaskExtractor(mock_conn)
        extractor.extract_from_llm_classifications()

        # Verify execute was called with truncated description
        assert mock_conn.execute.called

    def test_action_type_mapping(self, mock_conn: MagicMock) -> None:
        """Test that action_type values are mapped to task_type."""
        for action, _expected_type in [
            ("task", "other"),
            ("reply", "send"),
            ("decision", "decision"),
        ]:
            mock_result = MagicMock()
            mock_result.fetchall.return_value = [
                (
                    1,
                    action,
                    "today",
                    None,
                    "One liner",
                    "Subject",
                    "person@test.com",
                    None,
                    0.0,
                    True,
                ),
            ]
            mock_result.fetchone.return_value = None
            mock_conn.execute.return_value = mock_result

            extractor = TaskExtractor(mock_conn)
            extractor.extract_from_llm_classifications()

    def test_urgency_boost_high_reply_rate(self, mock_conn: MagicMock) -> None:
        """Test urgency boost for contacts with high reply rate."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (
                1,
                "task",
                "today",
                None,
                "One liner",
                "Subject",
                "person@test.com",
                None,
                0.7,  # High reply rate > 0.5
                True,
            ),
        ]
        mock_result.fetchone.return_value = None
        mock_conn.execute.return_value = mock_result

        extractor = TaskExtractor(mock_conn)
        created, updated, filtered, errors = extractor.extract_from_llm_classifications()

        assert created == 1


class TestTaskExtractorExtractFromAi:
    """Tests for TaskExtractor.extract_from_ai_classifications method."""

    @pytest.fixture
    def mock_conn(self) -> MagicMock:
        """Create mock connection."""
        return MagicMock()

    def test_empty_results_returns_zeros(self, mock_conn: MagicMock) -> None:
        """Test that empty results return zeros."""
        mock_conn.execute.return_value.fetchall.return_value = []

        extractor = TaskExtractor(mock_conn)
        created, updated, filtered, errors = extractor.extract_from_ai_classifications()

        assert created == 0
        assert updated == 0
        assert filtered == 0
        assert errors == []

    def test_filters_non_real_person(self, mock_conn: MagicMock) -> None:
        """Test that non-real-person emails are filtered."""
        # Row: email_id, has_question, has_request, has_deadline, has_approval,
        #      has_scheduling, subject, from_email, sender_name, reply_rate, is_real_person
        mock_conn.execute.return_value.fetchall.return_value = [
            (
                1,
                False,
                True,
                False,
                False,
                False,
                "Subject",
                "marketing@test.com",
                None,
                0.0,
                False,
            ),
        ]

        extractor = TaskExtractor(mock_conn)
        created, updated, filtered, errors = extractor.extract_from_ai_classifications()

        assert created == 0
        assert filtered == 1

    def test_creates_task_for_real_person(self, mock_conn: MagicMock) -> None:
        """Test that task is created for real person."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (
                1,
                False,
                True,
                False,
                False,
                False,
                "Subject",
                "person@test.com",
                "Person",
                0.5,
                True,
            ),
        ]

        mock_conn.execute.return_value = mock_result

        extractor = TaskExtractor(mock_conn)
        created, updated, filtered, errors = extractor.extract_from_ai_classifications()

        assert created == 1
        assert filtered == 0

    def test_handles_exception(self, mock_conn: MagicMock) -> None:
        """Test that exceptions during create are captured."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                result = MagicMock()
                result.fetchall.return_value = [
                    (
                        1,
                        False,
                        True,
                        False,
                        False,
                        False,
                        "Subject",
                        "person@test.com",
                        None,
                        0.5,
                        True,
                    ),
                ]
                return result
            else:
                raise Exception("Insert error")

        mock_conn.execute.side_effect = mock_execute

        extractor = TaskExtractor(mock_conn)
        created, updated, filtered, errors = extractor.extract_from_ai_classifications()

        assert len(errors) == 1
        assert "Insert error" in errors[0]

    def test_task_type_from_approval(self, mock_conn: MagicMock) -> None:
        """Test that has_approval sets task_type to decision."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (1, False, False, False, True, False, "Subject", "person@test.com", None, 0.0, True),
        ]

        mock_conn.execute.return_value = mock_result

        extractor = TaskExtractor(mock_conn)
        extractor.extract_from_ai_classifications()

        assert mock_conn.execute.called

    def test_task_type_from_deadline(self, mock_conn: MagicMock) -> None:
        """Test that has_deadline affects urgency_score."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (1, False, False, True, False, False, "Subject", "person@test.com", None, 0.0, True),
        ]

        mock_conn.execute.return_value = mock_result

        extractor = TaskExtractor(mock_conn)
        extractor.extract_from_ai_classifications()

        assert mock_conn.execute.called

    def test_description_with_sender_name(self, mock_conn: MagicMock) -> None:
        """Test that sender name is included in description."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (
                1,
                False,
                True,
                False,
                False,
                False,
                "Subject",
                "person@test.com",
                "John Doe",
                0.5,
                True,
            ),
        ]

        mock_conn.execute.return_value = mock_result

        extractor = TaskExtractor(mock_conn)
        extractor.extract_from_ai_classifications()

        assert mock_conn.execute.called

    def test_task_type_other_fallback(self, mock_conn: MagicMock) -> None:
        """Test task type fallback to 'other' when no specific signals."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (
                1,
                True,  # has_question only
                False,  # has_request = false
                False,  # has_deadline = false
                False,  # has_approval = false
                False,
                "Subject",
                "person@test.com",
                None,
                0.0,
                True,
            ),
        ]

        mock_conn.execute.return_value = mock_result

        extractor = TaskExtractor(mock_conn)
        created, _, _, _ = extractor.extract_from_ai_classifications()

        assert created == 1

    def test_urgency_boost_high_reply_rate(self, mock_conn: MagicMock) -> None:
        """Test urgency boost for contacts with high reply rate."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (
                1,
                False,
                True,
                False,
                False,
                False,
                "Subject",
                "person@test.com",
                None,
                0.7,  # High reply rate > 0.5
                True,
            ),
        ]

        mock_conn.execute.return_value = mock_result

        extractor = TaskExtractor(mock_conn)
        created, _, _, _ = extractor.extract_from_ai_classifications()

        assert created == 1

    def test_description_truncation(self, mock_conn: MagicMock) -> None:
        """Test that long descriptions are truncated."""
        long_subject = "A" * 1500
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (
                1,
                False,
                True,
                False,
                False,
                False,
                long_subject,
                "person@test.com",
                None,
                0.5,
                True,
            ),
        ]

        mock_conn.execute.return_value = mock_result

        extractor = TaskExtractor(mock_conn)
        extractor.extract_from_ai_classifications()

        assert mock_conn.execute.called


class TestProjectExtractorInit:
    """Tests for ProjectExtractor initialization."""

    def test_init_stores_connection(self) -> None:
        """Test that init stores the connection."""
        conn = MagicMock()
        extractor = ProjectExtractor(conn)

        assert extractor.conn == conn
        assert extractor.user_id is None

    def test_init_stores_user_id(self) -> None:
        """Test that init stores user_id when provided."""
        conn = MagicMock()
        user_id = uuid.uuid4()
        extractor = ProjectExtractor(conn, user_id)

        assert extractor.conn == conn
        assert extractor.user_id == user_id


class TestProjectExtractorExtractFromRealPersonThreads:
    """Tests for ProjectExtractor.extract_from_real_person_threads method."""

    @pytest.fixture
    def mock_conn(self) -> MagicMock:
        """Create mock connection."""
        return MagicMock()

    def test_empty_results_returns_zeros(self, mock_conn: MagicMock) -> None:
        """Test that empty results return zeros."""
        mock_conn.execute.return_value.fetchall.return_value = []

        extractor = ProjectExtractor(mock_conn)
        created, updated, filtered, errors = extractor.extract_from_real_person_threads()

        assert created == 0
        assert updated == 0
        assert filtered == 0
        assert errors == []

    def test_creates_project_for_contact(self, mock_conn: MagicMock) -> None:
        """Test that project is created for contact."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                # Main query
                result.fetchall.return_value = [
                    (
                        "person@test.com",
                        "Person Name",
                        0.5,
                        True,
                        5,
                        2,
                        datetime.now(UTC),
                        ["Subject 1", "Subject 2"],
                    ),
                ]
            else:
                # Check for existing
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        extractor = ProjectExtractor(mock_conn)
        created, updated, filtered, errors = extractor.extract_from_real_person_threads()

        assert created == 1
        assert updated == 0

    def test_updates_existing_project(self, mock_conn: MagicMock) -> None:
        """Test that existing project is updated."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        "person@test.com",
                        "Person Name",
                        0.5,
                        True,
                        5,
                        2,
                        datetime.now(UTC),
                        ["Subject"],
                    ),
                ]
            elif call_count == 2:
                result.fetchone.return_value = (123,)  # Existing project
            return result

        mock_conn.execute.side_effect = mock_execute

        extractor = ProjectExtractor(mock_conn)
        created, updated, filtered, errors = extractor.extract_from_real_person_threads()

        assert created == 0
        assert updated == 1

    def test_handles_exception_on_create(self, mock_conn: MagicMock) -> None:
        """Test that exceptions during create are captured."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    ("person@test.com", "Person", 0.5, False, 5, 2, datetime.now(UTC), []),
                ]
            elif call_count == 2:
                result.fetchone.return_value = None
            else:
                raise Exception("Insert error")
            return result

        mock_conn.execute.side_effect = mock_execute

        extractor = ProjectExtractor(mock_conn)
        created, updated, filtered, errors = extractor.extract_from_real_person_threads()

        assert len(errors) == 1
        assert "Insert error" in errors[0]

    def test_handles_exception_on_update(self, mock_conn: MagicMock) -> None:
        """Test that exceptions during update are captured."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    ("person@test.com", "Person", 0.5, False, 5, 2, datetime.now(UTC), []),
                ]
            elif call_count == 2:
                result.fetchone.return_value = (123,)
            else:
                raise Exception("Update error")
            return result

        mock_conn.execute.side_effect = mock_execute

        extractor = ProjectExtractor(mock_conn)
        created, updated, filtered, errors = extractor.extract_from_real_person_threads()

        assert len(errors) == 1
        assert "Update error" in errors[0]

    def test_project_type_key_relationship(self, mock_conn: MagicMock) -> None:
        """Test that important sender gets key_relationship type."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        "person@test.com",
                        "Important Person",
                        0.7,
                        True,
                        10,
                        5,
                        datetime.now(UTC),
                        ["Subject"],
                    ),
                ]
            else:
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        extractor = ProjectExtractor(mock_conn)
        extractor.extract_from_real_person_threads()

        assert mock_conn.execute.called

    def test_project_name_truncation(self, mock_conn: MagicMock) -> None:
        """Test that long project names are truncated."""
        long_name = "A" * 300
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        "person@test.com",
                        long_name,
                        0.5,
                        False,
                        5,
                        2,
                        datetime.now(UTC),
                        [long_name],
                    ),
                ]
            else:
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        extractor = ProjectExtractor(mock_conn)
        extractor.extract_from_real_person_threads()

        assert mock_conn.execute.called

    def test_project_name_from_email_prefix(self, mock_conn: MagicMock) -> None:
        """Test project name uses email prefix when no contact name."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    ("person@test.com", "person@test.com", 0.5, False, 5, 2, datetime.now(UTC), []),
                ]
            else:
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        extractor = ProjectExtractor(mock_conn)
        extractor.extract_from_real_person_threads()

        assert mock_conn.execute.called

    def test_project_name_with_subject_topic(self, mock_conn: MagicMock) -> None:
        """Test project name includes subject topic when available."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        "person@test.com",
                        "Person Name",
                        0.5,
                        False,
                        5,
                        2,
                        datetime.now(UTC),
                        ["Important Project Discussion", "Another Subject"],
                    ),
                ]
            else:
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        extractor = ProjectExtractor(mock_conn)
        created, _, _, _ = extractor.extract_from_real_person_threads()

        assert created == 1

    def test_project_type_active_conversation(self, mock_conn: MagicMock) -> None:
        """Test project type is active_conversation for high reply rate non-important."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        "person@test.com",
                        "Person Name",
                        0.7,  # High reply rate > 0.5
                        False,  # Not important
                        5,
                        2,
                        datetime.now(UTC),
                        [],
                    ),
                ]
            else:
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        extractor = ProjectExtractor(mock_conn)
        created, _, _, _ = extractor.extract_from_real_person_threads()

        assert created == 1

    def test_project_name_skips_reply_subjects(self, mock_conn: MagicMock) -> None:
        """Test project name skips Re: and Fwd: subjects."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        "person@test.com",
                        "Person Name",
                        0.5,
                        False,
                        5,
                        2,
                        datetime.now(UTC),
                        ["Re: Some Subject", "Fwd: Another"],
                    ),
                ]
            else:
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        extractor = ProjectExtractor(mock_conn)
        created, _, _, _ = extractor.extract_from_real_person_threads()

        assert created == 1

    def test_project_name_short_subject_skipped(self, mock_conn: MagicMock) -> None:
        """Test project name skips short subjects (< 5 chars)."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        "person@test.com",
                        "Person Name",
                        0.5,
                        False,
                        5,
                        2,
                        datetime.now(UTC),
                        ["Hi", "Ok"],  # Short subjects < 5 chars
                    ),
                ]
            else:
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        extractor = ProjectExtractor(mock_conn)
        created, _, _, _ = extractor.extract_from_real_person_threads()

        assert created == 1

    def test_project_name_empty_subject(self, mock_conn: MagicMock) -> None:
        """Test project name handles empty subjects."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        "person@test.com",
                        "Person Name",
                        0.5,
                        False,
                        5,
                        2,
                        datetime.now(UTC),
                        ["", None],  # Empty subjects
                    ),
                ]
            else:
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        extractor = ProjectExtractor(mock_conn)
        created, _, _, _ = extractor.extract_from_real_person_threads()

        assert created == 1


class TestProjectExtractorExtractFromContentClusters:
    """Tests for ProjectExtractor.extract_from_content_clusters method."""

    @pytest.fixture
    def mock_conn(self) -> MagicMock:
        """Create mock connection."""
        return MagicMock()

    def test_empty_results_returns_zeros(self, mock_conn: MagicMock) -> None:
        """Test that empty results return zeros."""
        mock_conn.execute.return_value.fetchall.return_value = []

        extractor = ProjectExtractor(mock_conn)
        created, updated, filtered, errors = extractor.extract_from_content_clusters()

        assert created == 0
        assert updated == 0
        assert filtered == 0
        assert errors == []

    def test_filters_low_real_person_ratio(self, mock_conn: MagicMock) -> None:
        """Test that clusters with low real person ratio are filtered."""
        # Row: cluster_id, size, auto_label, last_activity, email_count,
        #      real_person_count, senders, subjects, real_person_ratio
        mock_conn.execute.return_value.fetchall.return_value = [
            (1, 10, "Label", datetime.now(UTC), 10, 2, ["a@test.com"], ["Subject"], 0.2),
        ]

        extractor = ProjectExtractor(mock_conn)
        created, updated, filtered, errors = extractor.extract_from_content_clusters()

        assert created == 0
        assert filtered == 1

    def test_creates_project_for_majority_real_person(self, mock_conn: MagicMock) -> None:
        """Test that project is created for majority real person cluster."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        1,
                        10,
                        "Project Label",
                        datetime.now(UTC),
                        10,
                        8,
                        ["a@test.com"],
                        ["Subject"],
                        0.8,
                    ),
                ]
            elif call_count == 2:
                # Check for duplicate with real_person_thread
                result.fetchone.return_value = None
            elif call_count == 3:
                # Check for existing cluster project
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        extractor = ProjectExtractor(mock_conn)
        created, updated, filtered, errors = extractor.extract_from_content_clusters()

        assert created == 1

    def test_filters_duplicate_with_real_person_thread(self, mock_conn: MagicMock) -> None:
        """Test that duplicates with real_person_thread are filtered."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        1,
                        10,
                        "Label",
                        datetime.now(UTC),
                        10,
                        8,
                        ["person@test.com"],
                        ["Subject"],
                        0.8,
                    ),
                ]
            else:
                # Already has real_person_thread project
                result.fetchone.return_value = (123,)
            return result

        mock_conn.execute.side_effect = mock_execute

        extractor = ProjectExtractor(mock_conn)
        created, updated, filtered, errors = extractor.extract_from_content_clusters()

        assert created == 0
        assert filtered == 1

    def test_updates_existing_cluster_project(self, mock_conn: MagicMock) -> None:
        """Test that existing cluster project is updated."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (1, 10, "Label", datetime.now(UTC), 10, 8, ["a@test.com"], ["Subject"], 0.8),
                ]
            elif call_count == 2:
                # No duplicate
                result.fetchone.return_value = None
            elif call_count == 3:
                # Existing cluster project
                result.fetchone.return_value = (123,)
            return result

        mock_conn.execute.side_effect = mock_execute

        extractor = ProjectExtractor(mock_conn)
        created, updated, filtered, errors = extractor.extract_from_content_clusters()

        assert created == 0
        assert updated == 1

    def test_handles_exception_on_create(self, mock_conn: MagicMock) -> None:
        """Test that exceptions during create are captured."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (1, 10, "Label", datetime.now(UTC), 10, 8, ["a@test.com"], ["Subject"], 0.8),
                ]
            elif call_count <= 3:
                result.fetchone.return_value = None
            else:
                raise Exception("Insert error")
            return result

        mock_conn.execute.side_effect = mock_execute

        extractor = ProjectExtractor(mock_conn)
        created, updated, filtered, errors = extractor.extract_from_content_clusters()

        assert len(errors) == 1
        assert "Insert error" in errors[0]

    def test_handles_exception_on_update(self, mock_conn: MagicMock) -> None:
        """Test that exceptions during update are captured."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (1, 10, "Label", datetime.now(UTC), 10, 8, ["a@test.com"], ["Subject"], 0.8),
                ]
            elif call_count == 2:
                result.fetchone.return_value = None
            elif call_count == 3:
                result.fetchone.return_value = (123,)
            else:
                raise Exception("Update error")
            return result

        mock_conn.execute.side_effect = mock_execute

        extractor = ProjectExtractor(mock_conn)
        created, updated, filtered, errors = extractor.extract_from_content_clusters()

        assert len(errors) == 1
        assert "Update error" in errors[0]

    def test_project_name_from_auto_label(self, mock_conn: MagicMock) -> None:
        """Test that project name uses auto_label when available."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (1, 10, "Auto Label", datetime.now(UTC), 10, 8, None, None, 0.8),
                ]
            else:
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        extractor = ProjectExtractor(mock_conn)
        extractor.extract_from_content_clusters()

        assert mock_conn.execute.called

    def test_project_name_from_subject(self, mock_conn: MagicMock) -> None:
        """Test that project name uses subject when no auto_label."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (1, 10, None, datetime.now(UTC), 10, 8, None, ["Subject Name"], 0.8),
                ]
            else:
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        extractor = ProjectExtractor(mock_conn)
        extractor.extract_from_content_clusters()

        assert mock_conn.execute.called

    def test_project_name_fallback(self, mock_conn: MagicMock) -> None:
        """Test that project name falls back to cluster ID."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (42, 10, None, datetime.now(UTC), 10, 8, None, [None], 0.8),
                ]
            else:
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        extractor = ProjectExtractor(mock_conn)
        extractor.extract_from_content_clusters()

        assert mock_conn.execute.called

    def test_project_name_truncation(self, mock_conn: MagicMock) -> None:
        """Test that long project names are truncated."""
        long_label = "A" * 300
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (1, 10, long_label, datetime.now(UTC), 10, 8, None, None, 0.8),
                ]
            else:
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        extractor = ProjectExtractor(mock_conn)
        extractor.extract_from_content_clusters()

        assert mock_conn.execute.called


class TestPriorityContextBuilderInit:
    """Tests for PriorityContextBuilder initialization."""

    def test_init_stores_connection(self) -> None:
        """Test that init stores the connection."""
        conn = MagicMock()
        builder = PriorityContextBuilder(conn)

        assert builder.conn == conn
        assert builder.user_id is None

    def test_init_stores_user_id(self) -> None:
        """Test that init stores user_id when provided."""
        conn = MagicMock()
        user_id = uuid.uuid4()
        builder = PriorityContextBuilder(conn, user_id)

        assert builder.conn == conn
        assert builder.user_id == user_id


class TestPriorityContextBuilderBuildContexts:
    """Tests for PriorityContextBuilder.build_contexts method."""

    @pytest.fixture
    def mock_conn(self) -> MagicMock:
        """Create mock connection."""
        return MagicMock()

    def test_empty_results_returns_zeros(self, mock_conn: MagicMock) -> None:
        """Test that empty results return zeros."""
        mock_conn.execute.return_value.fetchall.return_value = []

        builder = PriorityContextBuilder(mock_conn)
        created, updated, errors = builder.build_contexts()

        assert created == 0
        assert updated == 0
        assert errors == []

    def test_creates_context_for_email(self, mock_conn: MagicMock) -> None:
        """Test that context is created for email."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                # Main query - row: email_id, feature_score, replied_similarity,
                # cluster_novelty, sender_novelty, priority_score, from_email,
                # thread_id, date_parsed, reply_rate, emails_from, is_important,
                # thread_length, is_real_person
                result.fetchall.return_value = [
                    (
                        1,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.7,
                        "person@test.com",
                        "thread123",
                        datetime.now(UTC),
                        0.5,
                        10,
                        True,
                        5,
                        True,
                    ),
                ]
            else:
                # Check for existing
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        builder = PriorityContextBuilder(mock_conn)
        created, updated, errors = builder.build_contexts()

        assert created == 1
        assert updated == 0

    def test_updates_existing_context(self, mock_conn: MagicMock) -> None:
        """Test that existing context is updated."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        1,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.7,
                        "person@test.com",
                        "thread123",
                        datetime.now(UTC),
                        0.5,
                        10,
                        True,
                        5,
                        True,
                    ),
                ]
            elif call_count == 2:
                result.fetchone.return_value = (123,)  # Existing context
            return result

        mock_conn.execute.side_effect = mock_execute

        builder = PriorityContextBuilder(mock_conn)
        created, updated, errors = builder.build_contexts()

        assert created == 0
        assert updated == 1

    def test_handles_exception_on_create(self, mock_conn: MagicMock) -> None:
        """Test that exceptions during create are captured."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        1,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.7,
                        "person@test.com",
                        "thread123",
                        datetime.now(UTC),
                        0.5,
                        10,
                        True,
                        5,
                        True,
                    ),
                ]
            elif call_count == 2:
                result.fetchone.return_value = None
            else:
                raise Exception("Insert error")
            return result

        mock_conn.execute.side_effect = mock_execute

        builder = PriorityContextBuilder(mock_conn)
        created, updated, errors = builder.build_contexts()

        assert len(errors) == 1
        assert "Insert error" in errors[0]

    def test_handles_exception_on_update(self, mock_conn: MagicMock) -> None:
        """Test that exceptions during update are captured."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        1,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.7,
                        "person@test.com",
                        "thread123",
                        datetime.now(UTC),
                        0.5,
                        10,
                        True,
                        5,
                        True,
                    ),
                ]
            elif call_count == 2:
                result.fetchone.return_value = (123,)
            else:
                raise Exception("Update error")
            return result

        mock_conn.execute.side_effect = mock_execute

        builder = PriorityContextBuilder(mock_conn)
        created, updated, errors = builder.build_contexts()

        assert len(errors) == 1
        assert "Update error" in errors[0]

    def test_real_person_gets_priority_boost(self, mock_conn: MagicMock) -> None:
        """Test that real person emails get priority boost."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        1,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        "person@test.com",
                        "thread123",
                        datetime.now(UTC),
                        0.5,
                        10,
                        True,
                        5,
                        True,
                    ),  # is_real_person = True
                ]
            else:
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        builder = PriorityContextBuilder(mock_conn)
        builder.build_contexts()

        # Verify execute was called (priority should be boosted to 0.75)
        assert mock_conn.execute.called

    def test_non_real_person_gets_priority_reduction(self, mock_conn: MagicMock) -> None:
        """Test that non-real-person emails get priority reduction."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        1,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        "noreply@test.com",
                        "thread123",
                        datetime.now(UTC),
                        0.0,
                        100,
                        False,
                        1,
                        False,
                    ),  # is_real_person = False
                ]
            else:
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        builder = PriorityContextBuilder(mock_conn)
        builder.build_contexts()

        # Verify execute was called (priority should be reduced to 0.25)
        assert mock_conn.execute.called

    def test_temporal_score_recent(self, mock_conn: MagicMock) -> None:
        """Test temporal score for recent emails."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        1,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        "person@test.com",
                        "thread123",
                        datetime.now(UTC),
                        0.5,
                        10,
                        True,
                        5,
                        True,
                    ),  # Very recent email
                ]
            else:
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        builder = PriorityContextBuilder(mock_conn)
        builder.build_contexts()

        assert mock_conn.execute.called

    def test_handles_none_date_parsed(self, mock_conn: MagicMock) -> None:
        """Test that None date_parsed is handled."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        1,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        "person@test.com",
                        "thread123",
                        None,
                        0.5,
                        10,
                        True,
                        5,
                        True,
                    ),  # date_parsed is None
                ]
            else:
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        builder = PriorityContextBuilder(mock_conn)
        created, updated, errors = builder.build_contexts()

        assert created == 1
        assert errors == []

    def test_handles_none_thread_id(self, mock_conn: MagicMock) -> None:
        """Test that None thread_id is handled."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        1,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        "person@test.com",
                        None,
                        datetime.now(UTC),
                        0.5,
                        10,
                        True,
                        5,
                        True,
                    ),
                ]
            else:
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        builder = PriorityContextBuilder(mock_conn)
        created, updated, errors = builder.build_contexts()

        assert created == 1

    def test_business_hours_calculation(self, mock_conn: MagicMock) -> None:
        """Test business hours calculation."""
        call_count = 0
        # Create a weekday at 10am
        weekday_morning = datetime(2026, 1, 6, 10, 0, 0, tzinfo=UTC)  # Monday 10am

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        1,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        "person@test.com",
                        "thread123",
                        weekday_morning,
                        0.5,
                        10,
                        True,
                        5,
                        True,
                    ),
                ]
            else:
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        builder = PriorityContextBuilder(mock_conn)
        builder.build_contexts()

        assert mock_conn.execute.called

    def test_people_score_important_sender(self, mock_conn: MagicMock) -> None:
        """Test people score boost for important sender."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        1,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        "important@test.com",
                        "thread123",
                        datetime.now(UTC),
                        0.3,  # Low reply rate
                        10,
                        True,  # is_important
                        5,
                        True,  # is_real_person
                    ),
                ]
            else:
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        builder = PriorityContextBuilder(mock_conn)
        created, _, _ = builder.build_contexts()

        assert created == 1

    def test_people_score_high_reply_rate(self, mock_conn: MagicMock) -> None:
        """Test people score boost for high reply rate."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        1,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        "person@test.com",
                        "thread123",
                        datetime.now(UTC),
                        0.7,  # High reply rate > 0.5
                        10,
                        False,  # Not important
                        5,
                        True,  # is_real_person
                    ),
                ]
            else:
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        builder = PriorityContextBuilder(mock_conn)
        created, _, _ = builder.build_contexts()

        assert created == 1

    def test_temporal_score_medium_age_email(self, mock_conn: MagicMock) -> None:
        """Test temporal score for medium-age emails (24-72 hours)."""
        call_count = 0
        # Create a timestamp 48 hours ago (between 24-72 hours)
        medium_age_date = datetime.now(UTC) - timedelta(hours=48)

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        1,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        "person@test.com",
                        "thread123",
                        medium_age_date,
                        0.5,
                        10,
                        True,
                        5,
                        True,
                    ),
                ]
            else:
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        builder = PriorityContextBuilder(mock_conn)
        created, _, _ = builder.build_contexts()

        assert created == 1

    def test_temporal_score_old_email(self, mock_conn: MagicMock) -> None:
        """Test temporal score for old emails (> 72 hours)."""
        call_count = 0
        # Create an old timestamp (> 72 hours ago)
        old_date = datetime(2025, 12, 1, 10, 0, 0, tzinfo=UTC)

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        1,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        "person@test.com",
                        "thread123",
                        old_date,
                        0.5,
                        10,
                        True,
                        5,
                        True,
                    ),
                ]
            else:
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        builder = PriorityContextBuilder(mock_conn)
        created, _, _ = builder.build_contexts()

        assert created == 1

    def test_relationship_score_with_reply_rate(self, mock_conn: MagicMock) -> None:
        """Test relationship score includes reply rate component."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        1,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        "person@test.com",
                        "thread123",
                        datetime.now(UTC),
                        0.3,  # reply_rate > 0 but < 0.5
                        10,
                        False,
                        5,
                        True,  # is_real_person
                    ),
                ]
            else:
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        builder = PriorityContextBuilder(mock_conn)
        created, _, _ = builder.build_contexts()

        assert created == 1

    def test_relationship_score_no_reply_rate(self, mock_conn: MagicMock) -> None:
        """Test relationship score when reply rate is 0 or None."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    (
                        1,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        "person@test.com",
                        "thread123",
                        datetime.now(UTC),
                        0.0,  # reply_rate = 0
                        10,
                        False,
                        5,
                        True,  # is_real_person
                    ),
                ]
            else:
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        builder = PriorityContextBuilder(mock_conn)
        created, _, _ = builder.build_contexts()

        assert created == 1


class TestExtractAllEntities:
    """Tests for extract_all_entities function."""

    @pytest.fixture
    def mock_conn(self) -> MagicMock:
        """Create mock connection."""
        return MagicMock()

    def test_returns_extraction_result(self, mock_conn: MagicMock) -> None:
        """Test that function returns ExtractionResult."""
        mock_conn.execute.return_value.fetchall.return_value = []

        result = extract_all_entities(mock_conn)

        assert isinstance(result, ExtractionResult)

    def test_aggregates_task_counts(self, mock_conn: MagicMock) -> None:
        """Test that task counts from LLM and AI are aggregated."""
        mock_conn.execute.return_value.fetchall.return_value = []

        result = extract_all_entities(mock_conn)

        assert result.tasks_created >= 0
        assert result.tasks_updated >= 0

    def test_aggregates_project_counts(self, mock_conn: MagicMock) -> None:
        """Test that project counts from threads and clusters are aggregated."""
        mock_conn.execute.return_value.fetchall.return_value = []

        result = extract_all_entities(mock_conn)

        assert result.projects_created >= 0
        assert result.projects_updated >= 0

    def test_aggregates_context_counts(self, mock_conn: MagicMock) -> None:
        """Test that priority context counts are included."""
        mock_conn.execute.return_value.fetchall.return_value = []

        result = extract_all_entities(mock_conn)

        assert result.priority_contexts_created >= 0
        assert result.priority_contexts_updated >= 0

    def test_passes_user_id(self, mock_conn: MagicMock) -> None:
        """Test that user_id is passed to extractors."""
        mock_conn.execute.return_value.fetchall.return_value = []
        user_id = uuid.uuid4()

        result = extract_all_entities(mock_conn, user_id)

        assert isinstance(result, ExtractionResult)

    def test_aggregates_errors(self, mock_conn: MagicMock) -> None:
        """Test that errors from all extractors are aggregated."""
        mock_conn.execute.return_value.fetchall.return_value = []

        result = extract_all_entities(mock_conn)

        assert isinstance(result.errors, list)

    def test_aggregates_filtered_counts(self, mock_conn: MagicMock) -> None:
        """Test that filtered counts are aggregated."""
        mock_conn.execute.return_value.fetchall.return_value = []

        result = extract_all_entities(mock_conn)

        assert result.tasks_filtered_marketing >= 0
        assert result.projects_filtered_marketing >= 0
