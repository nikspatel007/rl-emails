"""Tests for Stage 10: Compute priority rankings."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from rl_emails.core.config import Config
from rl_emails.pipeline.stages import stage_10_compute_priority
from rl_emails.pipeline.stages.base import StageResult


class TestCreateTables:
    """Tests for create_tables function."""

    def test_creates_tables(self) -> None:
        """Test table creation."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor

        stage_10_compute_priority.create_tables(conn)

        assert mock_cursor.execute.call_count == 2
        conn.commit.assert_called_once()


class TestComputeFeatureScores:
    """Tests for compute_feature_scores function."""

    def test_computes_person_scores(self) -> None:
        """Test feature score computation for person emails."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        # email_id, rel, urgency, is_person, service_imp
        mock_cursor.fetchall.return_value = [
            (1, 0.8, 0.6, 1, 0.5),  # Person email
        ]

        scores = stage_10_compute_priority.compute_feature_scores(conn)

        assert 1 in scores
        # Person: 0.8 * 0.6 + 0.6 * 0.4 = 0.48 + 0.24 = 0.72
        assert 0.7 <= scores[1] <= 0.75

    def test_computes_service_scores(self) -> None:
        """Test feature score computation for service emails."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            (1, 0.2, 0.6, 0, 0.8),  # Service email
        ]

        scores = stage_10_compute_priority.compute_feature_scores(conn)

        assert 1 in scores
        # Service: 0.8 * 0.5 + 0.6 * 0.5 = 0.4 + 0.3 = 0.7
        assert 0.65 <= scores[1] <= 0.75

    def test_empty_results(self) -> None:
        """Test with no results."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []

        scores = stage_10_compute_priority.compute_feature_scores(conn)

        assert scores == {}


class TestComputeRepliedSimilarity:
    """Tests for compute_replied_similarity function."""

    def test_no_centroid(self) -> None:
        """Test when no replied emails exist."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (None,)

        scores = stage_10_compute_priority.compute_replied_similarity(conn)

        assert scores == {}

    def test_with_centroid(self) -> None:
        """Test similarity computation with replied centroid."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = ("[0.1, 0.2]",)
        mock_cursor.fetchall.return_value = [
            (1, 0.8),
            (2, 0.4),
        ]

        scores = stage_10_compute_priority.compute_replied_similarity(conn)

        assert len(scores) == 2
        # Normalized: (0.8-0.4)/(0.8-0.4)=1, (0.4-0.4)/(0.8-0.4)=0
        assert scores[1] == 1.0
        assert scores[2] == 0.0

    def test_single_value(self) -> None:
        """Test normalization with single value."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = ("[0.1]",)
        mock_cursor.fetchall.return_value = [(1, 0.5)]

        scores = stage_10_compute_priority.compute_replied_similarity(conn)

        assert scores[1] == 0.0  # (0.5-0.5)/1 = 0

    def test_centroid_with_no_emails(self) -> None:
        """Test when centroid exists but no emails to compare."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = ("[0.1, 0.2]",)
        mock_cursor.fetchall.return_value = []  # No emails to compare

        scores = stage_10_compute_priority.compute_replied_similarity(conn)

        assert scores == {}


class TestComputeClusterNovelty:
    """Tests for compute_cluster_novelty function."""

    def test_computes_novelty(self) -> None:
        """Test cluster novelty computation."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            (1, 0.2),
            (2, 0.8),
        ]

        scores = stage_10_compute_priority.compute_cluster_novelty(conn)

        assert len(scores) == 2

    def test_empty_results(self) -> None:
        """Test with no results."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []

        scores = stage_10_compute_priority.compute_cluster_novelty(conn)

        assert scores == {}

    def test_filters_none(self) -> None:
        """Test None values are filtered."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            (1, 0.5),
            (2, None),
        ]

        scores = stage_10_compute_priority.compute_cluster_novelty(conn)

        assert len(scores) == 1
        assert 1 in scores


class TestComputeSenderNovelty:
    """Tests for compute_sender_novelty function."""

    def test_computes_novelty(self) -> None:
        """Test sender novelty computation."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            (1, 0.3),
            (2, 0.7),
        ]

        scores = stage_10_compute_priority.compute_sender_novelty(conn)

        assert len(scores) == 2


class TestComputePriorityScores:
    """Tests for compute_priority_scores function."""

    def test_combines_scores(self) -> None:
        """Test score combination."""
        feature_scores = {1: 0.8, 2: 0.4}
        replied_similarity = {1: 0.6}
        cluster_novelty = {1: 0.5}
        sender_novelty = {1: 0.3}

        results = stage_10_compute_priority.compute_priority_scores(
            feature_scores, replied_similarity, cluster_novelty, sender_novelty
        )

        assert 1 in results
        assert 2 in results
        assert "priority_score" in results[1]
        assert results[1]["feature_score"] == 0.8

    def test_uses_defaults(self) -> None:
        """Test default values for missing scores."""
        feature_scores = {1: 0.8}
        replied_similarity: dict[int, float] = {}
        cluster_novelty: dict[int, float] = {}
        sender_novelty: dict[int, float] = {}

        results = stage_10_compute_priority.compute_priority_scores(
            feature_scores, replied_similarity, cluster_novelty, sender_novelty
        )

        assert results[1]["replied_similarity"] == 0.5
        assert results[1]["cluster_novelty"] == 0.5
        assert results[1]["sender_novelty"] == 0.5


class TestDetermineLlmFlags:
    """Tests for determine_llm_flags function."""

    def test_high_priority(self) -> None:
        """Test high priority flag."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            (1, None, False, 0),
        ]

        priorities: dict[int, dict[str, Any]] = {1: {"priority_score": 0.8}}

        flags = stage_10_compute_priority.determine_llm_flags(conn, priorities)

        assert flags[1] == (True, "high_priority")

    def test_replied_training(self) -> None:
        """Test replied training flag."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            (1, "REPLIED", False, 0),
        ]

        priorities: dict[int, dict[str, Any]] = {1: {"priority_score": 0.3}}

        flags = stage_10_compute_priority.determine_llm_flags(conn, priorities)

        assert flags[1] == (True, "replied_training")

    def test_high_priority_ignored(self) -> None:
        """Test high priority ignored flag."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            (1, "IGNORED", False, 0),
        ]

        priorities: dict[int, dict[str, Any]] = {1: {"priority_score": 0.55}}

        flags = stage_10_compute_priority.determine_llm_flags(conn, priorities)

        assert flags[1] == (True, "high_priority_ignored")

    def test_novel_content(self) -> None:
        """Test novel content flag."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            (1, None, False, -1),  # content_cluster = -1 (noise)
        ]

        priorities: dict[int, dict[str, Any]] = {1: {"priority_score": 0.45}}

        flags = stage_10_compute_priority.determine_llm_flags(conn, priorities)

        assert flags[1] == (True, "novel_content")

    def test_no_flag(self) -> None:
        """Test no flag needed."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            (1, "ARCHIVED", False, 0),
        ]

        priorities: dict[int, dict[str, Any]] = {1: {"priority_score": 0.2}}

        flags = stage_10_compute_priority.determine_llm_flags(conn, priorities)

        assert flags[1] == (False, None)

    def test_missing_meta(self) -> None:
        """Test email not in metadata."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []

        priorities: dict[int, dict[str, Any]] = {1: {"priority_score": 0.2}}

        flags = stage_10_compute_priority.determine_llm_flags(conn, priorities)

        assert flags[1] == (False, None)


class TestSavePriorities:
    """Tests for save_priorities function."""

    @patch("rl_emails.pipeline.stages.stage_10_compute_priority.execute_values")
    def test_saves_priorities(self, mock_execute_values: MagicMock) -> None:
        """Test saving priorities."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor

        priorities: dict[int, dict[str, float]] = {
            1: {
                "feature_score": 0.8,
                "replied_similarity": 0.6,
                "cluster_novelty": 0.5,
                "sender_novelty": 0.4,
                "priority_score": 0.7,
            }
        }
        llm_flags: dict[int, tuple[bool, str | None]] = {1: (True, "high_priority")}

        result = stage_10_compute_priority.save_priorities(conn, priorities, llm_flags)

        assert result == 1
        mock_cursor.execute.assert_called()
        mock_execute_values.assert_called_once()
        conn.commit.assert_called_once()

    @patch("rl_emails.pipeline.stages.stage_10_compute_priority.execute_values")
    def test_default_llm_flags(self, mock_execute_values: MagicMock) -> None:
        """Test default LLM flags for missing entries."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor

        priorities: dict[int, dict[str, float]] = {
            1: {
                "feature_score": 0.5,
                "replied_similarity": 0.5,
                "cluster_novelty": 0.5,
                "sender_novelty": 0.5,
                "priority_score": 0.5,
            }
        }
        llm_flags: dict[int, tuple[bool, str | None]] = {}  # Empty

        result = stage_10_compute_priority.save_priorities(conn, priorities, llm_flags)

        assert result == 1


class TestRun:
    """Tests for run function."""

    @patch("rl_emails.pipeline.stages.stage_10_compute_priority.psycopg2.connect")
    def test_run_success(self, mock_connect: MagicMock) -> None:
        """Test successful run."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Mock feature scores query
        mock_cursor.fetchall.side_effect = [
            [(1, 0.5, 0.5, 1, 0.5)],  # compute_feature_scores
            [],  # compute_replied_similarity (second query)
            [],  # compute_cluster_novelty
            [],  # compute_sender_novelty
            [(1, None, False, 0)],  # determine_llm_flags
        ]
        mock_cursor.fetchone.return_value = (None,)  # No replied centroid

        with patch("rl_emails.pipeline.stages.stage_10_compute_priority.execute_values"):
            config = Config(database_url="postgresql://test")
            result = stage_10_compute_priority.run(config)

        assert isinstance(result, StageResult)
        assert result.success is True
        mock_conn.close.assert_called_once()
