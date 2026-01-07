"""Tests for pipeline stages base module."""

from __future__ import annotations

from rl_emails.pipeline.stages.base import StageResult


class TestStageResult:
    """Tests for StageResult dataclass."""

    def test_create_successful_result(self) -> None:
        """Test creating a successful stage result."""
        result = StageResult(
            success=True,
            records_processed=100,
            duration_seconds=5.5,
            message="Stage completed successfully",
        )

        assert result.success is True
        assert result.records_processed == 100
        assert result.duration_seconds == 5.5
        assert result.message == "Stage completed successfully"
        assert result.metadata is None

    def test_create_failed_result(self) -> None:
        """Test creating a failed stage result."""
        result = StageResult(
            success=False,
            records_processed=50,
            duration_seconds=2.3,
            message="Stage failed due to database error",
        )

        assert result.success is False
        assert result.records_processed == 50
        assert result.duration_seconds == 2.3
        assert result.message == "Stage failed due to database error"

    def test_create_result_with_metadata(self) -> None:
        """Test creating a result with metadata."""
        metadata = {"errors": 5, "warnings": ["Low memory", "Slow network"]}
        result = StageResult(
            success=True,
            records_processed=1000,
            duration_seconds=10.0,
            message="Completed with warnings",
            metadata=metadata,
        )

        assert result.metadata == metadata
        assert result.metadata["errors"] == 5
        assert len(result.metadata["warnings"]) == 2

    def test_str_success(self) -> None:
        """Test string representation for successful result."""
        result = StageResult(
            success=True,
            records_processed=1000,
            duration_seconds=5.5,
            message="Import complete",
        )

        result_str = str(result)
        assert "SUCCESS" in result_str
        assert "Import complete" in result_str
        assert "1,000 records" in result_str
        assert "5.5s" in result_str

    def test_str_failure(self) -> None:
        """Test string representation for failed result."""
        result = StageResult(
            success=False,
            records_processed=0,
            duration_seconds=1.2,
            message="Connection failed",
        )

        result_str = str(result)
        assert "FAILED" in result_str
        assert "Connection failed" in result_str

    def test_default_metadata_is_none(self) -> None:
        """Test that metadata defaults to None."""
        result = StageResult(
            success=True,
            records_processed=0,
            duration_seconds=0.1,
            message="No records",
        )

        assert result.metadata is None

    def test_zero_records(self) -> None:
        """Test result with zero records processed."""
        result = StageResult(
            success=True,
            records_processed=0,
            duration_seconds=0.01,
            message="No data to process",
        )

        assert result.success is True
        assert result.records_processed == 0
        assert "0 records" in str(result)
