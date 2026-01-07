"""Base classes and types for pipeline stages.

This module provides the core types used by all pipeline stages:
- StageResult: The result returned by each stage's run() function
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageResult:
    """Result returned by each pipeline stage.

    Attributes:
        success: Whether the stage completed successfully.
        records_processed: Number of records processed by this stage.
        duration_seconds: Time taken to execute the stage.
        message: Human-readable status message.
        metadata: Optional additional data about the execution.
    """

    success: bool
    records_processed: int
    duration_seconds: float
    message: str
    metadata: dict[str, Any] | None = field(default=None)

    def __str__(self) -> str:
        """Return human-readable string representation."""
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"{status}: {self.message} "
            f"({self.records_processed:,} records in {self.duration_seconds:.1f}s)"
        )
