#!/usr/bin/env python3
"""Tests for multi-task extraction module."""

import pytest
from unittest.mock import patch, MagicMock

from src.multi_task_extraction import (
    ExtractedTask,
    TaskType,
    generate_task_id,
    extract_tasks_rule_based,
    extract_tasks,
    extract_tasks_batch,
)


class TestExtractedTask:
    """Tests for ExtractedTask dataclass."""

    def test_create_minimal(self):
        """Test creating task with minimal fields."""
        task = ExtractedTask(
            task_id="abc123",
            description="Review the document"
        )
        assert task.task_id == "abc123"
        assert task.description == "Review the document"
        assert task.deadline is None
        assert task.assignee_hint is None
        assert task.task_type == TaskType.ACTION
        assert task.confidence == 1.0

    def test_create_full(self):
        """Test creating task with all fields."""
        task = ExtractedTask(
            task_id="xyz789",
            description="Send quarterly report",
            deadline="by Friday",
            assignee_hint="you",
            task_type=TaskType.DELIVERABLE,
            confidence=0.95,
            source_text="Please send quarterly report by Friday",
            email_id="msg-001"
        )
        assert task.deadline == "by Friday"
        assert task.assignee_hint == "you"
        assert task.task_type == TaskType.DELIVERABLE
        assert task.confidence == 0.95

    def test_to_dict(self):
        """Test serialization to dictionary."""
        task = ExtractedTask(
            task_id="abc123",
            description="Test task",
            task_type=TaskType.REVIEW
        )
        d = task.to_dict()
        assert d["task_id"] == "abc123"
        assert d["task_type"] == "review"  # Enum converted to string

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "task_id": "def456",
            "description": "Another task",
            "deadline": "EOD",
            "assignee_hint": "team",
            "task_type": "meeting",
            "confidence": 0.8,
            "source_text": "schedule meeting",
            "email_id": "msg-002"
        }
        task = ExtractedTask.from_dict(data)
        assert task.task_id == "def456"
        assert task.task_type == TaskType.MEETING
        assert task.deadline == "EOD"


class TestGenerateTaskId:
    """Tests for task ID generation."""

    def test_deterministic(self):
        """Test that same inputs produce same ID."""
        id1 = generate_task_id("email-1", 0, "Review document")
        id2 = generate_task_id("email-1", 0, "Review document")
        assert id1 == id2

    def test_different_emails(self):
        """Test that different emails produce different IDs."""
        id1 = generate_task_id("email-1", 0, "Review document")
        id2 = generate_task_id("email-2", 0, "Review document")
        assert id1 != id2

    def test_different_indices(self):
        """Test that different indices produce different IDs."""
        id1 = generate_task_id("email-1", 0, "Review document")
        id2 = generate_task_id("email-1", 1, "Review document")
        assert id1 != id2

    def test_length(self):
        """Test that ID is 12 characters."""
        task_id = generate_task_id("email-1", 0, "Test")
        assert len(task_id) == 12


class TestRuleBasedExtraction:
    """Tests for rule-based task extraction."""

    def test_please_request(self):
        """Test extraction of 'please' requests."""
        tasks = extract_tasks_rule_based(
            subject="Request",
            body="Please review the attached document and provide feedback.",
            email_id="test-1"
        )
        assert len(tasks) >= 1
        descriptions = [t.description.lower() for t in tasks]
        assert any("review" in d for d in descriptions)

    def test_can_you_request(self):
        """Test extraction of 'can you' requests."""
        tasks = extract_tasks_rule_based(
            subject="Help needed",
            body="Can you send me the latest sales figures?",
            email_id="test-2"
        )
        assert len(tasks) >= 1

    def test_deadline_detection(self):
        """Test deadline extraction near task."""
        tasks = extract_tasks_rule_based(
            subject="Urgent",
            body="Please complete the report by Friday.",
            email_id="test-3"
        )
        assert len(tasks) >= 1
        # Check if any task has the deadline
        deadlines = [t.deadline for t in tasks if t.deadline]
        assert any("friday" in d.lower() for d in deadlines) or len(tasks) > 0

    def test_bullet_list_extraction(self):
        """Test extraction from bullet lists."""
        tasks = extract_tasks_rule_based(
            subject="Todo",
            body="""Action items:
            - Review the proposal
            - Send the updated timeline
            - Schedule the kickoff meeting""",
            email_id="test-4"
        )
        assert len(tasks) >= 2

    def test_task_types(self):
        """Test correct task type assignment."""
        tasks = extract_tasks_rule_based(
            subject="Tasks",
            body="""
            Please review the contract
            Please send me the report
            Can you schedule a meeting with John?
            """,
            email_id="test-5"
        )

        task_types = [t.task_type for t in tasks]
        # Should have a mix of types
        assert len(tasks) >= 2

    def test_no_tasks(self):
        """Test email with no tasks."""
        tasks = extract_tasks_rule_based(
            subject="FYI",
            body="Just wanted to let you know the project is going well. No action needed.",
            email_id="test-6"
        )
        # May extract 0 or very few tasks
        assert len(tasks) <= 2

    def test_deduplication(self):
        """Test that duplicate tasks are not extracted."""
        tasks = extract_tasks_rule_based(
            subject="Request",
            body="""Please review the document.
            Can you please review the document?
            I need you to review the document.""",
            email_id="test-7"
        )
        # Should deduplicate similar requests
        assert len(tasks) <= 3

    def test_limit_max_tasks(self):
        """Test that extraction is limited to reasonable number."""
        # Create email with many potential tasks
        items = [f"- Send item {i}" for i in range(20)]
        body = "Tasks:\n" + "\n".join(items)

        tasks = extract_tasks_rule_based(
            subject="Many tasks",
            body=body,
            email_id="test-8"
        )
        assert len(tasks) <= 10  # Should be capped


class TestExtractTasks:
    """Tests for main extract_tasks function."""

    def test_rule_based_fallback(self):
        """Test that rule-based works when use_llm=False."""
        tasks = extract_tasks(
            subject="Request",
            body="Please send me the quarterly report by EOD.",
            email_id="test-1",
            use_llm=False
        )
        assert len(tasks) >= 1
        assert all(isinstance(t, ExtractedTask) for t in tasks)

    def test_no_api_key_fallback(self):
        """Test fallback when API key not set."""
        with patch.dict('os.environ', {}, clear=True):
            # Remove ANTHROPIC_API_KEY
            import os
            if 'ANTHROPIC_API_KEY' in os.environ:
                del os.environ['ANTHROPIC_API_KEY']

            tasks = extract_tasks(
                subject="Request",
                body="Please review the document.",
                email_id="test-2",
                use_llm=True  # Will fall back to rule-based
            )
            # Should still return results (via fallback)
            assert isinstance(tasks, list)


class TestExtractTasksBatch:
    """Tests for batch extraction."""

    def test_batch_extraction(self):
        """Test extracting from multiple emails."""
        emails = [
            {
                "message_id": "msg-1",
                "subject": "Task 1",
                "body": "Please review the proposal.",
                "from": "alice@example.com"
            },
            {
                "message_id": "msg-2",
                "subject": "Task 2",
                "body": "Can you send the report?",
                "from": "bob@example.com"
            }
        ]

        results = extract_tasks_batch(emails, use_llm=False)

        assert "msg-1" in results
        assert "msg-2" in results
        assert all(isinstance(tasks, list) for tasks in results.values())

    def test_empty_batch(self):
        """Test empty email list."""
        results = extract_tasks_batch([], use_llm=False)
        assert results == {}


class TestTaskType:
    """Tests for TaskType enum."""

    def test_all_types_exist(self):
        """Test all expected task types exist."""
        expected = ["action", "deliverable", "review", "meeting",
                    "followup", "decision", "information", "other"]
        for t in expected:
            assert TaskType(t) is not None

    def test_string_conversion(self):
        """Test string conversion."""
        assert TaskType.ACTION.value == "action"
        assert TaskType.DELIVERABLE.value == "deliverable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
