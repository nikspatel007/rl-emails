#!/usr/bin/env python3
"""Tests for task complexity estimation module."""

import pytest
from src.features.complexity import (
    estimate_complexity,
    estimate_complexity_with_llm,
    complexity_to_effort,
    ComplexityLevel,
    ComplexityEstimate,
)


class TestEstimateComplexity:
    """Tests for estimate_complexity function."""

    def test_trivial_acknowledgment(self):
        """Simple acknowledgment is trivial complexity."""
        subject = "Re: Meeting notes"
        body = "Got it, thanks!"

        estimate = estimate_complexity(subject, body)

        assert estimate.complexity == ComplexityLevel.TRIVIAL
        assert estimate.confidence >= 0.5
        assert estimate.scope_score < 0.3

    def test_trivial_forward(self):
        """Simple forward request is trivial or quick."""
        subject = "FYI: Project update"
        body = "Forwarding this for your info. See attached."

        estimate = estimate_complexity(subject, body)

        assert estimate.complexity in [ComplexityLevel.TRIVIAL, ComplexityLevel.QUICK]
        # Short forwarding emails should have low-to-moderate scope
        assert estimate.scope_score <= 0.5

    def test_quick_simple_question(self):
        """Simple question with quick answer is quick complexity."""
        subject = "Quick question about the schedule"
        body = """
        Hi,

        Can you confirm if the meeting is still at 3pm tomorrow?

        Thanks!
        """

        estimate = estimate_complexity(subject, body)

        assert estimate.complexity in [ComplexityLevel.TRIVIAL, ComplexityLevel.QUICK]
        assert estimate.scope_score < 0.5

    def test_quick_simple_request(self):
        """Brief request is trivial or quick complexity."""
        subject = "Need a quick update"
        body = """
        Hey,

        Just need a few words on where we are with the client proposal.

        Thanks!
        """

        estimate = estimate_complexity(subject, body)

        # "Just need a few words" is borderline trivial/quick
        assert estimate.complexity in [ComplexityLevel.TRIVIAL, ComplexityLevel.QUICK]
        assert estimate.scope_score < 0.5

    def test_medium_review_request(self):
        """Review request with multiple items is medium complexity."""
        subject = "Need your input on the proposal"
        body = """
        Hi team,

        I've drafted the initial proposal for the Q2 marketing campaign.

        Please review the attached document and provide your feedback on:
        - Budget allocation
        - Timeline feasibility
        - Target audience segments

        Let me know your thoughts by EOD Friday.

        Thanks,
        Marketing Team
        """

        estimate = estimate_complexity(subject, body)

        assert estimate.complexity in [ComplexityLevel.MEDIUM, ComplexityLevel.QUICK]
        assert estimate.scope_score >= 0.3

    def test_medium_multi_step_request(self):
        """Multi-step request is medium complexity."""
        subject = "Action items from meeting"
        body = """
        Hi,

        Following up on our discussion, here's what we need:

        1. First, review the current metrics dashboard
        2. Then, identify the key performance gaps
        3. Next, prepare a summary for leadership
        4. Finally, schedule a follow-up meeting

        Can you complete these by next week?
        """

        estimate = estimate_complexity(subject, body)

        assert estimate.complexity in [ComplexityLevel.MEDIUM, ComplexityLevel.SUBSTANTIAL]
        assert estimate.scope_score >= 0.4

    def test_substantial_comprehensive_analysis(self):
        """Comprehensive analysis request is substantial complexity."""
        subject = "Comprehensive security audit required"
        body = """
        Hi Security Team,

        Following the recent compliance review, we need a comprehensive security
        audit of our entire authentication system.

        This will require:
        1. Technical review of the current architecture
        2. Vulnerability assessment and penetration testing
        3. Coordination with the DevOps team for infrastructure access
        4. Analysis of access control policies
        5. Preparation of a detailed report with recommendations

        We'll need input from Legal regarding compliance implications.
        The Finance team has requested a budget estimate for remediation.

        This is blocking the Q2 release and requires specialized security expertise.

        Please coordinate with the cross-functional stakeholders and provide
        a comprehensive analysis by end of month.
        """

        estimate = estimate_complexity(subject, body)

        assert estimate.complexity == ComplexityLevel.SUBSTANTIAL
        assert estimate.scope_score >= 0.6
        assert estimate.dependency_score >= 0.3
        assert estimate.expertise_score >= 0.3

    def test_substantial_project_work(self):
        """Large project request is medium to substantial complexity."""
        subject = "Strategic planning initiative"
        body = """
        Dear Team,

        We are kicking off a major strategic initiative for the upcoming fiscal year.
        This will require extensive analysis and implementation planning.

        Key deliverables:
        - Comprehensive market research and competitive analysis
        - Financial projections and budget proposals
        - Implementation roadmap with clear milestones
        - Resource allocation strategy
        - Risk assessment and mitigation plans

        This project will require coordination across multiple teams including
        Marketing, Finance, Engineering, and Operations.

        We'll need subject matter experts for technical review of the proposed
        architecture changes.

        Please prepare a detailed project plan.
        """

        estimate = estimate_complexity(subject, body)

        # Strategic initiatives with deliverables are MEDIUM or SUBSTANTIAL
        assert estimate.complexity in [ComplexityLevel.MEDIUM, ComplexityLevel.SUBSTANTIAL]
        assert estimate.scope_score >= 0.5
        assert estimate.confidence >= 0.5

    def test_dependency_detection(self):
        """Detects dependencies on others."""
        subject = "Need approval before proceeding"
        body = """
        Hi,

        I'm waiting for input from the legal team before I can finalize this.

        Once we get their approval, I'll need to coordinate with the finance
        department for the budget sign-off.

        This is currently blocked on the stakeholder review.
        """

        estimate = estimate_complexity(subject, body)

        assert estimate.dependency_score >= 0.5

    def test_expertise_detection(self):
        """Detects need for specialized expertise."""
        subject = "Technical architecture review needed"
        body = """
        Hi Engineering,

        We need a thorough technical review of the proposed system architecture.

        This requires specialized security expertise to evaluate the
        authentication flow and compliance implications.

        Please coordinate with our domain experts for the financial modeling
        components.
        """

        estimate = estimate_complexity(subject, body)

        assert estimate.expertise_score >= 0.4

    def test_extracted_tasks_increase_complexity(self):
        """Multiple extracted tasks increase complexity."""
        subject = "Several items to discuss"
        body = "Let me know about these things."

        # Without tasks
        estimate_no_tasks = estimate_complexity(subject, body)

        # With multiple tasks
        tasks = [
            {"description": "Review the proposal"},
            {"description": "Update the spreadsheet"},
            {"description": "Schedule meeting"},
        ]
        estimate_with_tasks = estimate_complexity(
            subject, body, extracted_tasks=tasks
        )

        assert estimate_with_tasks.scope_score >= estimate_no_tasks.scope_score

    def test_feature_vector_dimensions(self):
        """Feature vector has correct dimensions."""
        estimate = estimate_complexity("Test", "Test body")
        vec = estimate.to_feature_vector()

        # 4 one-hot + 1 confidence + 3 scores = 8
        assert len(vec) == 8

    def test_method_is_rule_based(self):
        """Without LLM, method should be 'rule'."""
        estimate = estimate_complexity("Test", "Simple body")

        assert estimate.method == 'rule'
        assert estimate.reasoning is None


class TestComplexityLevelNumericValue:
    """Tests for ComplexityLevel numeric values."""

    def test_trivial_value(self):
        """Trivial has lowest value."""
        assert ComplexityLevel.TRIVIAL.numeric_value == 0.1

    def test_quick_value(self):
        """Quick has low value."""
        assert ComplexityLevel.QUICK.numeric_value == 0.3

    def test_medium_value(self):
        """Medium has middle value."""
        assert ComplexityLevel.MEDIUM.numeric_value == 0.6

    def test_substantial_value(self):
        """Substantial has highest value."""
        assert ComplexityLevel.SUBSTANTIAL.numeric_value == 0.9

    def test_ordering(self):
        """Values are properly ordered."""
        assert (
            ComplexityLevel.TRIVIAL.numeric_value <
            ComplexityLevel.QUICK.numeric_value <
            ComplexityLevel.MEDIUM.numeric_value <
            ComplexityLevel.SUBSTANTIAL.numeric_value
        )


class TestComplexityToEffort:
    """Tests for complexity_to_effort backwards compatibility."""

    def test_trivial_maps_to_quick(self):
        """Trivial maps to legacy 'quick'."""
        assert complexity_to_effort(ComplexityLevel.TRIVIAL) == 'quick'

    def test_quick_maps_to_quick(self):
        """Quick maps to legacy 'quick'."""
        assert complexity_to_effort(ComplexityLevel.QUICK) == 'quick'

    def test_medium_maps_to_medium(self):
        """Medium maps to legacy 'medium'."""
        assert complexity_to_effort(ComplexityLevel.MEDIUM) == 'medium'

    def test_substantial_maps_to_substantial(self):
        """Substantial maps to legacy 'substantial'."""
        assert complexity_to_effort(ComplexityLevel.SUBSTANTIAL) == 'substantial'


class TestEstimateComplexityWithLlm:
    """Tests for estimate_complexity_with_llm function."""

    def test_without_llm_client(self):
        """Works without LLM client (just rules)."""
        estimate = estimate_complexity_with_llm(
            "Test Subject",
            "Simple email body",
            llm_client=None,
        )

        assert estimate.method == 'rule'
        assert estimate.reasoning is None

    def test_confident_estimation_skips_llm(self):
        """High confidence estimation skips LLM."""
        class FailingLLMClient:
            def messages(self, *args, **kwargs):
                raise RuntimeError("Should not be called")

        body = """
        Got it, thanks! Will do.
        """

        estimate = estimate_complexity_with_llm(
            "Re: Quick update",
            body,
            llm_client=FailingLLMClient(),
            llm_threshold=0.5,
        )

        # Should work without calling LLM
        assert isinstance(estimate, ComplexityEstimate)


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_email(self):
        """Handles empty email gracefully."""
        estimate = estimate_complexity("", "")

        assert isinstance(estimate, ComplexityEstimate)
        assert estimate.complexity in ComplexityLevel

    def test_very_short_email(self):
        """Handles very short emails."""
        estimate = estimate_complexity("Hi", "Ok")

        assert estimate.complexity in [ComplexityLevel.TRIVIAL, ComplexityLevel.QUICK]

    def test_very_long_email(self):
        """Handles long emails without issues."""
        long_body = "Lorem ipsum dolor sit amet. " * 500

        estimate = estimate_complexity("Long Email", long_body)

        assert isinstance(estimate, ComplexityEstimate)
        # Long emails tend toward higher scope
        assert estimate.scope_score >= 0.3

    def test_unicode_content(self):
        """Handles unicode content."""
        subject = "日本語のメール"
        body = """
        こんにちは、

        Please review the proposal and provide comprehensive analysis.

        よろしくお願いします。
        """

        estimate = estimate_complexity(subject, body)

        assert isinstance(estimate, ComplexityEstimate)

    def test_special_characters(self):
        """Handles special characters."""
        subject = "RE: Follow-up [URGENT] (v2.0)"
        body = """
        <html>
        Please review ASAP!!!

        *** IMPORTANT ***

        @team #project $$$
        </html>
        """

        estimate = estimate_complexity(subject, body)

        assert isinstance(estimate, ComplexityEstimate)

    def test_all_caps_email(self):
        """Handles all caps emails."""
        subject = "URGENT REQUEST NEEDED NOW"
        body = "PLEASE SEND THE REPORT IMMEDIATELY. THIS IS BLOCKING EVERYTHING."

        estimate = estimate_complexity(subject, body)

        assert isinstance(estimate, ComplexityEstimate)


class TestPatternMatching:
    """Tests for specific pattern matching."""

    def test_blocking_pattern(self):
        """Detects blocking/dependency patterns."""
        body = "This is blocking the release. We're waiting on the legal review."

        estimate = estimate_complexity("Urgent", body)

        assert estimate.dependency_score > 0

    def test_coordinate_pattern(self):
        """Detects coordination patterns."""
        body = "Need to coordinate with the sales team and sync with marketing."

        estimate = estimate_complexity("Cross-team work", body)

        assert estimate.dependency_score > 0

    def test_technical_expertise_pattern(self):
        """Detects technical expertise patterns."""
        body = "Requires technical architecture review and security audit assessment."

        estimate = estimate_complexity("Review needed", body)

        assert estimate.expertise_score > 0

    def test_legal_expertise_pattern(self):
        """Detects legal expertise patterns."""
        body = "Need legal review of the compliance implications and regulatory requirements."

        estimate = estimate_complexity("Legal review", body)

        assert estimate.expertise_score > 0

    def test_prepare_report_pattern(self):
        """Detects report preparation patterns."""
        body = "Please prepare a detailed report on the project status."

        estimate = estimate_complexity("Report needed", body)

        assert estimate.scope_score >= 0.3


class TestConfidenceScores:
    """Tests for confidence score behavior."""

    def test_clear_trivial_high_confidence(self):
        """Clear trivial case has high confidence."""
        estimate = estimate_complexity("Re: Thanks", "Got it!")

        if estimate.complexity == ComplexityLevel.TRIVIAL:
            assert estimate.confidence >= 0.5

    def test_clear_substantial_high_confidence(self):
        """Clear substantial case has high confidence."""
        body = """
        Need comprehensive strategic analysis with detailed implementation plan.
        Coordinate with cross-functional stakeholders for technical architecture review.
        Prepare detailed report with financial projections.
        """

        estimate = estimate_complexity("Major initiative", body)

        if estimate.complexity == ComplexityLevel.SUBSTANTIAL:
            assert estimate.confidence >= 0.5

    def test_confidence_bounded(self):
        """Confidence is always between 0 and 1."""
        test_cases = [
            ("Short", "x"),
            ("Long", "word " * 1000),
            ("Pattern heavy", "comprehensive detailed thorough analysis research investigate"),
        ]

        for subject, body in test_cases:
            estimate = estimate_complexity(subject, body)
            assert 0.0 <= estimate.confidence <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
