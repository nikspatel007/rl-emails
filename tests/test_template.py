#!/usr/bin/env python3
"""Tests for template pattern detection module."""

import pytest
from src.features.template import (
    detect_template,
    detect_template_with_llm,
    compute_template_score,
    TemplateFeatures,
)


class TestDetectTemplate:
    """Tests for detect_template function."""

    def test_shipping_notification(self):
        """Detects shipping notification as transactional template."""
        subject = "Your order has shipped!"
        body = """
        Dear {{customer_name}},

        Great news! Your order #12345 has been shipped.
        Tracking Number: 1Z999AA10123456784
        Estimated Delivery: January 8, 2026

        Track your package: https://example.com/track/12345

        ---

        You're receiving this email because you made a purchase at Example Store.

        To unsubscribe from shipping notifications, click here.
        © 2026 Example Store. All rights reserved.
        """

        features = detect_template(subject, body)

        assert features.is_templated is True
        assert features.template_type == 'transactional'
        assert features.template_confidence >= 0.5
        assert 'shipping_notification' in features.detected_services
        assert features.placeholder_count >= 1
        assert features.unsubscribe_signals >= 1

    def test_password_reset(self):
        """Detects password reset as transactional template."""
        subject = "Reset your password"
        body = """
        Hi,

        We received a request to reset your password. Click the link below
        to reset your password:

        https://example.com/reset?token=abc123

        If you didn't request this, please ignore this email.

        This link will expire in 24 hours.

        Thanks,
        The Example Team
        """

        features = detect_template(subject, body)

        assert features.is_templated is True
        assert features.template_type == 'transactional'
        assert 'password_reset' in features.detected_services

    def test_marketing_email(self):
        """Detects marketing email with promotional content."""
        subject = "50% OFF - Limited Time Offer!"
        body = """
        Hi there!

        Don't miss our biggest sale of the year!

        Use code SAVE50 at checkout for 50% off your entire order.

        Shop Now: https://example.com/sale

        ---

        This email was sent to you because you subscribed to our newsletter.

        To unsubscribe or manage your email preferences, click here.
        Forward this email to a friend.

        © 2026 Example Store | Privacy Policy | Terms of Service
        """

        features = detect_template(subject, body)

        assert features.is_templated is True
        assert features.template_type in ['marketing', 'newsletter']
        assert features.marketing_footer_signals >= 2
        assert features.unsubscribe_signals >= 1

    def test_newsletter(self):
        """Detects newsletter format."""
        subject = "Weekly Digest - Top Stories from This Week"
        body = """
        This Week's Top Stories

        1. Feature Article: AI in Healthcare
        2. Tech News: Latest Product Announcements
        3. Opinion: The Future of Remote Work

        ---

        Read more at our website.

        You're receiving this because you signed up for our newsletter.
        Unsubscribe | View in browser

        Powered by MailingList Pro
        """

        features = detect_template(subject, body)

        assert features.is_templated is True
        assert features.template_type in ['newsletter', 'marketing']
        assert features.unsubscribe_signals >= 1

    def test_regular_email_not_templated(self):
        """Personal email should not be detected as templated."""
        subject = "Quick question about the project"
        body = """
        Hi Sarah,

        Just wanted to follow up on our conversation yesterday.
        Do you have time to meet this week to discuss the proposal?

        Let me know what works for you.

        Best,
        John
        """

        features = detect_template(subject, body)

        assert features.is_templated is False
        assert features.template_type == 'none'
        assert features.template_confidence < 0.3
        assert features.placeholder_count == 0

    def test_placeholder_detection(self):
        """Detects various placeholder formats."""
        body = """
        Hello {{name}},

        Your order ${order_id} has been confirmed.
        Delivery to %CUSTOMER_ADDRESS%.

        Reference: [[ref_number]]
        """

        features = detect_template("Order Confirmation", body)

        assert features.placeholder_count >= 3
        assert features.is_templated is True

    def test_order_confirmation(self):
        """Detects order confirmation emails."""
        subject = "Thank you for your purchase!"
        body = """
        Order Confirmation

        Thank you for your order #ORD-98765

        Items:
        - Widget A x 2
        - Gadget B x 1

        Total: $45.99

        Your order will be processed within 1-2 business days.

        Receipt for order #ORD-98765 attached.
        """

        features = detect_template(subject, body)

        assert features.is_templated is True
        assert 'order_confirmation' in features.detected_services

    def test_social_notification(self):
        """Detects social media notification emails."""
        subject = "John Smith mentioned you in a comment"
        body = """
        Someone commented on your post!

        John Smith mentioned you in a comment on your photo.

        "Hey @user, check this out!"

        You have 5 new notifications.

        Click here to view and respond.

        ---

        To manage notification preferences, visit your settings.
        """

        features = detect_template(subject, body)

        assert features.is_templated is True
        assert 'social_notification' in features.detected_services

    def test_structural_detection(self):
        """Detects structured/formatted emails."""
        subject = "Monthly Report"
        body = """
        ================================
        MONTHLY SALES REPORT
        ================================

        * Region A: $100,000
        * Region B: $85,000
        * Region C: $120,000

        --------------------------------

        1. Top performer: Jane Doe
        2. Most improved: Bob Smith
        3. Rising star: Alice Wang

        ================================
        """

        features = detect_template(subject, body)

        assert features.structural_repetition_score > 0
        # May or may not be classified as templated depending on threshold

    def test_feature_vector_dimensions(self):
        """Feature vector has correct dimensions."""
        features = detect_template("Test", "Test body")
        vec = features.to_feature_vector()

        assert len(vec) == 12  # Expected dimension

    def test_threshold_parameter(self):
        """Custom threshold affects is_templated flag."""
        subject = "Test"
        body = "Simple email with unsubscribe link. Unsubscribe here."

        # Lower threshold - more likely to flag as templated
        features_low = detect_template(subject, body, threshold=0.1)
        # Higher threshold - less likely to flag
        features_high = detect_template(subject, body, threshold=0.8)

        assert features_low.template_confidence == features_high.template_confidence
        # Low threshold may mark as templated while high doesn't
        if features_low.template_confidence >= 0.1 and features_low.template_confidence < 0.8:
            assert features_low.is_templated is True
            assert features_high.is_templated is False


class TestComputeTemplateScore:
    """Tests for compute_template_score function."""

    def test_non_templated_full_score(self):
        """Non-templated emails get full priority score."""
        features = TemplateFeatures(
            is_templated=False,
            template_confidence=0.1,
            template_type='none',
            placeholder_count=0,
            unsubscribe_signals=0,
            marketing_footer_signals=0,
            service_pattern_matches=0,
            detected_services=[],
            has_bulk_headers=False,
            structural_repetition_score=0.0,
            llm_analyzed=False,
            llm_template_type=None,
            llm_confidence=None,
        )

        score = compute_template_score(features)
        assert score == 1.0

    def test_marketing_low_score(self):
        """Marketing emails get low priority score."""
        features = TemplateFeatures(
            is_templated=True,
            template_confidence=0.8,
            template_type='marketing',
            placeholder_count=2,
            unsubscribe_signals=3,
            marketing_footer_signals=4,
            service_pattern_matches=0,
            detected_services=[],
            has_bulk_headers=True,
            structural_repetition_score=0.5,
            llm_analyzed=False,
            llm_template_type=None,
            llm_confidence=None,
        )

        score = compute_template_score(features)
        assert score < 0.2

    def test_transactional_moderate_score(self):
        """Transactional emails get moderate priority (still somewhat important)."""
        features = TemplateFeatures(
            is_templated=True,
            template_confidence=0.7,
            template_type='transactional',
            placeholder_count=1,
            unsubscribe_signals=1,
            marketing_footer_signals=1,
            service_pattern_matches=2,
            detected_services=['shipping_notification'],
            has_bulk_headers=False,
            structural_repetition_score=0.3,
            llm_analyzed=False,
            llm_template_type=None,
            llm_confidence=None,
        )

        score = compute_template_score(features)
        assert score >= 0.3  # Floor for important transactional

    def test_newsletter_low_score(self):
        """Newsletter emails get low priority score."""
        features = TemplateFeatures(
            is_templated=True,
            template_confidence=0.9,
            template_type='newsletter',
            placeholder_count=0,
            unsubscribe_signals=2,
            marketing_footer_signals=2,
            service_pattern_matches=1,
            detected_services=['newsletter'],
            has_bulk_headers=True,
            structural_repetition_score=0.6,
            llm_analyzed=False,
            llm_template_type=None,
            llm_confidence=None,
        )

        score = compute_template_score(features)
        assert score < 0.3


class TestDetectTemplateWithLlm:
    """Tests for detect_template_with_llm function."""

    def test_without_llm_client(self):
        """Works without LLM client (just regex)."""
        features = detect_template_with_llm(
            "Test Subject",
            "Regular email body",
            llm_client=None,
        )

        assert features.llm_analyzed is False
        assert features.llm_template_type is None

    def test_confident_detection_skips_llm(self):
        """High confidence detection skips LLM."""
        # Create a mock LLM client that would fail if called
        class FailingLLMClient:
            def messages(self, *args, **kwargs):
                raise RuntimeError("Should not be called")

        body = """
        {{customer_name}},
        Your order #12345 shipped!
        Track: https://track.example.com

        Unsubscribe | Privacy Policy | Terms
        © 2026 Example Corp
        """

        features = detect_template_with_llm(
            "Your order has shipped",
            body,
            llm_client=FailingLLMClient(),
            llm_threshold=0.5,
        )

        # Should detect with high confidence from regex alone
        assert features.is_templated is True
        # LLM not called because confidence is high enough


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_email(self):
        """Handles empty email gracefully."""
        features = detect_template("", "")

        assert features.is_templated is False
        assert features.template_confidence == 0.0

    def test_very_long_email(self):
        """Handles long emails without issues."""
        long_body = "Lorem ipsum " * 5000

        features = detect_template("Long Email", long_body)

        assert isinstance(features, TemplateFeatures)

    def test_unicode_content(self):
        """Handles unicode content."""
        subject = "你好 - Order Confirmation"
        body = """
        Dear 用户,

        Your order has shipped! 🎉

        Tracking: ABC123

        © 2026 Example Store
        Unsubscribe from emails
        """

        features = detect_template(subject, body)

        assert isinstance(features, TemplateFeatures)
        assert features.unsubscribe_signals >= 1

    def test_bulk_headers_detection(self):
        """Detects bulk mail headers."""
        features = detect_template(
            "Newsletter",
            "Check out our latest updates!",
            headers={
                'List-Unsubscribe': '<mailto:unsub@example.com>',
                'Precedence': 'bulk',
            }
        )

        assert features.has_bulk_headers is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
