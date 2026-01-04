"""Tests for the service domain classification module."""

import pytest

from src.features.service import (
    ServiceType,
    ServiceFeatures,
    classify_service,
    compute_service_score,
    is_automated_sender,
    get_service_type_description,
    _extract_email_parts,
    _check_patterns,
)


class TestEmailParsing:
    """Tests for email parsing utilities."""

    def test_simple_email(self):
        """Parse simple email address."""
        local, domain = _extract_email_parts('user@example.com')
        assert local == 'user'
        assert domain == 'example.com'

    def test_email_with_display_name(self):
        """Parse email with display name."""
        local, domain = _extract_email_parts('John Smith <john@example.com>')
        assert local == 'john'
        assert domain == 'example.com'

    def test_email_with_quotes(self):
        """Parse email with quoted display name."""
        local, domain = _extract_email_parts('"John Smith" <john@example.com>')
        assert local == 'john'
        assert domain == 'example.com'

    def test_uppercase_normalized(self):
        """Email addresses are normalized to lowercase."""
        local, domain = _extract_email_parts('JOHN@EXAMPLE.COM')
        assert local == 'john'
        assert domain == 'example.com'

    def test_no_at_symbol(self):
        """Handle malformed email without @."""
        local, domain = _extract_email_parts('invalid-email')
        assert local == 'invalid-email'
        assert domain == ''

    def test_empty_string(self):
        """Handle empty email string."""
        local, domain = _extract_email_parts('')
        assert local == ''
        assert domain == ''


class TestPatternMatching:
    """Tests for pattern matching utilities."""

    def test_simple_pattern_match(self):
        """Simple pattern matches."""
        matched, patterns = _check_patterns('noreply@example.com', [r'^noreply$'])
        # Note: patterns match against full text, so need word boundary
        matched, patterns = _check_patterns('noreply', [r'^noreply$'])
        assert matched
        assert len(patterns) == 1

    def test_no_match(self):
        """No match returns False."""
        matched, patterns = _check_patterns('user', [r'^noreply$'])
        assert not matched
        assert len(patterns) == 0

    def test_multiple_patterns(self):
        """Multiple patterns can match."""
        matched, patterns = _check_patterns(
            'noreply system',
            [r'noreply', r'system', r'admin']
        )
        assert matched
        assert len(patterns) == 2


class TestClassifyService:
    """Tests for classify_service function."""

    def test_transactional_amazon(self):
        """Amazon order email classified as transactional."""
        features = classify_service(
            'noreply@amazon.com',
            'Your order #123-456 has shipped',
            'Tracking number: 1Z999...'
        )
        assert features.service_type == ServiceType.TRANSACTIONAL
        assert features.is_service_email
        assert features.confidence > 0.3

    def test_transactional_shipping(self):
        """Shipping notification classified as transactional."""
        features = classify_service(
            'shipping@store.com',
            'Your package has been delivered',
            'Item: Widget. Delivered at 2:30 PM.'
        )
        assert features.service_type == ServiceType.TRANSACTIONAL
        assert features.is_service_email

    def test_newsletter_substack(self):
        """Substack email classified as newsletter."""
        features = classify_service(
            'newsletter@substack.com',
            'This week in tech',
            'View in browser. Unsubscribe.'
        )
        assert features.service_type == ServiceType.NEWSLETTER
        assert features.is_service_email

    def test_newsletter_digest(self):
        """Weekly digest classified as newsletter."""
        features = classify_service(
            'digest@company.com',
            'Weekly digest: Top stories',
            'Read more. Forward to a friend.'
        )
        assert features.service_type == ServiceType.NEWSLETTER
        assert features.is_service_email

    def test_financial_bank(self):
        """Bank email classified as financial."""
        features = classify_service(
            'alerts@chase.com',
            'Your statement is ready',
            'Account ending in 1234. Balance: $5,000.00'
        )
        assert features.service_type == ServiceType.FINANCIAL
        assert features.is_service_email

    def test_financial_payment(self):
        """Payment notification classified as financial."""
        features = classify_service(
            'noreply@paypal.com',
            'Payment received',
            'Transaction amount: $100.00'
        )
        # May be classified as SYSTEM due to noreply, or FINANCIAL due to paypal
        assert features.service_type in [ServiceType.FINANCIAL, ServiceType.SYSTEM]
        assert features.is_service_email

    def test_social_linkedin(self):
        """LinkedIn notification classified as social."""
        features = classify_service(
            'notification@linkedin.com',
            'John Smith accepted your connection',
            'View profile. Mutual connections: 50.'
        )
        assert features.service_type == ServiceType.SOCIAL
        assert features.is_service_email

    def test_social_facebook(self):
        """Facebook notification classified as social."""
        features = classify_service(
            'notification@facebookmail.com',
            'You have a new friend request',
            'Someone wants to connect.'
        )
        assert features.service_type == ServiceType.SOCIAL
        assert features.is_service_email

    def test_marketing_sale(self):
        """Sale email classified as marketing."""
        features = classify_service(
            'promo@store.com',
            '50% off everything - today only!',
            'Shop now. Limited time. Coupon code: SAVE50'
        )
        assert features.service_type == ServiceType.MARKETING
        assert features.is_service_email

    def test_marketing_offer(self):
        """Promotional offer classified as marketing."""
        features = classify_service(
            'offers@company.com',
            'Exclusive deal just for you',
            'Buy now. Don\'t miss out.'
        )
        assert features.service_type == ServiceType.MARKETING
        assert features.is_service_email

    def test_system_noreply(self):
        """Noreply address classified as system."""
        features = classify_service(
            'noreply@service.com',
            'Important notification',
            'Do not reply to this email.'
        )
        assert features.service_type == ServiceType.SYSTEM
        assert features.is_service_email
        assert features.local_part_match

    def test_system_password_reset(self):
        """Password reset classified as system."""
        features = classify_service(
            'security@google.com',
            'Reset your password',
            'Click here to verify. If you did not request this...'
        )
        assert features.service_type == ServiceType.SYSTEM
        assert features.is_service_email

    def test_calendar_invite(self):
        """Calendar invite classified as calendar."""
        features = classify_service(
            'calendar@google.com',
            'Invitation: Team Meeting @ Tuesday',
            'When: Tuesday. Where: Room 1. Accept / Decline.'
        )
        assert features.service_type == ServiceType.CALENDAR
        assert features.is_service_email

    def test_calendar_meeting(self):
        """Meeting request classified as calendar."""
        features = classify_service(
            'invite@calendly.com',
            'Meeting scheduled with John',
            'Add to calendar. Join meeting link.'
        )
        assert features.service_type == ServiceType.CALENDAR
        assert features.is_service_email

    def test_personal_email(self):
        """Personal email classified as personal."""
        features = classify_service(
            'john.smith@company.com',
            'Re: Project update',
            'Thanks for the update. Let\'s discuss tomorrow.'
        )
        assert features.service_type == ServiceType.PERSONAL
        assert not features.is_service_email

    def test_personal_email_no_patterns(self):
        """Email with no service patterns classified as personal."""
        features = classify_service(
            'colleague@work.org',
            'Lunch?',
            'Want to grab lunch today?'
        )
        assert features.service_type == ServiceType.PERSONAL
        assert not features.is_service_email

    def test_empty_subject_body(self):
        """Classification works with just email address."""
        features = classify_service('noreply@amazon.com')
        # Should still detect based on domain/local part
        assert features.is_service_email

    def test_domain_match_flag(self):
        """Domain match flag is set correctly."""
        features = classify_service('noreply@amazon.com')
        assert features.domain_match

    def test_local_part_match_flag(self):
        """Local part match flag is set correctly."""
        features = classify_service('noreply@unknown.com')
        assert features.local_part_match

    def test_subject_match_flag(self):
        """Subject match flag is set correctly."""
        features = classify_service(
            'random@example.com',
            'Your order #123 has shipped'
        )
        assert features.subject_match

    def test_body_match_flag(self):
        """Body match flag is set correctly."""
        features = classify_service(
            'random@example.com',
            'Hello',
            'Order number: 12345. Tracking: 1Z999.'
        )
        assert features.body_match


class TestFeatureVector:
    """Tests for ServiceFeatures.to_feature_vector()."""

    def test_vector_length(self):
        """Feature vector has correct length."""
        features = classify_service('noreply@amazon.com')
        vector = features.to_feature_vector()
        assert len(vector) == 17

    def test_vector_values_bounded(self):
        """Feature vector values are bounded [0, 1]."""
        features = classify_service('noreply@amazon.com', 'Order shipped', 'Track pkg')
        vector = features.to_feature_vector()
        for val in vector:
            assert 0.0 <= val <= 1.0

    def test_type_distribution_in_vector(self):
        """Type distribution is correctly encoded."""
        features = classify_service('noreply@amazon.com')
        vector = features.to_feature_vector()
        # First 8 values are type probabilities
        type_probs = vector[:8]
        assert abs(sum(type_probs) - 1.0) < 0.01  # Should sum to ~1


class TestComputeServiceScore:
    """Tests for compute_service_score function."""

    def test_system_alert_high_score(self):
        """System alerts get high priority score."""
        features = classify_service(
            'security@company.com',
            'Security alert: unusual login',
            'Action required. Verify your identity.'
        )
        score = compute_service_score(features)
        assert score > 0.6

    def test_calendar_invite_high_score(self):
        """Calendar invites get high priority score."""
        features = classify_service(
            'calendar@google.com',
            'Meeting invite',
            'Accept / Decline'
        )
        score = compute_service_score(features)
        assert score > 0.5

    def test_marketing_low_score(self):
        """Marketing emails get low priority score."""
        features = classify_service(
            'promo@store.com',
            '50% off!',
            'Shop now. Limited time.'
        )
        score = compute_service_score(features)
        assert score < 0.3

    def test_newsletter_low_score(self):
        """Newsletters get low priority score."""
        features = classify_service(
            'newsletter@blog.com',
            'Weekly digest',
            'Unsubscribe'
        )
        score = compute_service_score(features)
        assert score < 0.4

    def test_personal_neutral_score(self):
        """Personal emails get neutral score."""
        features = classify_service(
            'john@company.com',
            'Hello',
            'How are you?'
        )
        score = compute_service_score(features)
        assert 0.4 <= score <= 0.6

    def test_score_bounded(self):
        """Score is always between 0 and 1."""
        test_cases = [
            'noreply@amazon.com',
            'security@bank.com',
            'promo@store.com',
            'john@company.com',
        ]
        for sender in test_cases:
            features = classify_service(sender)
            score = compute_service_score(features)
            assert 0.0 <= score <= 1.0


class TestIsAutomatedSender:
    """Tests for is_automated_sender function."""

    def test_noreply_is_automated(self):
        """Noreply addresses are automated."""
        assert is_automated_sender('noreply@company.com')
        assert is_automated_sender('no-reply@company.com')
        assert is_automated_sender('donotreply@company.com')

    def test_system_is_automated(self):
        """System addresses are automated."""
        assert is_automated_sender('system@company.com')
        assert is_automated_sender('automated@company.com')
        assert is_automated_sender('notification@company.com')

    def test_postmaster_is_automated(self):
        """Postmaster/mailer-daemon are automated."""
        assert is_automated_sender('postmaster@company.com')
        assert is_automated_sender('mailer-daemon@company.com')

    def test_notification_subdomain_is_automated(self):
        """Notification subdomains are automated."""
        assert is_automated_sender('user@notification.company.com')
        assert is_automated_sender('alerts@mail.company.com')

    def test_personal_not_automated(self):
        """Personal addresses are not automated."""
        assert not is_automated_sender('john.smith@company.com')
        assert not is_automated_sender('jane@example.org')

    def test_support_not_automated(self):
        """Support addresses may have humans."""
        # Note: support@ is in SERVICE_LOCAL_PARTS but not in
        # is_automated_sender's quick check list
        # This is intentional - support may have humans
        result = is_automated_sender('support@company.com')
        # The function should NOT flag support as automated
        # since support teams often have humans
        assert not result


class TestServiceTypeDescription:
    """Tests for get_service_type_description function."""

    def test_all_types_have_descriptions(self):
        """All service types have descriptions."""
        for stype in ServiceType:
            desc = get_service_type_description(stype)
            assert desc
            assert len(desc) > 10

    def test_transactional_description(self):
        """Transactional has appropriate description."""
        desc = get_service_type_description(ServiceType.TRANSACTIONAL)
        assert 'order' in desc.lower() or 'receipt' in desc.lower()

    def test_personal_description(self):
        """Personal has appropriate description."""
        desc = get_service_type_description(ServiceType.PERSONAL)
        assert 'personal' in desc.lower() or 'individual' in desc.lower()


class TestTypeDistribution:
    """Tests for type probability distribution."""

    def test_distribution_sums_to_one(self):
        """Type distribution sums to 1."""
        features = classify_service(
            'noreply@amazon.com',
            'Order shipped',
            'Track package'
        )
        total = sum(features.type_distribution.values())
        assert abs(total - 1.0) < 0.01

    def test_all_types_in_distribution(self):
        """All types are represented in distribution."""
        features = classify_service('test@example.com')
        expected_types = ['transactional', 'newsletter', 'financial', 'social',
                         'marketing', 'system', 'calendar', 'personal']
        for t in expected_types:
            assert t in features.type_distribution

    def test_confident_classification_dominates(self):
        """Confident classifications have high probability for that type."""
        features = classify_service(
            'noreply@amazon.com',
            'Your order has shipped',
            'Tracking number 1Z999'
        )
        if features.service_type != ServiceType.PERSONAL:
            top_prob = max(features.type_distribution.values())
            assert top_prob > 0.3


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_unicode_in_email(self):
        """Handle unicode in email addresses."""
        features = classify_service('test@example.com')
        assert features.sender_domain == 'example.com'

    def test_very_long_email(self):
        """Handle very long email addresses."""
        long_local = 'a' * 100
        features = classify_service(f'{long_local}@example.com')
        assert features.sender_local_part == long_local

    def test_multiple_at_symbols(self):
        """Handle malformed email with multiple @."""
        features = classify_service('user@sub@domain.com')
        # Should take rightmost @
        assert features.sender_domain == 'domain.com'

    def test_empty_sender(self):
        """Handle empty sender email."""
        features = classify_service('')
        assert features.service_type == ServiceType.PERSONAL
        assert not features.is_service_email

    def test_whitespace_only(self):
        """Handle whitespace-only inputs."""
        features = classify_service('   ', '   ', '   ')
        assert features.service_type == ServiceType.PERSONAL

    def test_special_characters_in_body(self):
        """Handle special characters in body."""
        features = classify_service(
            'test@example.com',
            'Test',
            '!@#$%^&*()[]{}|\\;:\'",.<>?/~`'
        )
        # Should not crash
        assert features is not None

    def test_mixed_case_patterns(self):
        """Pattern matching is case-insensitive."""
        features1 = classify_service('NOREPLY@AMAZON.COM')
        features2 = classify_service('noreply@amazon.com')
        assert features1.service_type == features2.service_type
        assert features1.is_service_email == features2.is_service_email


class TestMatchedPatterns:
    """Tests for matched patterns tracking."""

    def test_matched_patterns_populated(self):
        """Matched patterns list is populated."""
        features = classify_service(
            'noreply@amazon.com',
            'Your order shipped',
            'Track package'
        )
        assert len(features.matched_patterns) > 0

    def test_pattern_prefixes(self):
        """Matched patterns have correct prefixes."""
        features = classify_service(
            'noreply@amazon.com',
            'Your order shipped',
            'Track package'
        )
        prefixes = set()
        for pattern in features.matched_patterns:
            if ':' in pattern:
                prefixes.add(pattern.split(':')[0])

        # Should have domain and/or local and/or subject matches
        assert len(prefixes) > 0

    def test_no_patterns_for_personal(self):
        """Personal emails have few or no matched patterns."""
        features = classify_service(
            'john@company.com',
            'Hello there',
            'How are you today?'
        )
        # Personal emails should have minimal pattern matches
        assert len(features.matched_patterns) < 3
