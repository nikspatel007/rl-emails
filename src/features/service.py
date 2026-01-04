#!/usr/bin/env python3
"""Service domain classification for email filtering.

Classifies emails from service senders (noreply@, support@, etc.) into:
- TRANSACTIONAL: Order confirmations, receipts, shipping notifications
- NEWSLETTER: Subscribed content, digests, updates
- FINANCIAL: Bank statements, payment notifications, investment updates
- SOCIAL: Social network notifications, connection requests
- MARKETING: Promotions, sales, advertisements
- SYSTEM: Password resets, security alerts, account notifications
- CALENDAR: Event invites, reminders, scheduling

This module provides both rule-based classification (domain/sender patterns)
and hooks for ML-based classification using email content.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False


class ServiceType(Enum):
    """Classification types for service emails."""
    TRANSACTIONAL = "transactional"
    NEWSLETTER = "newsletter"
    FINANCIAL = "financial"
    SOCIAL = "social"
    MARKETING = "marketing"
    SYSTEM = "system"
    CALENDAR = "calendar"
    PERSONAL = "personal"  # Not a service email


# Domain patterns for each service type
# Maps service type to list of domain regex patterns
DOMAIN_PATTERNS: dict[ServiceType, list[str]] = {
    ServiceType.TRANSACTIONAL: [
        r'(?:order|shipping|delivery|tracking|receipt|confirmation)s?[@.]',
        r'@(?:amazon|ebay|etsy|shopify|stripe|square|paypal)\.com$',
        r'@(?:fedex|ups|usps|dhl)\.com$',
        r'orders?@',
        r'shipping@',
        r'receipts?@',
    ],
    ServiceType.NEWSLETTER: [
        r'newsletter@',
        r'digest@',
        r'updates?@',
        r'(?:weekly|daily|monthly)@',
        r'@(?:substack|mailchimp|constantcontact|sendgrid)\.com$',
        r'news@',
        r'editorial@',
    ],
    ServiceType.FINANCIAL: [
        r'@(?:.*\.)?(?:bank|chase|wellsfargo|bofa|citi|capitalone|amex|visa|mastercard)\.com$',
        r'@(?:paypal|venmo|zelle|cashapp|robinhood|fidelity|schwab|vanguard)\.com$',
        r'statements?@',
        r'alerts?@.*(?:bank|financial|credit)',
        r'@.*(?:credit|loan|mortgage)\.com$',
        r'banking@',
        r'investments?@',
    ],
    ServiceType.SOCIAL: [
        r'@(?:facebook|fb|instagram|twitter|x|linkedin|tiktok|snapchat|pinterest)\.com$',
        r'@(?:facebookmail|twittermail)\.com$',
        r'notification@',
        r'(?:friend|connection|follow).*@',
        r'@.*social.*\.com$',
        r'@discord\.com$',
        r'@slack\.com$',
        r'@teams\.microsoft\.com$',
    ],
    ServiceType.MARKETING: [
        r'(?:promo|promotion|offer|sale|deal|discount)s?@',
        r'marketing@',
        r'campaign@',
        r'(?:special|exclusive)@',
        r'@(?:constantcontact|hubspot|marketo)\.com$',
        r'(?:email|e-?mail).*(?:blast|campaign)@',
    ],
    ServiceType.SYSTEM: [
        r'(?:noreply|no-reply|donotreply|do-not-reply)@',
        r'(?:password|security|verify|verification|confirm|auth)@',
        r'(?:admin|administrator|system|support|help|service)@',
        r'@.*\.(?:notifications?|alerts?)\.com$',
        r'(?:account|accounts)@',
        r'automated?@',
        r'mailer-daemon@',
        r'postmaster@',
        r'notification@',
    ],
    ServiceType.CALENDAR: [
        r'calendar@',
        r'(?:invite|invitation|event|meeting|schedule|booking)s?@',
        r'@calendar\.google\.com$',
        r'@.*calendly\.com$',
        r'@outlook\.com$.*(?:calendar|invite)',
        r'rsvp@',
        r'appointments?@',
    ],
}

# Local part patterns that indicate automated/service senders
SERVICE_LOCAL_PARTS = [
    (ServiceType.SYSTEM, [
        r'^noreply$', r'^no-reply$', r'^donotreply$', r'^do-not-reply$',
        r'^admin$', r'^administrator$', r'^system$', r'^support$',
        r'^help$', r'^service$', r'^info$', r'^contact$',
        r'^postmaster$', r'^mailer-daemon$', r'^bounce',
        r'^automated?$', r'^robot$', r'^bot$',
    ]),
    (ServiceType.TRANSACTIONAL, [
        r'^order', r'^orders$', r'^shipping$', r'^tracking$',
        r'^receipt', r'^confirmation', r'^delivery$',
    ]),
    (ServiceType.NEWSLETTER, [
        r'^newsletter', r'^digest$', r'^updates?$', r'^news$',
        r'^weekly$', r'^daily$', r'^monthly$', r'^editorial$',
    ]),
    (ServiceType.FINANCIAL, [
        r'^statement', r'^banking$', r'^alerts?$',
        r'^payments?$', r'^billing$', r'^invoice',
    ]),
    (ServiceType.SOCIAL, [
        r'^notification', r'^friend', r'^follow', r'^connect',
        r'^message$', r'^mention', r'^reply$',
    ]),
    (ServiceType.MARKETING, [
        r'^promo', r'^offers?$', r'^deals?$', r'^sales?$',
        r'^marketing$', r'^campaign', r'^special$',
    ]),
    (ServiceType.CALENDAR, [
        r'^calendar$', r'^invite', r'^event', r'^meeting',
        r'^schedule', r'^rsvp$', r'^booking', r'^appointment',
    ]),
]

# Subject line patterns for classification
SUBJECT_PATTERNS: dict[ServiceType, list[str]] = {
    ServiceType.TRANSACTIONAL: [
        r'\b(?:order|purchase)\s*(?:#|number|confirmation)',
        r'\b(?:shipped|delivered|tracking)',
        r'\b(?:receipt|invoice)\s*(?:for|from)',
        r'\byour\s+(?:order|package|delivery)\b',
    ],
    ServiceType.NEWSLETTER: [
        r'\b(?:weekly|daily|monthly)\s+(?:digest|update|newsletter)',
        r'\bnewsletter\b',
        r'\b(?:this\s+week|today)\s+(?:in|at|from)\b',
        r'\bedition\s*[#:\d]',
    ],
    ServiceType.FINANCIAL: [
        r'\b(?:statement|balance|payment)\s+(?:available|ready|due)',
        r'\b(?:account|credit\s*card)\s+(?:alert|activity|statement)',
        r'\b(?:payment|transfer)\s+(?:received|sent|confirmed)',
        r'\b(?:bank|financial)\s+(?:notice|alert|update)',
    ],
    ServiceType.SOCIAL: [
        r'\b(?:added|accepted)\s+(?:you|your)\s+(?:as\s+)?(?:friend|connection)',
        r'\b(?:mentioned|tagged|replied)\s+(?:you|to\s+you)',
        r'\bnew\s+(?:message|follower|connection|friend)',
        r'\b(?:someone|people)\s+(?:liked|commented|shared)',
    ],
    ServiceType.MARKETING: [
        r'\b(?:sale|offer|deal|discount)\s+(?:ends?|expires?|today|now)',
        r'\b(?:exclusive|limited|special)\s+(?:offer|deal|discount)',
        r'\b(?:\d+%|half)\s+off\b',
        r'\bdon\'?t\s+miss\b',
        r'\bfree\s+(?:shipping|delivery|trial)',
    ],
    ServiceType.SYSTEM: [
        r'\b(?:password|security)\s+(?:reset|change|alert)',
        r'\b(?:verify|confirm)\s+(?:your|email|account)',
        r'\b(?:account|login)\s+(?:alert|activity|notification)',
        r'\b(?:action\s+)?required\b',
        r'\bunusual\s+(?:activity|sign-?in)',
    ],
    ServiceType.CALENDAR: [
        r'\b(?:invitation|invite|event)\s*:',
        r'\b(?:meeting|call|appointment)\s+(?:scheduled|reminder|invite)',
        r'\b(?:accepted|declined|tentative)\s*:',
        r'\brsvp\b',
        r'\b(?:join|attend)\s+(?:meeting|event|call)',
    ],
}

# Body content patterns (for ML-assisted classification)
BODY_PATTERNS: dict[ServiceType, list[str]] = {
    ServiceType.TRANSACTIONAL: [
        r'\border\s*(?:#|number|id)\s*[:\s]?\s*\w+',
        r'\btracking\s*(?:#|number)\b',
        r'\b(?:item|product|qty|quantity|price|total|subtotal)\b',
        r'\bunsubscribe\b.*\breceipts?\b',
    ],
    ServiceType.NEWSLETTER: [
        r'\bview\s+(?:in\s+)?(?:browser|online)\b',
        r'\bunsubscribe\b',
        r'\b(?:read|view)\s+(?:more|full\s+(?:article|story))\b',
        r'\bforwarded?\s+(?:to\s+)?(?:a\s+)?friend\b',
    ],
    ServiceType.FINANCIAL: [
        r'\b(?:balance|available|minimum\s+payment)\s*:?\s*\$[\d,.]+',
        r'\b(?:account\s+ending\s+in|account\s*#)\s*\d+',
        r'\b(?:transaction|payment)\s+(?:date|amount|reference)',
        r'\bfdic\b|\bsipc\b',
    ],
    ServiceType.SOCIAL: [
        r'\bview\s+(?:profile|post|photo|message)\b',
        r'\b(?:like|comment|share|reply)\s+(?:button|now|back)\b',
        r'\bconnect\s+(?:with|on)\b',
        r'\bmutual\s+(?:friends?|connections?)\b',
    ],
    ServiceType.MARKETING: [
        r'\bshop\s+now\b',
        r'\b(?:buy|order|get\s+yours?)\s+(?:now|today)\b',
        r'\blimited\s+(?:time|stock|availability)\b',
        r'\b(?:coupon|promo)\s*code\b',
        r'\bclick\s+(?:here|below)\s+to\s+(?:buy|order|shop)\b',
    ],
    ServiceType.SYSTEM: [
        r'\b(?:click|tap)\s+(?:here|below|the\s+(?:link|button))\s+to\s+(?:verify|confirm|reset)',
        r'\bif\s+you\s+did(?:n\'?t|not)\s+(?:request|make|initiate)\b',
        r'\b(?:expires?|valid)\s+(?:in|for)\s+\d+\s+(?:hour|minute|day)',
        r'\bdo\s+not\s+(?:reply|respond)\s+(?:to\s+)?this\s+(?:email|message)\b',
    ],
    ServiceType.CALENDAR: [
        r'\b(?:when|date|time)\s*:\s*',
        r'\b(?:where|location|venue)\s*:\s*',
        r'\b(?:accept|decline|maybe|tentative)\b',
        r'\badd\s+to\s+calendar\b',
        r'\b(?:join|dial\s+in|video\s+link|meeting\s+link)\b',
    ],
}


@dataclass
class ServiceFeatures:
    """Service classification features for an email."""
    # Classification results
    service_type: ServiceType
    confidence: float  # 0-1, how confident in the classification
    type_distribution: dict[str, float]  # Probability for each type

    # Detection signals
    is_service_email: bool  # True if from a service sender
    domain_match: bool  # Domain pattern matched
    local_part_match: bool  # Local part pattern matched
    subject_match: bool  # Subject pattern matched
    body_match: bool  # Body pattern matched

    # Sender info
    sender_local_part: str
    sender_domain: str

    # Metadata
    matched_patterns: list[str]  # Which patterns triggered

    def to_feature_vector(self) -> Union["np.ndarray", list[float]]:
        """Convert to numpy array for ML pipeline.

        Returns 17-dimensional vector:
        - 8 service type probabilities (including PERSONAL)
        - 4 detection signal flags
        - 1 is_service flag
        - 1 confidence score
        - 3 reserved for future use
        """
        # Type probabilities (8)
        type_order = [
            'transactional', 'newsletter', 'financial', 'social',
            'marketing', 'system', 'calendar', 'personal'
        ]
        type_probs = [self.type_distribution.get(t, 0.0) for t in type_order]

        # Detection flags (4)
        flags = [
            1.0 if self.domain_match else 0.0,
            1.0 if self.local_part_match else 0.0,
            1.0 if self.subject_match else 0.0,
            1.0 if self.body_match else 0.0,
        ]

        # Scalar features (5)
        scalars = [
            1.0 if self.is_service_email else 0.0,
            self.confidence,
            0.0,  # Reserved
            0.0,  # Reserved
            0.0,  # Reserved
        ]

        values = type_probs + flags + scalars
        if HAS_NUMPY:
            return np.array(values, dtype=np.float32)
        return values


def _extract_email_parts(email_addr: str) -> tuple[str, str]:
    """Extract local part and domain from email address."""
    email_addr = email_addr.strip().lower()

    # Handle display name format: "John Smith" <john.smith@example.com>
    match = re.search(r'<([^>]+)>', email_addr)
    if match:
        email_addr = match.group(1)

    if '@' in email_addr:
        local, domain = email_addr.rsplit('@', 1)
        return local, domain
    return email_addr, ''


def _check_patterns(text: str, patterns: list[str]) -> tuple[bool, list[str]]:
    """Check if text matches any patterns, return matched patterns."""
    text_lower = text.lower()
    matched = []
    for pattern in patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            matched.append(pattern)
    return len(matched) > 0, matched


def _compute_type_scores(
    sender_email: str,
    subject: str,
    body: str,
) -> tuple[dict[ServiceType, float], list[str]]:
    """Compute scores for each service type.

    Returns (scores dict, list of matched patterns).
    """
    local_part, domain = _extract_email_parts(sender_email)
    full_email = f"{local_part}@{domain}"
    all_matched = []

    scores: dict[ServiceType, float] = {st: 0.0 for st in ServiceType}

    # Check domain patterns (weight: 3.0)
    for stype, patterns in DOMAIN_PATTERNS.items():
        matched, pattern_list = _check_patterns(full_email, patterns)
        if matched:
            scores[stype] += 3.0
            all_matched.extend([f"domain:{p}" for p in pattern_list])

    # Check local part patterns (weight: 2.5)
    for stype, patterns in SERVICE_LOCAL_PARTS:
        matched, pattern_list = _check_patterns(local_part, patterns)
        if matched:
            scores[stype] += 2.5
            all_matched.extend([f"local:{p}" for p in pattern_list])

    # Check subject patterns (weight: 2.0)
    for stype, patterns in SUBJECT_PATTERNS.items():
        matched, pattern_list = _check_patterns(subject, patterns)
        if matched:
            scores[stype] += 2.0
            all_matched.extend([f"subject:{p}" for p in pattern_list])

    # Check body patterns (weight: 1.5)
    for stype, patterns in BODY_PATTERNS.items():
        matched, pattern_list = _check_patterns(body, patterns)
        if matched:
            scores[stype] += 1.5
            all_matched.extend([f"body:{p}" for p in pattern_list])

    return scores, all_matched


def _normalize_scores(scores: dict[ServiceType, float]) -> dict[str, float]:
    """Normalize scores to probability distribution."""
    # Filter out PERSONAL for normalization (it's the default/fallback)
    service_scores = {k: v for k, v in scores.items() if k != ServiceType.PERSONAL}
    total = sum(service_scores.values())

    if total == 0:
        # No service patterns matched - likely personal email
        return {st.value: 0.0 for st in ServiceType if st != ServiceType.PERSONAL} | {"personal": 1.0}

    # Normalize service types
    normalized = {st.value: score / total for st, score in service_scores.items()}

    # Personal probability is inverse of max service probability
    max_service_prob = max(normalized.values()) if normalized else 0.0
    normalized["personal"] = max(0.0, 1.0 - max_service_prob * 1.5)

    # Re-normalize to sum to 1
    total = sum(normalized.values())
    return {k: v / total for k, v in normalized.items()}


def classify_service(
    sender_email: str,
    subject: str = "",
    body: str = "",
) -> ServiceFeatures:
    """Classify email sender as a service type.

    Uses rule-based pattern matching on:
    - Sender domain and local part
    - Subject line patterns
    - Body content patterns

    Args:
        sender_email: Full sender email address
        subject: Email subject line (optional)
        body: Email body text (optional)

    Returns:
        ServiceFeatures with classification results
    """
    local_part, domain = _extract_email_parts(sender_email)
    full_email = f"{local_part}@{domain}"

    # Compute scores for each type
    raw_scores, matched_patterns = _compute_type_scores(sender_email, subject, body)

    # Check individual pattern match types
    domain_matched = any(
        _check_patterns(full_email, patterns)[0]
        for patterns in DOMAIN_PATTERNS.values()
    )
    local_matched = any(
        _check_patterns(local_part, patterns)[0]
        for _, patterns in SERVICE_LOCAL_PARTS
    )
    subject_matched = any(
        _check_patterns(subject, patterns)[0]
        for patterns in SUBJECT_PATTERNS.values()
    ) if subject else False
    body_matched = any(
        _check_patterns(body, patterns)[0]
        for patterns in BODY_PATTERNS.values()
    ) if body else False

    # Normalize to probability distribution
    type_distribution = _normalize_scores(raw_scores)

    # Determine primary type and confidence
    if sum(raw_scores.values()) == 0:
        # No patterns matched - personal email
        primary_type = ServiceType.PERSONAL
        confidence = 0.8  # Fairly confident it's personal
    else:
        # Find highest scoring service type
        primary_type = max(raw_scores, key=raw_scores.get)
        if raw_scores[primary_type] == 0:
            primary_type = ServiceType.PERSONAL
            confidence = 0.8
        else:
            confidence = type_distribution.get(primary_type.value, 0.0)

    # Determine if this is a service email
    is_service = primary_type != ServiceType.PERSONAL and confidence > 0.3

    return ServiceFeatures(
        service_type=primary_type,
        confidence=confidence,
        type_distribution=type_distribution,
        is_service_email=is_service,
        domain_match=domain_matched,
        local_part_match=local_matched,
        subject_match=subject_matched,
        body_match=body_matched,
        sender_local_part=local_part,
        sender_domain=domain,
        matched_patterns=matched_patterns,
    )


def compute_service_score(features: ServiceFeatures) -> float:
    """Compute service email handling score.

    Returns a score indicating how to prioritize the email:
    - Higher scores = needs attention (system alerts, calendar)
    - Lower scores = can be batched/deferred (newsletters, marketing)

    Args:
        features: ServiceFeatures from classify_service()

    Returns:
        Score from 0 to 1
    """
    if not features.is_service_email:
        # Personal emails get neutral score (handled by other features)
        return 0.5

    # Priority weights by service type
    type_weights = {
        ServiceType.SYSTEM: 0.8,       # Security alerts, password resets
        ServiceType.CALENDAR: 0.75,    # Meeting invites need attention
        ServiceType.TRANSACTIONAL: 0.6, # Order updates, receipts
        ServiceType.FINANCIAL: 0.65,   # Bank alerts
        ServiceType.SOCIAL: 0.4,       # Social notifications
        ServiceType.NEWSLETTER: 0.25,  # Can batch
        ServiceType.MARKETING: 0.15,   # Low priority
        ServiceType.PERSONAL: 0.5,     # Neutral
    }

    base_score = type_weights.get(features.service_type, 0.5)

    # Confidence modifier
    score = base_score * (0.7 + 0.3 * features.confidence)

    # Multiple signal matches boost confidence in the score
    match_count = sum([
        features.domain_match,
        features.local_part_match,
        features.subject_match,
        features.body_match,
    ])
    if match_count >= 3:
        score *= 1.1  # Boost for strong signal
    elif match_count == 1:
        score *= 0.9  # Slight penalty for weak signal

    return max(0.0, min(1.0, score))


def is_automated_sender(sender_email: str) -> bool:
    """Quick check if sender appears to be automated.

    Faster than full classification when you just need a boolean.

    Args:
        sender_email: Sender email address

    Returns:
        True if sender appears automated/service
    """
    local_part, domain = _extract_email_parts(sender_email)

    # Quick checks for common automated sender patterns
    automated_locals = [
        'noreply', 'no-reply', 'donotreply', 'do-not-reply',
        'mailer-daemon', 'postmaster', 'bounce', 'automated',
        'system', 'admin', 'notification', 'alert',
    ]

    if any(auto in local_part for auto in automated_locals):
        return True

    # Check for notification subdomains
    if any(x in domain for x in ['notification', 'alert', 'mail.', 'email.']):
        return True

    return False


def get_service_type_description(service_type: ServiceType) -> str:
    """Get human-readable description of service type.

    Args:
        service_type: ServiceType enum value

    Returns:
        Description string
    """
    descriptions = {
        ServiceType.TRANSACTIONAL: "Order confirmations, receipts, shipping updates",
        ServiceType.NEWSLETTER: "Subscribed content, digests, editorial updates",
        ServiceType.FINANCIAL: "Bank statements, payment alerts, investment updates",
        ServiceType.SOCIAL: "Social network notifications, messages, friend requests",
        ServiceType.MARKETING: "Promotions, sales, advertisements, offers",
        ServiceType.SYSTEM: "Password resets, security alerts, account notifications",
        ServiceType.CALENDAR: "Event invites, meeting requests, schedule reminders",
        ServiceType.PERSONAL: "Personal email from an individual",
    }
    return descriptions.get(service_type, "Unknown service type")


if __name__ == '__main__':
    # Example usage and testing
    test_cases = [
        {
            'sender': 'noreply@amazon.com',
            'subject': 'Your order #123-456 has shipped',
            'body': 'Track your package with tracking number 1Z999...',
        },
        {
            'sender': 'newsletter@substack.com',
            'subject': 'This week in tech: AI advances',
            'body': 'View in browser. Forward to a friend. Unsubscribe.',
        },
        {
            'sender': 'alerts@chase.com',
            'subject': 'Your account statement is ready',
            'body': 'Account ending in 1234. Balance: $5,000.00',
        },
        {
            'sender': 'notification@linkedin.com',
            'subject': 'John Smith accepted your connection request',
            'body': 'View profile. You have 500+ mutual connections.',
        },
        {
            'sender': 'offers@store.com',
            'subject': 'SALE! 50% off everything - today only!',
            'body': 'Shop now. Limited time. Coupon code: SAVE50',
        },
        {
            'sender': 'security@google.com',
            'subject': 'Verify your account',
            'body': 'Click here to verify. If you did not request this...',
        },
        {
            'sender': 'calendar@google.com',
            'subject': 'Invitation: Team Meeting @ Tue Jan 5',
            'body': 'When: Tuesday. Where: Conference Room. Accept / Decline',
        },
        {
            'sender': 'john.smith@company.com',
            'subject': 'Re: Project update',
            'body': 'Thanks for the update. Lets discuss tomorrow.',
        },
    ]

    print("Service Domain Classification Results")
    print("=" * 60)

    for i, tc in enumerate(test_cases, 1):
        features = classify_service(
            tc['sender'],
            tc.get('subject', ''),
            tc.get('body', ''),
        )

        print(f"\n{i}. {tc['sender']}")
        print(f"   Subject: {tc.get('subject', 'N/A')[:50]}...")
        print(f"   Type: {features.service_type.value}")
        print(f"   Confidence: {features.confidence:.2f}")
        print(f"   Is Service: {features.is_service_email}")
        print(f"   Score: {compute_service_score(features):.2f}")
        print(f"   Signals: domain={features.domain_match}, local={features.local_part_match}, "
              f"subject={features.subject_match}, body={features.body_match}")

        # Show top 3 type probabilities
        sorted_probs = sorted(features.type_distribution.items(), key=lambda x: -x[1])[:3]
        prob_str = ", ".join(f"{t}={p:.2f}" for t, p in sorted_probs)
        print(f"   Distribution: {prob_str}")
