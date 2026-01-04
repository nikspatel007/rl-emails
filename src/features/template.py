#!/usr/bin/env python3
"""Template pattern detection for email content.

Detects templated/automated emails by analyzing:
- Variable placeholders ({{name}}, {name}, %NAME%, etc.)
- Unsubscribe links and marketing footers
- Repetitive structural patterns
- Service email signatures (password reset, shipping, receipts)
- Bulk mail headers and indicators

Uses regex for fast detection with LLM fallback for edge cases.
"""

import re
from dataclasses import dataclass
from typing import Optional, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False


# Variable placeholder patterns (common template systems)
PLACEHOLDER_PATTERNS = [
    r'\{\{[^}]+\}\}',           # Mustache/Handlebars: {{name}}
    r'\{%[^%]+%\}',             # Jinja2/Django: {% if %}
    r'\$\{[^}]+\}',             # JavaScript template literals: ${name}
    r'%[A-Z_]+%',               # Batch/legacy: %NAME%
    r'\[\[.+?\]\]',             # Wiki-style: [[variable]]
    r'<\?=?\s*\$[^?]+\?>',      # PHP: <?= $var ?>
    r'\#\{[^}]+\}',             # Ruby: #{name}
]

# Unsubscribe and marketing patterns
UNSUBSCRIBE_PATTERNS = [
    r'\b(?:unsubscribe|opt.?out|remove\s+(?:me|yourself))\b',
    r'\b(?:manage\s+(?:subscriptions?|preferences)|email\s+preferences)\b',
    r'\b(?:stop\s+receiving|no\s+longer\s+(?:wish|want)\s+to\s+receive)\b',
    r'click\s+here\s+to\s+unsubscribe',
    r'if\s+you\s+(?:no\s+longer|don.?t)\s+(?:wish|want)\s+to\s+receive',
]

VIEW_IN_BROWSER_PATTERNS = [
    r'\b(?:view|open|read)\s+(?:this\s+)?(?:email|message|newsletter)\s+(?:in|on)\s+(?:your\s+)?(?:browser|web)',
    r'\b(?:having\s+trouble\s+viewing|can.?t\s+see\s+(?:this|the\s+images))',
    r'\b(?:view\s+online|view\s+as\s+webpage)',
]

# Marketing footer patterns
MARKETING_FOOTER_PATTERNS = [
    r'\b(?:this\s+email\s+was\s+sent\s+(?:to|by))\b',
    r'\b(?:you\s+(?:are\s+)?receiv(?:ed|ing)\s+this\s+(?:because|as))\b',
    r'\b(?:copyright|©)\s*(?:19|20)\d{2}',
    r'\b(?:all\s+rights\s+reserved)\b',
    r'\b(?:privacy\s+policy|terms\s+(?:of\s+)?(?:service|use))\b',
    r'\b(?:powered\s+by|sent\s+(?:via|using|with))\s+\w+',
    r'\b(?:forward\s+(?:this\s+)?(?:to\s+a\s+friend|email))\b',
]

# Service email patterns (common automated services)
SERVICE_EMAIL_PATTERNS = {
    'password_reset': [
        r'\b(?:password\s+reset|reset\s+(?:your\s+)?password)\b',
        r'\b(?:forgot\s+(?:your\s+)?password|password\s+recovery)\b',
        r'\b(?:click\s+(?:here|the\s+link)\s+to\s+reset)\b',
    ],
    'account_verification': [
        r'\b(?:verify\s+(?:your\s+)?(?:email|account)|email\s+verification)\b',
        r'\b(?:confirm\s+(?:your\s+)?(?:email|account|address))\b',
        r'\b(?:activate\s+(?:your\s+)?account)\b',
    ],
    'shipping_notification': [
        r'\b(?:your\s+order\s+(?:has\s+)?(?:shipped|been\s+shipped))\b',
        r'\b(?:tracking\s+(?:number|information|details))\b',
        r'\b(?:package\s+(?:is\s+)?(?:on\s+(?:its|the)\s+way|in\s+transit))\b',
        r'\b(?:estimated\s+delivery|delivery\s+date)\b',
    ],
    'order_confirmation': [
        r'\b(?:order\s+confirm(?:ed|ation)|thank\s+you\s+for\s+your\s+(?:order|purchase))\b',
        r'\b(?:order\s+(?:#|number|id)[:\s]*\w+)\b',
        r'\b(?:receipt|invoice)\s+(?:for|from)\b',
    ],
    'subscription_notification': [
        r'\b(?:subscription\s+(?:confirm|renew|cancel))\b',
        r'\b(?:your\s+(?:free\s+)?trial)\b',
        r'\b(?:billing\s+(?:cycle|period|statement))\b',
    ],
    'newsletter': [
        r'\b(?:weekly|daily|monthly)\s+(?:digest|roundup|update|newsletter)\b',
        r'\b(?:this\s+week(?:.s)?\s+(?:top|best|featured))\b',
        r'\b(?:news\s+(?:from|at)|latest\s+(?:news|updates))\b',
    ],
    'social_notification': [
        r'\b(?:someone\s+(?:commented|liked|shared|followed|mentioned))\b',
        r'\b(?:you\s+have\s+\d+\s+(?:new\s+)?(?:notifications?|messages?|followers?))\b',
        r'\b(?:new\s+(?:comment|like|follow|mention)\s+(?:on|from))\b',
    ],
    'calendar_invite': [
        r'\b(?:invitation|invite)\s+(?:to\s+)?(?:a\s+)?(?:meeting|event|calendar)\b',
        r'\b(?:you(?:.ve|\s+have)\s+been\s+invited)\b',
        r'\b(?:accept|decline|tentative)\b.*\b(?:this\s+)?(?:invitation|meeting)\b',
    ],
}

# Bulk mail header indicators (from email headers)
BULK_HEADER_PATTERNS = [
    r'^List-Unsubscribe:',
    r'^Precedence:\s*(?:bulk|list|junk)',
    r'^X-Mailer:\s*(?:mailchimp|sendgrid|mailgun|constant\s*contact|campaign\s*monitor)',
    r'^X-Campaign',
    r'^X-Mailgun',
    r'^X-SG-',  # SendGrid
    r'^X-MC-',  # Mailchimp
]

# Structural repetition patterns (indicates templated layout)
STRUCTURE_PATTERNS = [
    r'(?:^[-=*]{3,}\s*$)',      # Horizontal dividers
    r'(?:\|\s*[-:]+\s*\|)+',   # Table borders
    r'(?:^\s*[-*•]\s+\S)',      # Bullet points
    r'(?:^\s*\d+[.)]\s+\S)',    # Numbered lists
]


@dataclass
class TemplateFeatures:
    """Template detection features for an email."""
    # Detection results
    is_templated: bool
    template_confidence: float  # 0-1
    template_type: str  # 'service', 'marketing', 'newsletter', 'transactional', 'unknown', 'none'

    # Pattern match counts
    placeholder_count: int
    unsubscribe_signals: int
    marketing_footer_signals: int
    service_pattern_matches: int

    # Detected service types
    detected_services: list[str]  # e.g., ['password_reset', 'newsletter']

    # Structural analysis
    has_bulk_headers: bool
    structural_repetition_score: float  # 0-1

    # LLM analysis (if performed)
    llm_analyzed: bool
    llm_template_type: Optional[str]
    llm_confidence: Optional[float]

    def to_feature_vector(self) -> Union["np.ndarray", list[float]]:
        """Convert to numpy array for ML pipeline.

        Returns 12-dimensional vector:
        - 1 is_templated flag
        - 1 template_confidence
        - 4 pattern counts (normalized)
        - 1 has_bulk_headers flag
        - 1 structural_repetition_score
        - 4 template type one-hot (service, marketing, newsletter, transactional)
        """
        # Template type one-hot encoding
        type_encoding = {
            'service': [1.0, 0.0, 0.0, 0.0],
            'marketing': [0.0, 1.0, 0.0, 0.0],
            'newsletter': [0.0, 0.0, 1.0, 0.0],
            'transactional': [0.0, 0.0, 0.0, 1.0],
            'unknown': [0.25, 0.25, 0.25, 0.25],
            'none': [0.0, 0.0, 0.0, 0.0],
        }

        values = [
            1.0 if self.is_templated else 0.0,
            self.template_confidence,
            min(self.placeholder_count, 10) / 10.0,
            min(self.unsubscribe_signals, 5) / 5.0,
            min(self.marketing_footer_signals, 5) / 5.0,
            min(self.service_pattern_matches, 10) / 10.0,
            1.0 if self.has_bulk_headers else 0.0,
            self.structural_repetition_score,
        ] + type_encoding.get(self.template_type, [0.0, 0.0, 0.0, 0.0])

        if HAS_NUMPY:
            return np.array(values, dtype=np.float32)
        return values


def _count_pattern_matches(text: str, patterns: list[str]) -> int:
    """Count total matches for a list of regex patterns."""
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text, re.IGNORECASE | re.MULTILINE))
    return count


def _detect_placeholders(text: str) -> int:
    """Detect template variable placeholders."""
    count = 0
    for pattern in PLACEHOLDER_PATTERNS:
        count += len(re.findall(pattern, text))
    return count


def _detect_service_patterns(text: str) -> tuple[int, list[str]]:
    """Detect service email patterns and identify types."""
    total_matches = 0
    detected_services = []

    for service_type, patterns in SERVICE_EMAIL_PATTERNS.items():
        matches = _count_pattern_matches(text, patterns)
        if matches > 0:
            total_matches += matches
            detected_services.append(service_type)

    return total_matches, detected_services


def _check_bulk_headers(headers: dict) -> bool:
    """Check for bulk mail indicators in email headers."""
    # Combine relevant headers for checking
    header_text = '\n'.join([
        f"{k}: {v}" for k, v in headers.items()
        if k.lower().startswith(('list-', 'precedence', 'x-'))
    ])

    for pattern in BULK_HEADER_PATTERNS:
        if re.search(pattern, header_text, re.IGNORECASE | re.MULTILINE):
            return True
    return False


def _compute_structural_score(text: str) -> float:
    """Compute structural repetition score based on formatting patterns."""
    if not text:
        return 0.0

    lines = text.split('\n')
    if len(lines) < 3:
        return 0.0

    # Count structural elements
    divider_count = 0
    bullet_count = 0
    numbered_count = 0

    for line in lines:
        if re.match(r'^[-=*]{3,}\s*$', line.strip()):
            divider_count += 1
        if re.match(r'^\s*[-*•]\s+\S', line):
            bullet_count += 1
        if re.match(r'^\s*\d+[.)]\s+\S', line):
            numbered_count += 1

    # Structural score: more dividers and lists = more templated
    line_count = len(lines)
    divider_ratio = min(divider_count / max(line_count / 20, 1), 1.0)
    list_ratio = min((bullet_count + numbered_count) / max(line_count / 5, 1), 1.0)

    return (divider_ratio * 0.4 + list_ratio * 0.6)


def _classify_template_type(
    service_matches: list[str],
    unsubscribe_signals: int,
    marketing_signals: int,
    placeholder_count: int,
) -> str:
    """Classify the template type based on detected patterns."""
    # Service emails (password reset, shipping, etc.)
    if service_matches:
        if any(s in service_matches for s in ['password_reset', 'account_verification',
                                                'shipping_notification', 'order_confirmation']):
            return 'transactional'
        if 'newsletter' in service_matches:
            return 'newsletter'
        return 'service'

    # Marketing emails (heavy on unsubscribe and marketing patterns)
    if unsubscribe_signals >= 2 and marketing_signals >= 2:
        return 'marketing'

    # Newsletter (has unsubscribe but less marketing)
    if unsubscribe_signals >= 1 and marketing_signals >= 1:
        return 'newsletter'

    # Some template patterns but unclear type
    if placeholder_count >= 2 or marketing_signals >= 1:
        return 'unknown'

    return 'none'


def detect_template(
    subject: str,
    body: str,
    *,
    headers: Optional[dict] = None,
    threshold: float = 0.3,
) -> TemplateFeatures:
    """Detect if an email is templated/automated.

    Args:
        subject: Email subject line
        body: Email body text
        headers: Optional email headers dict
        threshold: Confidence threshold for is_templated flag

    Returns:
        TemplateFeatures with detection results
    """
    combined_text = f"{subject}\n{body}"
    headers = headers or {}

    # Detect patterns
    placeholder_count = _detect_placeholders(body)
    unsubscribe_signals = _count_pattern_matches(combined_text, UNSUBSCRIBE_PATTERNS)
    unsubscribe_signals += _count_pattern_matches(combined_text, VIEW_IN_BROWSER_PATTERNS)
    marketing_signals = _count_pattern_matches(combined_text, MARKETING_FOOTER_PATTERNS)
    service_matches, detected_services = _detect_service_patterns(combined_text)
    has_bulk_headers = _check_bulk_headers(headers)
    structural_score = _compute_structural_score(body)

    # Compute confidence score
    confidence = 0.0

    # Strong indicators
    if placeholder_count >= 3:
        confidence += 0.4
    elif placeholder_count >= 1:
        confidence += 0.2

    if unsubscribe_signals >= 2:
        confidence += 0.3
    elif unsubscribe_signals >= 1:
        confidence += 0.15

    if marketing_signals >= 2:
        confidence += 0.2
    elif marketing_signals >= 1:
        confidence += 0.1

    if service_matches >= 2:
        confidence += 0.3
    elif service_matches >= 1:
        confidence += 0.15

    if has_bulk_headers:
        confidence += 0.2

    # Structural repetition
    confidence += structural_score * 0.2

    # Cap at 1.0
    confidence = min(1.0, confidence)

    # Classify template type
    template_type = _classify_template_type(
        detected_services,
        unsubscribe_signals,
        marketing_signals,
        placeholder_count,
    )

    # Determine if templated based on threshold
    is_templated = confidence >= threshold

    return TemplateFeatures(
        is_templated=is_templated,
        template_confidence=confidence,
        template_type=template_type,
        placeholder_count=placeholder_count,
        unsubscribe_signals=unsubscribe_signals,
        marketing_footer_signals=marketing_signals,
        service_pattern_matches=service_matches,
        detected_services=detected_services,
        has_bulk_headers=has_bulk_headers,
        structural_repetition_score=structural_score,
        llm_analyzed=False,
        llm_template_type=None,
        llm_confidence=None,
    )


def detect_template_with_llm(
    subject: str,
    body: str,
    *,
    headers: Optional[dict] = None,
    threshold: float = 0.3,
    llm_threshold: float = 0.5,
    llm_client: Optional[object] = None,
) -> TemplateFeatures:
    """Detect templates with LLM fallback for edge cases.

    Uses regex-based detection first, then falls back to LLM for
    emails with medium confidence (edge cases).

    Args:
        subject: Email subject line
        body: Email body text
        headers: Optional email headers dict
        threshold: Confidence threshold for is_templated flag
        llm_threshold: Confidence below which to trigger LLM analysis
        llm_client: Optional LLM client for edge case analysis

    Returns:
        TemplateFeatures with detection results (possibly LLM-enhanced)
    """
    # First pass: regex-based detection
    features = detect_template(subject, body, headers=headers, threshold=threshold)

    # Check if LLM analysis is needed for edge cases
    needs_llm = (
        llm_client is not None and
        features.template_confidence > 0.1 and  # Some signal present
        features.template_confidence < llm_threshold  # But not confident
    )

    if not needs_llm:
        return features

    # LLM analysis for edge cases
    try:
        llm_result = _analyze_with_llm(subject, body, llm_client)

        if llm_result:
            # Update features with LLM analysis
            features.llm_analyzed = True
            features.llm_template_type = llm_result.get('template_type')
            features.llm_confidence = llm_result.get('confidence', 0.0)

            # Combine regex and LLM confidence
            # Weight LLM more heavily when regex is uncertain
            regex_weight = features.template_confidence
            llm_weight = features.llm_confidence or 0.0
            combined_confidence = (
                regex_weight * 0.4 + llm_weight * 0.6
            )
            features.template_confidence = combined_confidence
            features.is_templated = combined_confidence >= threshold

            # Update template type if LLM provides one
            if features.llm_template_type and features.template_type == 'unknown':
                features.template_type = features.llm_template_type
    except Exception:
        # LLM analysis failed, keep regex results
        pass

    return features


def _analyze_with_llm(
    subject: str,
    body: str,
    llm_client: object,
) -> Optional[dict]:
    """Analyze email with LLM to determine if templated.

    Args:
        subject: Email subject
        body: Email body
        llm_client: LLM client with create() method

    Returns:
        Dict with 'template_type' and 'confidence' or None
    """
    prompt = f"""Analyze this email and determine if it's a templated/automated email.

Subject: {subject[:200]}
Body (first 1000 chars): {body[:1000]}

Classify as one of:
- 'transactional': Password resets, shipping notifications, receipts, order confirmations
- 'marketing': Promotional emails, sales, offers
- 'newsletter': Regular digest/update emails
- 'service': Other automated service emails
- 'none': Personal/human-written email

Respond with JSON only:
{{"template_type": "<type>", "confidence": <0.0-1.0>, "reasoning": "<brief>"}}"""

    try:
        # Assumes llm_client has a method like .messages.create()
        if hasattr(llm_client, 'messages'):
            response = llm_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text
        elif hasattr(llm_client, 'create'):
            response = llm_client.create(prompt=prompt, max_tokens=150)
            content = response.get('content', '')
        else:
            return None

        # Parse JSON response
        import json
        # Extract JSON from response
        json_match = re.search(r'\{[^}]+\}', content)
        if json_match:
            result = json.loads(json_match.group())
            return {
                'template_type': result.get('template_type', 'unknown'),
                'confidence': float(result.get('confidence', 0.5)),
            }
    except Exception:
        pass

    return None


def compute_template_score(features: TemplateFeatures) -> float:
    """Compute template score for priority adjustment.

    Templated emails generally warrant lower priority scores.

    Args:
        features: TemplateFeatures from detect_template()

    Returns:
        Score from 0 to 1 (lower = more templated/automated)
    """
    if not features.is_templated:
        return 1.0  # Not templated = full priority

    # Base score reduction for templated emails
    score = 0.5

    # Type-based adjustments
    type_weights = {
        'transactional': 0.4,  # Still somewhat important (receipts, shipping)
        'service': 0.3,
        'newsletter': 0.2,
        'marketing': 0.1,
        'unknown': 0.3,
    }
    score *= type_weights.get(features.template_type, 0.3)

    # Confidence adjustment
    # High confidence template = lower priority
    score *= (1.0 - features.template_confidence * 0.5)

    # Some templates are still important
    important_services = {'password_reset', 'account_verification',
                          'shipping_notification', 'order_confirmation'}
    if any(s in features.detected_services for s in important_services):
        score = max(score, 0.3)  # Floor for important transactional emails

    return max(0.0, min(1.0, score))


if __name__ == '__main__':
    # Example usage
    sample_subject = "Your order has shipped!"
    sample_body = """
    Dear {{customer_name}},

    Great news! Your order #12345 has been shipped.

    Tracking Number: 1Z999AA10123456784
    Estimated Delivery: January 8, 2026

    Track your package: https://example.com/track/12345

    ---

    You're receiving this email because you made a purchase at Example Store.

    To unsubscribe from shipping notifications, click here.
    © 2026 Example Store. All rights reserved.
    Privacy Policy | Terms of Service
    """

    features = detect_template(sample_subject, sample_body)

    print("Template Detection Results:")
    print(f"  Is templated: {features.is_templated}")
    print(f"  Confidence: {features.template_confidence:.2f}")
    print(f"  Type: {features.template_type}")
    print()
    print("  Pattern matches:")
    print(f"    Placeholders: {features.placeholder_count}")
    print(f"    Unsubscribe signals: {features.unsubscribe_signals}")
    print(f"    Marketing signals: {features.marketing_footer_signals}")
    print(f"    Service patterns: {features.service_pattern_matches}")
    print(f"    Detected services: {features.detected_services}")
    print()
    print(f"  Has bulk headers: {features.has_bulk_headers}")
    print(f"  Structural score: {features.structural_repetition_score:.2f}")
    print()
    print(f"  Template score: {compute_template_score(features):.2f}")
    print(f"  Feature vector ({len(features.to_feature_vector())} dims)")

    # Test with a regular email
    print("\n" + "="*60 + "\n")
    regular_subject = "Quick question about the project"
    regular_body = """
    Hi Sarah,

    Just wanted to follow up on our conversation yesterday.
    Do you have time to meet this week to discuss the proposal?

    Let me know what works for you.

    Best,
    John
    """

    regular_features = detect_template(regular_subject, regular_body)
    print("Regular Email Results:")
    print(f"  Is templated: {regular_features.is_templated}")
    print(f"  Confidence: {regular_features.template_confidence:.2f}")
    print(f"  Type: {regular_features.template_type}")
    print(f"  Template score: {compute_template_score(regular_features):.2f}")
