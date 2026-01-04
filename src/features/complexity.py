#!/usr/bin/env python3
"""Task complexity estimation using LLM with rule-based fallback.

Estimates task complexity on a 4-level scale:
- trivial: Can be done in seconds (quick acknowledgment, simple forward)
- quick: Under 5 minutes (short reply, simple lookup)
- medium: 5-30 minutes (research, multi-step response)
- substantial: 30+ minutes (analysis, report, complex coordination)

Factors considered:
- Scope: How much work is actually being requested?
- Dependencies: Does it require input from others?
- Domain expertise: Does it need specialized knowledge?
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


class ComplexityLevel(str, Enum):
    """Task complexity levels."""
    TRIVIAL = "trivial"      # Seconds - quick acknowledgment, simple forward
    QUICK = "quick"          # <5 min - short reply, simple lookup
    MEDIUM = "medium"        # 5-30 min - research, multi-step response
    SUBSTANTIAL = "substantial"  # 30+ min - analysis, report, coordination

    @property
    def numeric_value(self) -> float:
        """Numeric value for ML pipeline (0-1 scale)."""
        return {
            ComplexityLevel.TRIVIAL: 0.1,
            ComplexityLevel.QUICK: 0.3,
            ComplexityLevel.MEDIUM: 0.6,
            ComplexityLevel.SUBSTANTIAL: 0.9,
        }[self]


@dataclass
class ComplexityEstimate:
    """Result of task complexity estimation."""
    # Primary result
    complexity: ComplexityLevel
    confidence: float  # 0-1

    # Factor analysis
    scope_score: float        # 0-1, higher = larger scope
    dependency_score: float   # 0-1, higher = more dependencies
    expertise_score: float    # 0-1, higher = more expertise needed

    # Method used
    method: str  # 'rule', 'llm', 'combined'

    # Reasoning (if LLM was used)
    reasoning: Optional[str] = None

    def to_feature_vector(self) -> Union["np.ndarray", list[float]]:
        """Convert to feature vector for ML pipeline.

        Returns 8-dimensional vector:
        - 4 complexity one-hot
        - 1 confidence
        - 3 factor scores (scope, dependency, expertise)
        """
        # One-hot encoding for complexity
        complexity_onehot = [0.0, 0.0, 0.0, 0.0]
        idx = list(ComplexityLevel).index(self.complexity)
        complexity_onehot[idx] = 1.0

        values = complexity_onehot + [
            self.confidence,
            self.scope_score,
            self.dependency_score,
            self.expertise_score,
        ]

        if HAS_NUMPY:
            return np.array(values, dtype=np.float32)
        return values


# Scope indicators - what's being asked
TRIVIAL_SCOPE_PATTERNS = [
    r'\b(?:ack(?:nowledge)?|got\s+it|thanks|ty|ok(?:ay)?)\b',
    r'\b(?:forward(?:ed|ing)?)\s+(?:this|the)\b',
    r'\b(?:just\s+)?(?:fyi|for\s+your\s+info(?:rmation)?)\b',
    r'\b(?:see\s+(?:attached|below))\b',
    r'\b(?:noted|understood|will\s+do)\b',
]

QUICK_SCOPE_PATTERNS = [
    r'\b(?:quick\s+(?:question|note|update))\b',
    r'\b(?:can\s+you\s+confirm)\b',
    r'\b(?:let\s+me\s+know(?:\s+(?:if|when))?)\b',
    r'\b(?:check\s+(?:this|on|if))\b',
    r'\b(?:simple|easy|short|brief)\s+(?:request|question|task)\b',
    r'\b(?:just\s+(?:need|want)(?:ed)?)\b',
    r'\b(?:few\s+(?:minutes|words))\b',
]

SUBSTANTIAL_SCOPE_PATTERNS = [
    r'\b(?:comprehensive|detailed|thorough|full)\s+(?:review|analysis|report)\b',
    r'\b(?:prepare|create|draft|write)\s+(?:a\s+)?(?:proposal|report|presentation|document)\b',
    r'\b(?:research|investigate|analyze|evaluate)\b',
    r'\b(?:strategy|strategic|planning|roadmap)\b',
    r'\b(?:implementation|implement|build|develop)\b',
    r'\b(?:all\s+(?:the|of)\s+|every(?:thing)?|complete(?:ly)?)\b',
    r'\b(?:project|initiative|program)\b',
]

# Dependency indicators - requires others
DEPENDENCY_PATTERNS = [
    r'\b(?:after|once|when)\s+(?:you|they|we)\s+(?:get|have|receive)\b',
    r'\b(?:wait(?:ing)?\s+(?:for|on)|pending)\b',
    r'\b(?:depends?\s+on|contingent|blocked\s+(?:by|on))\b',
    r'\b(?:need(?:s)?\s+(?:input|approval|sign[\s-]?off)\s+from)\b',
    r'\b(?:coordinate|collaborate|sync)\s+with\b',
    r'\b(?:loop\s+in|involve|include)\s+\w+\b',
    r'\b(?:get\s+(?:back\s+to\s+)?(?:you|me)\s+(?:after|once))\b',
    r'\b(?:stakeholder|cross[\s-]?functional|multi[\s-]?team)\b',
]

# Expertise indicators - needs specialized knowledge
EXPERTISE_PATTERNS = [
    r'\b(?:technical|architecture|engineering|system)\s+(?:review|design|analysis)\b',
    r'\b(?:legal|compliance|regulatory|policy)\s+(?:review|implications|considerations)\b',
    r'\b(?:financial|budget|forecast|accounting)\s+(?:analysis|review|model)\b',
    r'\b(?:security|privacy|audit)\s+(?:review|assessment|implications)\b',
    r'\b(?:expert(?:ise)?|specialist|SME|subject\s+matter)\b',
    r'\b(?:complex|sophisticated|nuanced|intricate)\b',
    r'\b(?:domain\s+knowledge|specialized|technical\s+depth)\b',
]

# Multi-step indicators
MULTI_STEP_PATTERNS = [
    r'\b(?:step\s+\d|first|then|next|after\s+that|finally)\b',
    r'\b(?:multiple|several|various)\s+(?:steps|tasks|items|parts)\b',
]

# Medium scope patterns (review, feedback, provide input)
MEDIUM_SCOPE_PATTERNS = [
    r'\b(?:please\s+)?review\b',
    r'\b(?:provide|give|share)\s+(?:your\s+)?(?:feedback|input|thoughts|comments)\b',
    r'\b(?:need\s+your\s+input)\b',
    r'\b(?:key\s+)?deliverables?\b',
    r'\b(?:action\s+items?|tasks?)\b',
]


def _count_matches(text: str, patterns: list[str]) -> int:
    """Count total pattern matches in text."""
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text, re.IGNORECASE | re.MULTILINE))
    return count


def _count_bullet_items(text: str) -> int:
    """Count bullet and numbered list items."""
    bullet_pattern = r'^\s*[-•*]\s+\S'
    numbered_pattern = r'^\s*\d+[.)]\s+\S'

    lines = text.split('\n')
    count = 0
    for line in lines:
        if re.match(bullet_pattern, line) or re.match(numbered_pattern, line):
            count += 1
    return count


def _compute_scope_score(text: str) -> float:
    """Compute scope score (0=trivial, 1=substantial)."""
    trivial_count = _count_matches(text, TRIVIAL_SCOPE_PATTERNS)
    quick_count = _count_matches(text, QUICK_SCOPE_PATTERNS)
    medium_count = _count_matches(text, MEDIUM_SCOPE_PATTERNS)
    substantial_count = _count_matches(text, SUBSTANTIAL_SCOPE_PATTERNS)
    multi_step = _count_matches(text, MULTI_STEP_PATTERNS)
    bullet_count = _count_bullet_items(text)

    # Word count as a signal
    word_count = len(text.split())

    # Base score from patterns - priority order matters
    if substantial_count >= 3 or (substantial_count >= 2 and bullet_count >= 3):
        base_score = 0.9
    elif substantial_count >= 2 or (substantial_count >= 1 and bullet_count >= 4):
        base_score = 0.8
    elif substantial_count >= 1 and (medium_count >= 1 or bullet_count >= 2):
        base_score = 0.7
    elif substantial_count >= 1:
        base_score = 0.6
    elif multi_step >= 2 or bullet_count >= 4:
        base_score = 0.6
    elif medium_count >= 2 or (medium_count >= 1 and bullet_count >= 2):
        base_score = 0.5
    elif medium_count >= 1 or bullet_count >= 3:
        base_score = 0.45
    elif multi_step >= 1 or bullet_count >= 2:
        base_score = 0.4
    elif trivial_count >= 2 and quick_count == 0 and medium_count == 0:
        base_score = 0.1
    elif trivial_count >= 1 and word_count < 30:
        base_score = 0.15
    elif quick_count >= 1 and word_count < 80:
        base_score = 0.25
    elif quick_count >= 1:
        base_score = 0.35
    else:
        base_score = 0.35  # Default middle-low

    # Adjust by text length (longer emails often = more scope)
    if word_count > 300:
        base_score = min(1.0, base_score + 0.15)
    elif word_count > 150:
        base_score = min(1.0, base_score + 0.1)
    elif word_count > 80:
        base_score = min(1.0, base_score + 0.05)
    elif word_count < 30:
        base_score = max(0.0, base_score - 0.1)

    return base_score


def _compute_dependency_score(text: str) -> float:
    """Compute dependency score (0=independent, 1=many dependencies)."""
    dep_count = _count_matches(text, DEPENDENCY_PATTERNS)

    if dep_count >= 3:
        return 0.9
    elif dep_count == 2:
        return 0.6
    elif dep_count == 1:
        return 0.3
    else:
        return 0.0


def _compute_expertise_score(text: str) -> float:
    """Compute expertise score (0=no special knowledge, 1=high expertise)."""
    expertise_count = _count_matches(text, EXPERTISE_PATTERNS)

    if expertise_count >= 3:
        return 0.9
    elif expertise_count == 2:
        return 0.6
    elif expertise_count == 1:
        return 0.4
    else:
        return 0.0


def _classify_complexity(
    scope_score: float,
    dependency_score: float,
    expertise_score: float,
) -> tuple[ComplexityLevel, float]:
    """Classify complexity based on factor scores.

    Returns (complexity_level, confidence).
    """
    # Weighted combination
    combined_score = (
        scope_score * 0.5 +
        dependency_score * 0.25 +
        expertise_score * 0.25
    )

    # Map to complexity level
    if combined_score < 0.2:
        level = ComplexityLevel.TRIVIAL
        confidence = 1.0 - combined_score * 5  # More certain at 0
    elif combined_score < 0.4:
        level = ComplexityLevel.QUICK
        distance_from_center = abs(combined_score - 0.3)
        confidence = 1.0 - distance_from_center * 5
    elif combined_score < 0.7:
        level = ComplexityLevel.MEDIUM
        distance_from_center = abs(combined_score - 0.55)
        confidence = 1.0 - distance_from_center * 3
    else:
        level = ComplexityLevel.SUBSTANTIAL
        confidence = min(1.0, combined_score)

    return level, max(0.3, min(1.0, confidence))


def estimate_complexity(
    subject: str,
    body: str,
    *,
    extracted_tasks: Optional[list[dict]] = None,
) -> ComplexityEstimate:
    """Estimate task complexity using rule-based analysis.

    Args:
        subject: Email subject line
        body: Email body text
        extracted_tasks: Optional list of tasks extracted from the email

    Returns:
        ComplexityEstimate with classification and factor scores
    """
    combined_text = f"{subject}\n\n{body}"

    # Compute factor scores
    scope_score = _compute_scope_score(combined_text)
    dependency_score = _compute_dependency_score(combined_text)
    expertise_score = _compute_expertise_score(combined_text)

    # Adjust scope based on number of extracted tasks
    if extracted_tasks:
        task_count = len(extracted_tasks)
        if task_count >= 3:
            scope_score = min(1.0, scope_score + 0.3)
        elif task_count == 2:
            scope_score = min(1.0, scope_score + 0.15)

    # Classify
    complexity, confidence = _classify_complexity(
        scope_score, dependency_score, expertise_score
    )

    return ComplexityEstimate(
        complexity=complexity,
        confidence=confidence,
        scope_score=scope_score,
        dependency_score=dependency_score,
        expertise_score=expertise_score,
        method='rule',
        reasoning=None,
    )


def estimate_complexity_with_llm(
    subject: str,
    body: str,
    *,
    extracted_tasks: Optional[list[dict]] = None,
    llm_client: Optional[object] = None,
    llm_threshold: float = 0.6,
) -> ComplexityEstimate:
    """Estimate complexity with LLM fallback for uncertain cases.

    Uses rule-based estimation first, then falls back to LLM when
    confidence is below threshold.

    Args:
        subject: Email subject line
        body: Email body text
        extracted_tasks: Optional list of tasks extracted from the email
        llm_client: LLM client with messages.create() method
        llm_threshold: Confidence below which to trigger LLM analysis

    Returns:
        ComplexityEstimate with classification and factor scores
    """
    # First pass: rule-based
    estimate = estimate_complexity(
        subject, body, extracted_tasks=extracted_tasks
    )

    # Check if LLM analysis is needed
    needs_llm = (
        llm_client is not None and
        estimate.confidence < llm_threshold
    )

    if not needs_llm:
        return estimate

    # LLM analysis
    try:
        llm_result = _analyze_complexity_with_llm(
            subject, body, extracted_tasks, llm_client
        )

        if llm_result:
            # Combine rule-based and LLM results
            llm_complexity = llm_result['complexity']
            llm_confidence = llm_result['confidence']
            llm_reasoning = llm_result.get('reasoning')

            # Weight LLM more when rule-based is uncertain
            rule_weight = estimate.confidence
            llm_weight = llm_confidence

            # If both agree, high confidence
            if llm_complexity == estimate.complexity:
                final_confidence = min(1.0, (rule_weight + llm_weight) / 2 + 0.2)
                final_complexity = estimate.complexity
            else:
                # Prefer LLM when rule-based is uncertain
                if rule_weight < llm_weight:
                    final_complexity = llm_complexity
                    final_confidence = llm_weight * 0.8
                else:
                    final_complexity = estimate.complexity
                    final_confidence = rule_weight * 0.8

            # Update factor scores based on LLM analysis
            return ComplexityEstimate(
                complexity=final_complexity,
                confidence=final_confidence,
                scope_score=llm_result.get('scope_score', estimate.scope_score),
                dependency_score=llm_result.get('dependency_score', estimate.dependency_score),
                expertise_score=llm_result.get('expertise_score', estimate.expertise_score),
                method='combined',
                reasoning=llm_reasoning,
            )

    except Exception:
        # LLM failed, return rule-based result
        pass

    return estimate


def _analyze_complexity_with_llm(
    subject: str,
    body: str,
    extracted_tasks: Optional[list[dict]],
    llm_client: object,
) -> Optional[dict]:
    """Analyze task complexity using LLM.

    Args:
        subject: Email subject
        body: Email body (truncated)
        extracted_tasks: List of extracted tasks
        llm_client: LLM client

    Returns:
        Dict with complexity analysis or None
    """
    # Format extracted tasks if present
    tasks_str = ""
    if extracted_tasks:
        task_items = [f"- {t.get('description', t)[:100]}" for t in extracted_tasks[:5]]
        tasks_str = f"\n\nExtracted tasks:\n" + "\n".join(task_items)

    prompt = f"""Analyze this email and estimate the complexity of responding/completing the request.

Subject: {subject[:200]}
Body (first 1500 chars): {body[:1500]}{tasks_str}

Classify complexity as ONE of:
- "trivial": Seconds. Simple acknowledgment, forward, FYI.
- "quick": Under 5 min. Short reply, simple lookup, quick confirmation.
- "medium": 5-30 min. Research needed, multi-step response, some back-and-forth.
- "substantial": 30+ min. Deep analysis, report writing, complex coordination.

Consider:
1. SCOPE: How much actual work is requested?
2. DEPENDENCIES: Does it need input/approval from others?
3. EXPERTISE: Does it require specialized domain knowledge?

Respond with JSON only:
{{"complexity": "<level>", "confidence": <0.0-1.0>, "scope_score": <0.0-1.0>, "dependency_score": <0.0-1.0>, "expertise_score": <0.0-1.0>, "reasoning": "<brief explanation>"}}"""

    try:
        if hasattr(llm_client, 'messages'):
            response = llm_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text
        elif hasattr(llm_client, 'create'):
            response = llm_client.create(prompt=prompt, max_tokens=200)
            content = response.get('content', '')
        else:
            return None

        # Parse JSON response
        import json
        json_match = re.search(r'\{[^}]+\}', content)
        if json_match:
            result = json.loads(json_match.group())

            # Convert complexity string to enum
            complexity_str = result.get('complexity', 'medium').lower()
            try:
                complexity = ComplexityLevel(complexity_str)
            except ValueError:
                complexity = ComplexityLevel.MEDIUM

            return {
                'complexity': complexity,
                'confidence': float(result.get('confidence', 0.5)),
                'scope_score': float(result.get('scope_score', 0.5)),
                'dependency_score': float(result.get('dependency_score', 0.0)),
                'expertise_score': float(result.get('expertise_score', 0.0)),
                'reasoning': result.get('reasoning'),
            }

    except Exception:
        pass

    return None


def complexity_to_effort(complexity: ComplexityLevel) -> str:
    """Convert complexity level to legacy effort string.

    For backwards compatibility with existing TaskFeatures.estimated_effort.
    """
    mapping = {
        ComplexityLevel.TRIVIAL: 'quick',
        ComplexityLevel.QUICK: 'quick',
        ComplexityLevel.MEDIUM: 'medium',
        ComplexityLevel.SUBSTANTIAL: 'substantial',
    }
    return mapping[complexity]


if __name__ == '__main__':
    # Example usage
    print("=" * 60)
    print("TASK COMPLEXITY ESTIMATION")
    print("=" * 60)

    # Trivial example
    trivial_subject = "Re: Meeting notes"
    trivial_body = "Got it, thanks!"
    estimate = estimate_complexity(trivial_subject, trivial_body)
    print(f"\n1. Trivial example:")
    print(f"   Subject: {trivial_subject}")
    print(f"   Complexity: {estimate.complexity.value}")
    print(f"   Confidence: {estimate.confidence:.2f}")
    print(f"   Scores: scope={estimate.scope_score:.2f}, dep={estimate.dependency_score:.2f}, exp={estimate.expertise_score:.2f}")

    # Quick example
    quick_subject = "Quick question about the schedule"
    quick_body = """
    Hi,

    Can you confirm if the meeting is still at 3pm tomorrow?

    Thanks!
    """
    estimate = estimate_complexity(quick_subject, quick_body)
    print(f"\n2. Quick example:")
    print(f"   Subject: {quick_subject}")
    print(f"   Complexity: {estimate.complexity.value}")
    print(f"   Confidence: {estimate.confidence:.2f}")
    print(f"   Scores: scope={estimate.scope_score:.2f}, dep={estimate.dependency_score:.2f}, exp={estimate.expertise_score:.2f}")

    # Medium example
    medium_subject = "Need your input on the proposal"
    medium_body = """
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
    estimate = estimate_complexity(medium_subject, medium_body)
    print(f"\n3. Medium example:")
    print(f"   Subject: {medium_subject}")
    print(f"   Complexity: {estimate.complexity.value}")
    print(f"   Confidence: {estimate.confidence:.2f}")
    print(f"   Scores: scope={estimate.scope_score:.2f}, dep={estimate.dependency_score:.2f}, exp={estimate.expertise_score:.2f}")

    # Substantial example
    substantial_subject = "Comprehensive security audit required"
    substantial_body = """
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
    estimate = estimate_complexity(substantial_subject, substantial_body)
    print(f"\n4. Substantial example:")
    print(f"   Subject: {substantial_subject}")
    print(f"   Complexity: {estimate.complexity.value}")
    print(f"   Confidence: {estimate.confidence:.2f}")
    print(f"   Scores: scope={estimate.scope_score:.2f}, dep={estimate.dependency_score:.2f}, exp={estimate.expertise_score:.2f}")

    print(f"\n   Feature vector ({len(estimate.to_feature_vector())} dims)")
