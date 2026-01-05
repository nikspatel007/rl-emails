#!/usr/bin/env python3
"""Tests for feature extraction modules.

Gate 4: Verify sender_frequency, service, task extraction,
and deadline parsing all have passing tests.
"""

import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.task import (
    TaskFeatures,
    ExtractedDeadline,
    ExtractedActionItem,
    extract_tasks,
    compute_task_score,
)
from src.features.people import (
    PeopleFeatures,
    extract_people_features,
    compute_people_score,
)
from src.features.topic import (
    TopicFeatures,
    classify_topic,
    compute_topic_score,
)
from src.features.urgency import (
    UrgencyFeatures,
    compute_email_urgency,
    batch_compute_urgency,
    urgency_to_priority_bucket,
)


class TestDeadlineParsing(unittest.TestCase):
    """Tests for deadline extraction and parsing."""

    def test_today_deadline(self):
        """Test detection of 'today' deadline."""
        features = extract_tasks(
            "Urgent: Need this today",
            "Please complete the report by today."
        )
        self.assertTrue(features.has_deadline)
        self.assertGreater(features.deadline_urgency, 0.8)

    def test_tomorrow_deadline(self):
        """Test detection of 'tomorrow' deadline."""
        features = extract_tasks(
            "Due tomorrow",
            "Submit your review by tomorrow morning."
        )
        self.assertTrue(features.has_deadline)
        self.assertGreater(features.deadline_urgency, 0.6)

    def test_eod_deadline(self):
        """Test detection of EOD acronym deadline."""
        features = extract_tasks(
            "EOD deadline",
            "I need the deliverable by EOD."
        )
        self.assertTrue(features.has_deadline)
        self.assertGreater(features.deadline_urgency, 0.8)

    def test_eob_cob_deadline(self):
        """Test detection of EOB/COB acronym deadlines."""
        for acronym in ['EOB', 'COB', 'EOW', 'EOM']:
            features = extract_tasks(
                f"{acronym} deadline",
                f"Please submit by {acronym}."
            )
            self.assertTrue(features.has_deadline, f"Failed for {acronym}")

    def test_day_name_deadline(self):
        """Test detection of day name deadlines."""
        features = extract_tasks(
            "Due Friday",
            "Can you have this ready by Friday?"
        )
        self.assertTrue(features.has_deadline)
        self.assertIsNotNone(features.deadline_text)
        self.assertIn('friday', features.deadline_text.lower())

    def test_date_format_deadline(self):
        """Test detection of date format deadlines."""
        features = extract_tasks(
            "Due 1/15",
            "The report is due by 1/15/2026."
        )
        self.assertTrue(features.has_deadline)

    def test_month_day_deadline(self):
        """Test detection of month-day format deadlines."""
        features = extract_tasks(
            "January 15th deadline",
            "Submit by Jan 15th please."
        )
        self.assertTrue(features.has_deadline)

    def test_end_of_period_deadline(self):
        """Test detection of 'end of week/month' deadlines."""
        features = extract_tasks(
            "End of week",
            "Please complete by the end of the week."
        )
        self.assertTrue(features.has_deadline)

    def test_no_deadline(self):
        """Test that emails without deadlines are detected correctly."""
        features = extract_tasks(
            "FYI: Project update",
            "Just wanted to share this update with you."
        )
        self.assertFalse(features.has_deadline)
        self.assertEqual(features.deadline_urgency, 0.0)

    def test_multiple_deadlines(self):
        """Test extraction of multiple deadlines."""
        features = extract_tasks(
            "Multiple items due",
            "Report due by Friday. Presentation by EOD Monday. Final review tomorrow."
        )
        self.assertTrue(features.has_deadline)
        self.assertGreaterEqual(len(features.deadlines), 2)


class TestTaskExtraction(unittest.TestCase):
    """Tests for task and action item extraction."""

    def test_please_request(self):
        """Test detection of 'please' requests."""
        features = extract_tasks(
            "Request",
            "Please review the attached document and provide feedback."
        )
        self.assertTrue(len(features.action_items) > 0)
        # 'please' alone gives weak assignment confidence (0.1)
        self.assertGreater(features.assignment_confidence, 0)

    def test_can_you_request(self):
        """Test detection of 'can you' requests."""
        features = extract_tasks(
            "Quick question",
            "Can you send me the latest numbers?"
        )
        self.assertTrue(len(features.action_items) > 0 or features.assignment_confidence > 0)

    def test_could_you_request(self):
        """Test detection of 'could you' requests."""
        features = extract_tasks(
            "Help needed",
            "Could you take a look at this issue?"
        )
        self.assertGreater(features.assignment_confidence, 0)

    def test_need_you_to_request(self):
        """Test detection of 'need you to' assignment."""
        features = extract_tasks(
            "Assignment",
            "I need you to complete this task before the meeting."
        )
        # 'need you to' is detected as action item (assignment type)
        self.assertTrue(len(features.action_items) > 0)
        self.assertIn('need you to', features.action_items[0].lower())

    def test_action_required_explicit(self):
        """Test detection of explicit action required."""
        features = extract_tasks(
            "Action Required",
            "Action item: Review the proposal and sign off."
        )
        self.assertTrue(len(features.action_items) > 0)

    def test_todo_explicit(self):
        """Test detection of explicit TODO."""
        features = extract_tasks(
            "Tasks",
            "TODO: Complete the documentation.\nTODO: Submit for review."
        )
        self.assertTrue(len(features.action_items) > 0)

    def test_deliverable_detection(self):
        """Test detection of deliverables."""
        features = extract_tasks(
            "Request",
            "Please send me the updated spreadsheet."
        )
        self.assertTrue(features.has_deliverable or len(features.action_items) > 0)

    def test_effort_estimation_quick(self):
        """Test quick effort estimation."""
        features = extract_tasks(
            "Quick favor",
            "Can you do a quick review of this?"
        )
        self.assertEqual(features.estimated_effort, 'quick')

    def test_effort_estimation_substantial(self):
        """Test substantial effort estimation."""
        features = extract_tasks(
            "Analysis needed",
            "We need a comprehensive analysis and detailed report."
        )
        self.assertEqual(features.estimated_effort, 'substantial')

    def test_blocker_detection(self):
        """Test blocker detection."""
        features = extract_tasks(
            "Blocker",
            "This is blocking the final presentation. We're waiting on your response."
        )
        self.assertTrue(features.is_blocker_for_others or features.requires_others)

    def test_no_action_items(self):
        """Test email with no action items."""
        features = extract_tasks(
            "FYI",
            "Just keeping you in the loop on this."
        )
        self.assertFalse(features.is_assigned_to_user)
        self.assertLess(features.assignment_confidence, 0.3)

    def test_feature_vector_dimensions(self):
        """Test that feature vector has correct dimensions."""
        features = extract_tasks(
            "Test",
            "Please review this by tomorrow."
        )
        vector = features.to_feature_vector()
        self.assertEqual(len(vector), 12)


class TestSenderFrequency(unittest.TestCase):
    """Tests for sender frequency and people features."""

    def test_emails_from_sender_30d(self):
        """Test sender frequency tracking."""
        email = {
            'from': 'frequent.sender@enron.com',
            'to': 'user@enron.com',
            'cc': '',
            'x_from': 'Frequent Sender',
        }
        user_context = {
            'emails_from_sender_30d': 50,
            'reply_rate_to_sender': 0.8,
            'avg_response_time_hours': 2.0,
            'last_interaction_days': 1,
        }
        features = extract_people_features(email, user_email='user@enron.com', user_context=user_context)
        self.assertEqual(features.emails_from_sender_30d, 50)

    def test_frequent_sender_penalty(self):
        """Test that very frequent senders get lower importance."""
        email = {
            'from': 'spammer@enron.com',
            'to': 'user@enron.com',
            'cc': '',
            'x_from': 'Spammer',
        }
        # High volume sender
        high_volume_context = {
            'emails_from_sender_30d': 100,
            'reply_rate_to_sender': 0.1,
            'avg_response_time_hours': 48.0,
            'last_interaction_days': 30,
        }
        features_high = extract_people_features(email, user_context=high_volume_context)

        # Low volume sender
        low_volume_context = {
            'emails_from_sender_30d': 5,
            'reply_rate_to_sender': 0.8,
            'avg_response_time_hours': 2.0,
            'last_interaction_days': 1,
        }
        features_low = extract_people_features(email, user_context=low_volume_context)

        # Low volume should have higher relationship strength
        self.assertGreater(features_low.relationship_strength, features_high.relationship_strength)

    def test_internal_sender_detection(self):
        """Test internal vs external sender detection."""
        internal_email = {'from': 'john@enron.com', 'to': '', 'cc': '', 'x_from': ''}
        external_email = {'from': 'client@external.com', 'to': '', 'cc': '', 'x_from': ''}

        internal_features = extract_people_features(internal_email)
        external_features = extract_people_features(external_email)

        self.assertTrue(internal_features.sender_is_internal)
        self.assertFalse(external_features.sender_is_internal)

    def test_automated_sender_detection(self):
        """Test automated sender detection."""
        automated_emails = [
            {'from': 'noreply@company.com', 'to': '', 'cc': '', 'x_from': ''},
            {'from': 'notifications@system.com', 'to': '', 'cc': '', 'x_from': ''},
            {'from': 'alert@alerts.enron.com', 'to': '', 'cc': '', 'x_from': 'Automated System'},
        ]
        for email in automated_emails:
            features = extract_people_features(email)
            self.assertTrue(features.sender_is_automated, f"Failed for {email['from']}")

    def test_executive_sender_detection(self):
        """Test executive/manager level detection."""
        exec_email = {'from': 'ceo@enron.com', 'to': '', 'cc': '', 'x_from': 'John Smith, CEO'}
        features = extract_people_features(exec_email)
        self.assertEqual(features.sender_org_level, 3)

    def test_manager_sender_detection(self):
        """Test manager level detection."""
        mgr_email = {'from': 'boss@enron.com', 'to': '', 'cc': '', 'x_from': 'Jane Doe, Manager'}
        features = extract_people_features(mgr_email)
        self.assertEqual(features.sender_org_level, 2)

    def test_reply_rate_scoring(self):
        """Test reply rate affects relationship strength."""
        email = {'from': 'colleague@enron.com', 'to': '', 'cc': '', 'x_from': ''}

        high_reply = {'reply_rate_to_sender': 0.9, 'avg_response_time_hours': 1.0,
                      'last_interaction_days': 1, 'emails_from_sender_30d': 20}
        low_reply = {'reply_rate_to_sender': 0.1, 'avg_response_time_hours': 48.0,
                     'last_interaction_days': 60, 'emails_from_sender_30d': 2}

        high_features = extract_people_features(email, user_context=high_reply)
        low_features = extract_people_features(email, user_context=low_reply)

        self.assertGreater(high_features.relationship_strength, low_features.relationship_strength)

    def test_feature_vector_dimensions(self):
        """Test that feature vector has correct dimensions."""
        email = {'from': 'test@enron.com', 'to': 'user@enron.com', 'cc': '', 'x_from': ''}
        features = extract_people_features(email)
        vector = features.to_feature_vector()
        self.assertEqual(len(vector), 15)


class TestTopicClassification(unittest.TestCase):
    """Tests for topic and service classification."""

    def test_meeting_scheduling(self):
        """Test meeting scheduling topic detection."""
        features = classify_topic(
            "Meeting request",
            "Can we schedule a meeting to discuss the project? Let me know your availability."
        )
        self.assertTrue(features.is_meeting_request)
        self.assertIn('meeting_scheduling', features.topic_distribution)

    def test_project_update(self):
        """Test project update detection."""
        features = classify_topic(
            "Weekly Status Update",
            "Here's the weekly progress report. All milestones completed."
        )
        self.assertTrue(features.is_status_update)

    def test_task_assignment_topic(self):
        """Test task assignment topic detection."""
        features = classify_topic(
            "Action Required",
            "Please review and complete the attached assignment."
        )
        self.assertTrue(features.is_action_request)

    def test_fyi_detection(self):
        """Test FYI-only email detection."""
        features = classify_topic(
            "FYI",
            "FYI - just wanted to share this with you. No action needed."
        )
        self.assertTrue(features.is_fyi_only)

    def test_decision_request(self):
        """Test decision request detection."""
        features = classify_topic(
            "Need your approval",
            "Please approve this proposal. We need your decision to proceed."
        )
        self.assertTrue(features.is_decision_needed)

    def test_problem_report(self):
        """Test problem/issue detection."""
        features = classify_topic(
            "Critical Issue",
            "We have a critical problem with the system. It's broken and needs urgent fix."
        )
        self.assertIn('problem_report', features.topic_distribution)
        self.assertTrue(features.is_escalation)

    def test_external_communication(self):
        """Test external/service communication detection."""
        features = classify_topic(
            "Client Proposal",
            "Attached is the proposal for our new client. Please review the contract terms."
        )
        self.assertIn('external_communication', features.topic_distribution)

    def test_question_detection(self):
        """Test question detection."""
        features = classify_topic(
            "Question",
            "What is the status of the project? When will it be ready?"
        )
        self.assertTrue(features.is_question)

    def test_urgency_detection(self):
        """Test urgency score calculation."""
        urgent_features = classify_topic(
            "URGENT: Critical deadline",
            "This is urgent! We need this ASAP. Critical priority."
        )
        normal_features = classify_topic(
            "Update",
            "Here's a regular update on the project."
        )
        self.assertGreater(urgent_features.urgency_score, normal_features.urgency_score)

    def test_sentiment_positive(self):
        """Test positive sentiment detection."""
        features = classify_topic(
            "Great job!",
            "Thanks for your excellent work. Great job on the presentation!"
        )
        self.assertGreater(features.sentiment_score, 0)

    def test_sentiment_negative(self):
        """Test negative sentiment detection."""
        features = classify_topic(
            "Issue",
            "Unfortunately we have a problem. I'm disappointed with the results."
        )
        self.assertLess(features.sentiment_score, 0)

    def test_topic_distribution_sums_to_one(self):
        """Test that topic distribution probabilities sum to 1."""
        features = classify_topic(
            "Test",
            "This is a test email with various content."
        )
        total = sum(features.topic_distribution.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_feature_vector_dimensions(self):
        """Test that feature vector has correct dimensions."""
        features = classify_topic("Test", "Test email body.")
        vector = features.to_feature_vector()
        self.assertEqual(len(vector), 20)


class TestComputeScores(unittest.TestCase):
    """Tests for score computation functions."""

    def test_task_score_with_deadline(self):
        """Test task score increases with deadline."""
        with_deadline = extract_tasks("Due today", "Complete by EOD.")
        without_deadline = extract_tasks("FYI", "Just sharing this info.")

        self.assertGreater(
            compute_task_score(with_deadline),
            compute_task_score(without_deadline)
        )

    def test_task_score_with_assignment(self):
        """Test task score increases with assignment."""
        assigned = extract_tasks("Request", "I need you to complete this task.")
        not_assigned = extract_tasks("FYI", "Just an update.")

        self.assertGreater(
            compute_task_score(assigned),
            compute_task_score(not_assigned)
        )

    def test_people_score_automated_low(self):
        """Test automated senders get low score."""
        email = {'from': 'noreply@system.com', 'to': '', 'cc': '', 'x_from': 'Automated'}
        features = extract_people_features(email)
        score = compute_people_score(features)
        self.assertLessEqual(score, 0.2)

    def test_people_score_executive_high(self):
        """Test executives get high score."""
        email = {'from': 'ceo@enron.com', 'to': 'user@enron.com', 'cc': '', 'x_from': 'CEO John'}
        features = extract_people_features(email, user_email='user@enron.com')
        score = compute_people_score(features)
        self.assertGreater(score, 0.5)

    def test_topic_score_action_request(self):
        """Test action requests get higher topic score."""
        features = classify_topic(
            "Action Required",
            "Please complete this task immediately."
        )
        score = compute_topic_score(features)
        self.assertGreater(score, 0.3)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple feature extractors."""

    def test_complex_email_all_features(self):
        """Test extraction from a complex email with multiple signals."""
        subject = "URGENT: Action Required - Project deadline tomorrow"
        body = """
        Hi Team,

        This is urgent and needs immediate attention. I need you to complete
        the following by EOD tomorrow:

        1. Review the attached proposal
        2. Send me the updated budget spreadsheet
        3. Prepare the presentation for the client meeting

        This is blocking our ability to proceed with the board presentation.
        Please confirm receipt and your availability.

        Thanks,
        John Smith, VP Operations
        """

        task_features = extract_tasks(subject, body)
        topic_features = classify_topic(subject, body)
        email_dict = {
            'from': 'john.smith@enron.com',
            'to': 'team@enron.com',
            'cc': '',
            'x_from': 'John Smith, VP Operations',
        }
        people_features = extract_people_features(email_dict)

        # Verify deadline detection
        self.assertTrue(task_features.has_deadline)
        self.assertGreater(task_features.deadline_urgency, 0.5)

        # Verify action items
        self.assertTrue(len(task_features.action_items) > 0 or task_features.is_assigned_to_user)

        # Verify urgency
        self.assertGreater(topic_features.urgency_score, 0)

        # Verify sender level (VP = executive/manager)
        self.assertGreaterEqual(people_features.sender_org_level, 2)


class TestUrgencyScoring(unittest.TestCase):
    """Tests for email-level urgency scoring."""

    def test_urgent_keywords_detected(self):
        """Test that urgent keywords are detected."""
        email = {
            'from': 'sender@enron.com',
            'to': 'user@enron.com',
            'subject': 'URGENT: Need response',
            'body': 'This is critical and needs immediate attention.',
        }
        urgency = compute_email_urgency(email)
        self.assertGreater(urgency.keyword_urgency, 0.8)
        self.assertIn('urgent', urgency.detected_keywords)
        self.assertIn('critical', urgency.detected_keywords)

    def test_asap_keyword(self):
        """Test ASAP keyword detection."""
        email = {
            'from': 'sender@enron.com',
            'to': 'user@enron.com',
            'subject': 'Need this ASAP',
            'body': 'Please send the report ASAP.',
        }
        urgency = compute_email_urgency(email)
        self.assertGreater(urgency.keyword_urgency, 0.8)
        self.assertIn('asap', urgency.detected_keywords)

    def test_deadline_affects_urgency(self):
        """Test that deadlines increase urgency."""
        with_deadline = {
            'from': 'sender@enron.com',
            'to': 'user@enron.com',
            'subject': 'Report needed',
            'body': 'Please complete by EOD today.',
        }
        without_deadline = {
            'from': 'sender@enron.com',
            'to': 'user@enron.com',
            'subject': 'FYI',
            'body': 'Just sharing this update.',
        }
        urgency_with = compute_email_urgency(with_deadline)
        urgency_without = compute_email_urgency(without_deadline)

        self.assertGreater(urgency_with.deadline_urgency, urgency_without.deadline_urgency)
        self.assertTrue(urgency_with.has_explicit_deadline)
        self.assertFalse(urgency_without.has_explicit_deadline)

    def test_executive_sender_increases_urgency(self):
        """Test that executive senders get higher urgency."""
        from_exec = {
            'from': 'ceo@enron.com',
            'to': 'user@enron.com',
            'x_from': 'John Smith, CEO',
            'subject': 'Quick question',
            'body': 'Can you look into this?',
        }
        from_peer = {
            'from': 'colleague@enron.com',
            'to': 'user@enron.com',
            'x_from': 'Jane Doe',
            'subject': 'Quick question',
            'body': 'Can you look into this?',
        }
        urgency_exec = compute_email_urgency(from_exec)
        urgency_peer = compute_email_urgency(from_peer)

        self.assertEqual(urgency_exec.sender_org_level, 3)
        self.assertLess(urgency_peer.sender_org_level, 3)
        self.assertGreater(urgency_exec.overall_urgency, urgency_peer.overall_urgency)

    def test_no_urgency_fyi(self):
        """Test that FYI emails have low urgency."""
        email = {
            'from': 'sender@enron.com',
            'to': 'user@enron.com',
            'subject': 'FYI',
            'body': 'Just keeping you in the loop. No action needed.',
        }
        urgency = compute_email_urgency(email)
        self.assertEqual(urgency.keyword_urgency, 0.0)
        self.assertEqual(len(urgency.detected_keywords), 0)
        self.assertLess(urgency.overall_urgency, 0.5)

    def test_priority_buckets(self):
        """Test priority bucket assignment."""
        self.assertEqual(urgency_to_priority_bucket(0.9), 'critical')
        self.assertEqual(urgency_to_priority_bucket(0.7), 'high')
        self.assertEqual(urgency_to_priority_bucket(0.5), 'medium')
        self.assertEqual(urgency_to_priority_bucket(0.2), 'low')

    def test_feature_vector_dimensions(self):
        """Test that urgency feature vector has correct dimensions."""
        email = {
            'from': 'sender@enron.com',
            'to': 'user@enron.com',
            'subject': 'Test',
            'body': 'Test email.',
        }
        urgency = compute_email_urgency(email)
        vector = urgency.to_feature_vector()
        self.assertEqual(len(vector), 8)

    def test_batch_compute(self):
        """Test batch urgency computation."""
        emails = [
            {'from': 'a@enron.com', 'to': 'user@enron.com', 'subject': 'URGENT', 'body': 'Critical'},
            {'from': 'b@enron.com', 'to': 'user@enron.com', 'subject': 'FYI', 'body': 'Update'},
            {'from': 'c@enron.com', 'to': 'user@enron.com', 'subject': 'Question', 'body': 'Info?'},
        ]
        results = batch_compute_urgency(emails)
        self.assertEqual(len(results), 3)
        # First email should have highest urgency
        self.assertGreater(results[0].overall_urgency, results[1].overall_urgency)


if __name__ == '__main__':
    unittest.main()
