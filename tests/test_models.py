# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Email Triage data models."""

import pytest

from models import EmailTriageAction, EmailTriageObservation


class TestEmailTriageAction:
    """Tests for EmailTriageAction model."""

    def test_default_action_all_none(self):
        """All fields default to None."""
        action = EmailTriageAction()
        assert action.classify is None
        assert action.priority is None
        assert action.route_to is None
        assert action.response_draft is None

    def test_classify_only(self):
        action = EmailTriageAction(classify="spam")
        assert action.classify == "spam"
        assert action.priority is None

    def test_full_action(self):
        action = EmailTriageAction(
            classify="urgent",
            priority=1,
            route_to="engineering",
            response_draft="We are investigating.",
        )
        assert action.classify == "urgent"
        assert action.priority == 1
        assert action.route_to == "engineering"
        assert action.response_draft == "We are investigating."

    def test_action_serialization(self):
        action = EmailTriageAction(classify="normal", priority=3)
        data = action.model_dump()
        assert data["classify"] == "normal"
        assert data["priority"] == 3
        assert data["route_to"] is None
        assert data["response_draft"] is None

    def test_action_from_dict(self):
        data = {"classify": "newsletter", "priority": 5}
        action = EmailTriageAction(**data)
        assert action.classify == "newsletter"
        assert action.priority == 5


class TestEmailTriageObservation:
    """Tests for EmailTriageObservation model."""

    def test_default_observation(self):
        obs = EmailTriageObservation()
        assert obs.email_id == ""
        assert obs.sender == ""
        assert obs.sender_name == ""
        assert obs.subject == ""
        assert obs.body == ""
        assert obs.timestamp == ""
        assert obs.has_attachments is False
        assert obs.is_reply is False
        assert obs.thread_count == 0
        assert obs.task_name == ""
        assert obs.task_description == ""
        assert obs.emails_remaining == 0
        assert obs.emails_processed == 0
        assert obs.total_emails == 0
        assert obs.feedback == ""
        assert obs.cumulative_score == 0.0

    def test_valid_categories_default(self):
        obs = EmailTriageObservation()
        assert "spam" in obs.valid_categories
        assert "urgent" in obs.valid_categories
        assert "normal" in obs.valid_categories
        assert "low_priority" in obs.valid_categories
        assert "newsletter" in obs.valid_categories
        assert len(obs.valid_categories) == 5

    def test_valid_departments_default(self):
        obs = EmailTriageObservation()
        expected = {
            "engineering", "sales", "support", "hr",
            "legal", "marketing", "finance", "executive",
        }
        assert set(obs.valid_departments) == expected

    def test_observation_with_email_data(self):
        obs = EmailTriageObservation(
            email_id="e001",
            sender="test@example.com",
            sender_name="Test User",
            subject="Test Subject",
            body="Test body",
            timestamp="2026-04-01T08:00:00Z",
            has_attachments=True,
            is_reply=True,
            thread_count=3,
            task_name="classification",
            total_emails=5,
        )
        assert obs.email_id == "e001"
        assert obs.sender == "test@example.com"
        assert obs.has_attachments is True
        assert obs.is_reply is True
        assert obs.thread_count == 3

    def test_observation_serialization(self):
        obs = EmailTriageObservation(email_id="e001", subject="Test")
        data = obs.model_dump()
        assert data["email_id"] == "e001"
        assert data["subject"] == "Test"
        assert isinstance(data["valid_categories"], list)
        assert isinstance(data["valid_departments"], list)
