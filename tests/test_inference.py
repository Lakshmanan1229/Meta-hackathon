# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for inference helper functions."""

import json

import pytest

from inference import (
    _build_system_prompt,
    _build_user_prompt,
    _fallback_action,
    _parse_action,
)


class TestBuildSystemPrompt:
    """Tests for _build_system_prompt."""

    def test_classification_prompt(self):
        prompt = _build_system_prompt("classification")
        assert "classify" in prompt.lower()
        assert "priority" in prompt.lower()
        assert "spam" in prompt
        assert "urgent" in prompt

    def test_routing_prompt(self):
        prompt = _build_system_prompt("routing")
        assert "route" in prompt.lower()
        assert "department" in prompt.lower()

    def test_full_triage_prompt(self):
        prompt = _build_system_prompt("full_triage")
        assert "response_draft" in prompt
        assert "classify" in prompt
        assert "route_to" in prompt

    def test_all_prompts_mention_valid_values(self):
        for task in ["classification", "routing", "full_triage"]:
            prompt = _build_system_prompt(task)
            assert "spam" in prompt
            assert "urgent" in prompt
            assert "normal" in prompt


class TestBuildUserPrompt:
    """Tests for _build_user_prompt."""

    def test_basic_prompt(self):
        email_obs = {
            "sender": "test@example.com",
            "sender_name": "Test User",
            "subject": "Test Subject",
            "body": "Test body content",
            "timestamp": "2026-04-01T08:00:00Z",
            "has_attachments": False,
            "is_reply": False,
            "thread_count": 1,
            "emails_processed": 0,
            "total_emails": 5,
            "emails_remaining": 5,
        }
        prompt = _build_user_prompt(email_obs, [], "", 0.0)
        assert "Test User" in prompt
        assert "Test Subject" in prompt
        assert "Test body content" in prompt

    def test_prompt_with_feedback(self):
        email_obs = {
            "sender": "t@t.com",
            "sender_name": "T",
            "subject": "S",
            "body": "B",
            "timestamp": "",
            "has_attachments": False,
            "is_reply": False,
            "thread_count": 1,
            "emails_processed": 1,
            "total_emails": 5,
            "emails_remaining": 4,
        }
        prompt = _build_user_prompt(email_obs, [], "Good job", 0.85)
        assert "Good job" in prompt
        assert "0.85" in prompt

    def test_prompt_includes_thread_count(self):
        email_obs = {
            "sender": "t@t.com",
            "sender_name": "T",
            "subject": "S",
            "body": "B",
            "timestamp": "",
            "has_attachments": False,
            "is_reply": True,
            "thread_count": 5,
            "emails_processed": 0,
            "total_emails": 5,
            "emails_remaining": 5,
        }
        prompt = _build_user_prompt(email_obs, [], "", 0.0)
        assert "5 emails" in prompt


class TestParseAction:
    """Tests for _parse_action."""

    def test_parse_valid_json(self):
        content = '{"classify": "spam", "priority": 5}'
        result = _parse_action(content, "classification")
        assert result["classify"] == "spam"
        assert result["priority"] == 5

    def test_parse_json_with_markdown(self):
        content = '```json\n{"classify": "urgent", "priority": 1}\n```'
        result = _parse_action(content, "classification")
        assert result["classify"] == "urgent"
        assert result["priority"] == 1

    def test_parse_json_with_generic_markdown(self):
        content = '```\n{"classify": "normal", "priority": 3}\n```'
        result = _parse_action(content, "classification")
        assert result["classify"] == "normal"

    def test_parse_json_embedded_in_text(self):
        content = 'Here is my analysis: {"classify": "spam", "priority": 5} Done.'
        result = _parse_action(content, "classification")
        assert result["classify"] == "spam"

    def test_parse_with_routing(self):
        content = '{"classify": "urgent", "priority": 1, "route_to": "engineering"}'
        result = _parse_action(content, "routing")
        assert result["route_to"] == "engineering"

    def test_route_ignored_for_classification_task(self):
        content = '{"classify": "urgent", "priority": 1, "route_to": "engineering"}'
        result = _parse_action(content, "classification")
        assert "route_to" not in result

    def test_response_ignored_for_routing_task(self):
        content = '{"classify": "urgent", "priority": 1, "route_to": "engineering", "response_draft": "hello"}'
        result = _parse_action(content, "routing")
        assert "response_draft" not in result

    def test_parse_full_triage_action(self):
        content = json.dumps({
            "classify": "urgent",
            "priority": 1,
            "route_to": "legal",
            "response_draft": "Acknowledged.",
        })
        result = _parse_action(content, "full_triage")
        assert result["classify"] == "urgent"
        assert result["priority"] == 1
        assert result["route_to"] == "legal"
        assert result["response_draft"] == "Acknowledged."

    def test_parse_invalid_json_fallback(self):
        content = "this is not json at all"
        result = _parse_action(content, "classification")
        assert result == _fallback_action("classification")

    def test_parse_empty_json_fallback(self):
        content = "{}"
        result = _parse_action(content, "classification")
        assert result == _fallback_action("classification")

    def test_parse_normalizes_classify(self):
        content = '{"classify": "  SPAM  ", "priority": 5}'
        result = _parse_action(content, "classification")
        assert result["classify"] == "spam"

    def test_parse_invalid_priority_defaults_to_3(self):
        content = '{"classify": "spam", "priority": "not_a_number"}'
        result = _parse_action(content, "classification")
        assert result["priority"] == 3


class TestFallbackAction:
    """Tests for _fallback_action."""

    def test_classification_fallback(self):
        action = _fallback_action("classification")
        assert action["classify"] == "normal"
        assert action["priority"] == 3
        assert "route_to" not in action
        assert "response_draft" not in action

    def test_routing_fallback(self):
        action = _fallback_action("routing")
        assert action["classify"] == "normal"
        assert action["priority"] == 3
        assert action["route_to"] == "support"
        assert "response_draft" not in action

    def test_full_triage_fallback(self):
        action = _fallback_action("full_triage")
        assert action["classify"] == "normal"
        assert action["priority"] == 3
        assert action["route_to"] == "support"
        assert "response_draft" != ""
