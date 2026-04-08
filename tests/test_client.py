# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Email Triage client."""

import importlib
import importlib.util
import sys
import types

import pytest

from models import EmailTriageAction, EmailTriageObservation

# client.py uses relative imports (.models) which fail when imported as a
# top-level module. Create a fake parent package so the relative import works.
import models as _models_mod

_pkg = types.ModuleType("email_triage_env")
_pkg.__path__ = ["."]
sys.modules["email_triage_env"] = _pkg
sys.modules["email_triage_env.models"] = _models_mod

# Load client.py directly as email_triage_env.client so relative imports work
_spec = importlib.util.spec_from_file_location(
    "email_triage_env.client",
    "client.py",
)
_client_mod = importlib.util.module_from_spec(_spec)
_client_mod.__package__ = "email_triage_env"
sys.modules["email_triage_env.client"] = _client_mod
_spec.loader.exec_module(_client_mod)

EmailTriageEnv = _client_mod.EmailTriageEnv


class TestStepPayload:
    """Tests for _step_payload serialization."""

    def test_full_action_payload(self):
        env = EmailTriageEnv.__new__(EmailTriageEnv)
        action = EmailTriageAction(
            classify="spam",
            priority=5,
            route_to="support",
            response_draft="This is spam.",
        )
        payload = env._step_payload(action)
        assert payload == {
            "classify": "spam",
            "priority": 5,
            "route_to": "support",
            "response_draft": "This is spam.",
        }

    def test_partial_action_payload(self):
        env = EmailTriageEnv.__new__(EmailTriageEnv)
        action = EmailTriageAction(classify="urgent", priority=1)
        payload = env._step_payload(action)
        assert payload == {"classify": "urgent", "priority": 1}
        assert "route_to" not in payload
        assert "response_draft" not in payload

    def test_empty_action_payload(self):
        env = EmailTriageEnv.__new__(EmailTriageEnv)
        action = EmailTriageAction()
        payload = env._step_payload(action)
        assert payload == {}


class TestParseResult:
    """Tests for _parse_result deserialization."""

    def test_parse_full_result(self):
        env = EmailTriageEnv.__new__(EmailTriageEnv)
        payload = {
            "observation": {
                "email_id": "e001",
                "sender": "test@example.com",
                "sender_name": "Test",
                "subject": "Test Subject",
                "body": "Test body",
                "timestamp": "2026-04-01T08:00:00Z",
                "has_attachments": False,
                "is_reply": False,
                "thread_count": 1,
                "task_name": "classification",
                "task_description": "Classify emails",
                "emails_remaining": 4,
                "emails_processed": 0,
                "total_emails": 5,
                "feedback": "Process the first email.",
                "cumulative_score": 0.0,
            },
            "done": False,
            "reward": 0.0,
        }
        result = env._parse_result(payload)
        assert result.observation.email_id == "e001"
        assert result.observation.sender == "test@example.com"
        assert result.observation.task_name == "classification"
        assert result.done is False
        assert result.reward == 0.0

    def test_parse_result_with_defaults(self):
        env = EmailTriageEnv.__new__(EmailTriageEnv)
        payload = {"observation": {}, "done": True, "reward": 0.8}
        result = env._parse_result(payload)
        assert result.observation.email_id == ""
        assert result.observation.sender == ""
        assert result.done is True
        assert result.reward == 0.8

    def test_parse_result_missing_observation(self):
        env = EmailTriageEnv.__new__(EmailTriageEnv)
        payload = {"done": False, "reward": 0.0}
        result = env._parse_result(payload)
        assert result.observation.email_id == ""


class TestParseState:
    """Tests for _parse_state deserialization."""

    def test_parse_state(self):
        env = EmailTriageEnv.__new__(EmailTriageEnv)
        payload = {"episode_id": "abc-123", "step_count": 5}
        state = env._parse_state(payload)
        assert state.episode_id == "abc-123"
        assert state.step_count == 5

    def test_parse_state_defaults(self):
        env = EmailTriageEnv.__new__(EmailTriageEnv)
        payload = {}
        state = env._parse_state(payload)
        assert state.episode_id is None
        assert state.step_count == 0
