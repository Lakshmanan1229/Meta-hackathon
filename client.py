# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Email Triage Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import EmailTriageAction, EmailTriageObservation


class EmailTriageEnv(
    EnvClient[EmailTriageAction, EmailTriageObservation, State]
):
    """
    Client for the Email Triage Environment.

    Example:
        >>> with EmailTriageEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.subject)
        ...
        ...     result = client.step(EmailTriageAction(classify="spam", priority=5))
        ...     print(result.observation.feedback)
    """

    def _step_payload(self, action: EmailTriageAction) -> Dict:
        """Convert EmailTriageAction to JSON payload."""
        payload = {}
        if action.classify is not None:
            payload["classify"] = action.classify
        if action.priority is not None:
            payload["priority"] = action.priority
        if action.route_to is not None:
            payload["route_to"] = action.route_to
        if action.response_draft is not None:
            payload["response_draft"] = action.response_draft
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[EmailTriageObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", {})
        observation = EmailTriageObservation(
            email_id=obs_data.get("email_id", ""),
            sender=obs_data.get("sender", ""),
            sender_name=obs_data.get("sender_name", ""),
            subject=obs_data.get("subject", ""),
            body=obs_data.get("body", ""),
            timestamp=obs_data.get("timestamp", ""),
            has_attachments=obs_data.get("has_attachments", False),
            is_reply=obs_data.get("is_reply", False),
            thread_count=obs_data.get("thread_count", 0),
            task_name=obs_data.get("task_name", ""),
            task_description=obs_data.get("task_description", ""),
            emails_remaining=obs_data.get("emails_remaining", 0),
            emails_processed=obs_data.get("emails_processed", 0),
            total_emails=obs_data.get("total_emails", 0),
            feedback=obs_data.get("feedback", ""),
            cumulative_score=obs_data.get("cumulative_score", 0.0),
            valid_categories=obs_data.get(
                "valid_categories",
                ["spam", "urgent", "normal", "low_priority", "newsletter"],
            ),
            valid_departments=obs_data.get(
                "valid_departments",
                ["engineering", "sales", "support", "hr", "legal", "marketing", "finance", "executive"],
            ),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
