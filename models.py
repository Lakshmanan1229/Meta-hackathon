# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Email Triage Environment.

The email_triage_env environment simulates an email triage workflow where
an AI agent must classify, prioritize, route, and respond to incoming emails.
"""

from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class EmailTriageAction(Action):
    """Action for the Email Triage environment.

    The agent can perform one action per step on the current email:
    - classify: Assign a category to the email
    - prioritize: Set the priority level
    - route: Route the email to a department
    - respond: Draft a response to the email
    """

    classify: Optional[str] = Field(
        default=None,
        description="Email category: 'spam', 'urgent', 'normal', 'low_priority', 'newsletter'",
    )
    priority: Optional[int] = Field(
        default=None,
        description="Priority level from 1 (highest) to 5 (lowest)",
    )
    route_to: Optional[str] = Field(
        default=None,
        description="Department to route to: 'engineering', 'sales', 'support', 'hr', 'legal', 'marketing', 'finance', 'executive'",
    )
    response_draft: Optional[str] = Field(
        default=None,
        description="Draft response text for the email",
    )


class EmailTriageObservation(Observation):
    """Observation from the Email Triage environment.

    Contains the current email to triage and status information.
    """

    # Current email details
    email_id: str = Field(default="", description="Unique identifier for the current email")
    sender: str = Field(default="", description="Email sender address")
    sender_name: str = Field(default="", description="Display name of the sender")
    subject: str = Field(default="", description="Email subject line")
    body: str = Field(default="", description="Email body content")
    timestamp: str = Field(default="", description="When the email was received")
    has_attachments: bool = Field(default=False, description="Whether email has attachments")
    is_reply: bool = Field(default=False, description="Whether this is a reply to a previous email")
    thread_count: int = Field(default=0, description="Number of emails in the thread")

    # Task context
    task_name: str = Field(default="", description="Current task: 'classification', 'routing', 'full_triage'")
    task_description: str = Field(default="", description="Description of what the agent should do")
    emails_remaining: int = Field(default=0, description="How many emails are left to process")
    emails_processed: int = Field(default=0, description="How many emails have been processed")
    total_emails: int = Field(default=0, description="Total emails in this episode")

    # Feedback from previous action
    feedback: str = Field(default="", description="Feedback on the previous action")
    cumulative_score: float = Field(default=0.0, description="Cumulative score so far")

    # Valid actions hint
    valid_categories: List[str] = Field(
        default_factory=lambda: ["spam", "urgent", "normal", "low_priority", "newsletter"],
        description="Valid classification categories",
    )
    valid_departments: List[str] = Field(
        default_factory=lambda: [
            "engineering", "sales", "support", "hr", "legal", "marketing", "finance", "executive"
        ],
        description="Valid routing departments",
    )
