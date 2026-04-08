# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Email Triage Environment Implementation.

A real-world email triage environment where an AI agent must classify,
prioritize, route, and respond to incoming emails. Supports 3 tasks
with increasing difficulty:
  - classification (easy): Classify emails by category and priority
  - routing (medium): Classify and route to correct department
  - full_triage (hard): Classify, route, and draft appropriate responses
"""

from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import EmailTriageAction, EmailTriageObservation
    from ..email_dataset import get_emails_for_task
except ImportError:
    from models import EmailTriageAction, EmailTriageObservation
    from email_dataset import get_emails_for_task


VALID_CATEGORIES = {"spam", "urgent", "normal", "low_priority", "newsletter"}
VALID_DEPARTMENTS = {
    "engineering", "sales", "support", "hr", "legal",
    "marketing", "finance", "executive",
}

# Scoring weights per task
TASK_WEIGHTS = {
    "classification": {"classify": 0.6, "priority": 0.4, "route": 0.0, "response": 0.0},
    "routing": {"classify": 0.3, "priority": 0.2, "route": 0.5, "response": 0.0},
    "full_triage": {"classify": 0.15, "priority": 0.1, "route": 0.25, "response": 0.5},
}


def _score_classification(predicted: Optional[str], ground_truth: str) -> float:
    """Score category classification (0.0 - 1.0)."""
    if predicted is None:
        return 0.0
    predicted = predicted.strip().lower()
    if predicted not in VALID_CATEGORIES:
        return 0.0
    if predicted == ground_truth:
        return 1.0
    # Partial credit for related categories
    related = {
        ("spam", "low_priority"): 0.2,
        ("low_priority", "spam"): 0.2,
        ("urgent", "normal"): 0.3,
        ("normal", "urgent"): 0.3,
        ("normal", "low_priority"): 0.3,
        ("low_priority", "normal"): 0.3,
        ("newsletter", "low_priority"): 0.4,
        ("low_priority", "newsletter"): 0.4,
        ("newsletter", "spam"): 0.2,
        ("spam", "newsletter"): 0.2,
    }
    return related.get((predicted, ground_truth), 0.0)


def _score_priority(predicted: Optional[int], ground_truth: int) -> float:
    """Score priority assignment (0.0 - 1.0)."""
    if predicted is None:
        return 0.0
    if not (1 <= predicted <= 5):
        return 0.0
    if predicted == ground_truth:
        return 1.0
    # Linear decay based on distance
    distance = abs(predicted - ground_truth)
    return max(0.0, 1.0 - distance * 0.25)


def _score_routing(predicted: Optional[str], ground_truth: str) -> float:
    """Score department routing (0.0 - 1.0)."""
    if predicted is None:
        return 0.0
    predicted = predicted.strip().lower()
    if predicted not in VALID_DEPARTMENTS:
        return 0.0
    if predicted == ground_truth:
        return 1.0
    # Partial credit for related departments
    related = {
        ("engineering", "support"): 0.3,
        ("support", "engineering"): 0.3,
        ("sales", "marketing"): 0.3,
        ("marketing", "sales"): 0.3,
        ("legal", "executive"): 0.3,
        ("executive", "legal"): 0.3,
        ("hr", "executive"): 0.2,
        ("executive", "hr"): 0.2,
        ("finance", "executive"): 0.2,
        ("executive", "finance"): 0.2,
    }
    return related.get((predicted, ground_truth), 0.0)


def _score_response(
    response: Optional[str],
    keywords: List[str],
    expected_tone: str,
) -> float:
    """Score response draft quality (0.0 - 1.0)."""
    if response is None or not response.strip():
        return 0.0

    response_lower = response.lower().strip()

    # If no response is expected (e.g., newsletters), penalize responding
    if expected_tone == "none":
        return 0.3 if len(response_lower) < 20 else 0.1

    score = 0.0

    # Keyword coverage (60% of response score)
    if keywords:
        matched = sum(1 for kw in keywords if kw.lower() in response_lower)
        keyword_ratio = matched / len(keywords)
        score += 0.6 * keyword_ratio

    # Length appropriateness (20% of response score)
    word_count = len(response_lower.split())
    if 20 <= word_count <= 200:
        score += 0.2
    elif 10 <= word_count < 20 or 200 < word_count <= 300:
        score += 0.1
    elif word_count < 5:
        score += 0.0
    else:
        score += 0.05

    # Tone indicators (20% of response score)
    tone_markers = {
        "professional": ["regards", "thank", "please", "sincerely", "best"],
        "formal": ["dear", "pursuant", "regards", "respectfully", "sincerely"],
        "warm": ["congratulations", "happy", "glad", "welcome", "appreciate"],
        "warning": ["do not", "caution", "suspicious", "report", "phishing", "scam"],
        "careful": ["investigating", "take seriously", "reviewing", "respond"],
        "casual": ["hey", "sounds", "great", "sure", "cool", "thanks"],
    }
    markers = tone_markers.get(expected_tone, [])
    if markers:
        tone_matches = sum(1 for m in markers if m in response_lower)
        tone_ratio = min(tone_matches / max(len(markers) * 0.4, 1), 1.0)
        score += 0.2 * tone_ratio

    return min(score, 1.0)


class EmailTriageEnvironment(Environment):
    """
    Email Triage Environment.

    An AI agent processes a queue of emails, making triage decisions for each one.
    Three difficulty levels test different aspects of email management:

    - classification (easy): Classify emails and assign priority
    - routing (medium): Classify, prioritize, and route to departments
    - full_triage (hard): Full triage including drafting responses

    Reward function provides partial credit for partially correct actions,
    with penalties for invalid actions and bonuses for streaks of correct decisions.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task: str = "classification"):
        """Initialize the email triage environment.

        Args:
            task: One of 'classification', 'routing', 'full_triage'
        """
        self._task = task if task in TASK_WEIGHTS else "classification"
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._emails: List[Dict[str, Any]] = []
        self._current_idx: int = 0
        self._scores: List[float] = []
        self._cumulative_score: float = 0.0
        self._done: bool = False
        self._streak: int = 0
        self._invalid_action_count: int = 0

    def reset(self) -> EmailTriageObservation:
        """Reset the environment and load emails for the current task."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._emails = get_emails_for_task(self._task)
        self._current_idx = 0
        self._scores = []
        self._cumulative_score = 0.0
        self._done = False
        self._streak = 0
        self._invalid_action_count = 0

        if not self._emails:
            return EmailTriageObservation(
                done=True,
                reward=0.0,
                feedback="No emails available for this task.",
                task_name=self._task,
            )

        email = self._emails[0]
        task_descriptions = {
            "classification": (
                "Classify each email into a category (spam, urgent, normal, "
                "low_priority, newsletter) and assign a priority level (1-5). "
                "Set 'classify' and 'priority' fields in your action."
            ),
            "routing": (
                "Classify each email, assign priority, and route it to the "
                "correct department. Set 'classify', 'priority', and 'route_to' "
                "fields in your action."
            ),
            "full_triage": (
                "Perform full email triage: classify the email, assign priority, "
                "route to the correct department, and draft an appropriate response. "
                "Set all four fields: 'classify', 'priority', 'route_to', and "
                "'response_draft' in your action."
            ),
        }

        return EmailTriageObservation(
            email_id=email["email_id"],
            sender=email["sender"],
            sender_name=email["sender_name"],
            subject=email["subject"],
            body=email["body"],
            timestamp=email["timestamp"],
            has_attachments=email["has_attachments"],
            is_reply=email["is_reply"],
            thread_count=email["thread_count"],
            task_name=self._task,
            task_description=task_descriptions.get(self._task, ""),
            emails_remaining=len(self._emails) - 1,
            emails_processed=0,
            total_emails=len(self._emails),
            feedback="Environment reset. Process the first email.",
            cumulative_score=0.0,
            done=False,
            reward=0.0,
        )

    def step(self, action: EmailTriageAction) -> EmailTriageObservation:  # type: ignore[override]
        """Process the agent's triage action on the current email.

        Args:
            action: EmailTriageAction with classification, priority, routing, and/or response

        Returns:
            EmailTriageObservation with next email or completion status
        """
        self._state.step_count += 1

        if self._done:
            return EmailTriageObservation(
                done=True,
                reward=0.0,
                feedback="Episode already completed.",
                task_name=self._task,
                cumulative_score=self._cumulative_score,
                emails_processed=len(self._emails),
                total_emails=len(self._emails),
            )

        if self._current_idx >= len(self._emails):
            self._done = True
            return EmailTriageObservation(
                done=True,
                reward=0.0,
                feedback="All emails processed.",
                task_name=self._task,
                cumulative_score=self._cumulative_score,
                emails_processed=len(self._emails),
                total_emails=len(self._emails),
            )

        # Score the action
        email = self._emails[self._current_idx]
        gt = email["ground_truth"]
        weights = TASK_WEIGHTS[self._task]

        # Check for completely empty/invalid action
        has_any_action = (
            action.classify is not None
            or action.priority is not None
            or action.route_to is not None
            or action.response_draft is not None
        )

        feedback_parts = []

        if not has_any_action:
            step_score = 0.0
            self._invalid_action_count += 1
            self._streak = 0
            feedback_parts.append("No action fields provided. Please set at least one field.")
            # Penalize repeated invalid actions
            if self._invalid_action_count >= 3:
                step_score = -0.1
                feedback_parts.append("Repeated empty actions detected. Penalty applied.")
        else:
            # Score each component
            cls_score = _score_classification(action.classify, gt["category"])
            pri_score = _score_priority(action.priority, gt["priority"])
            rte_score = _score_routing(action.route_to, gt["department"])
            rsp_score = _score_response(
                action.response_draft, gt["response_keywords"], gt["response_tone"]
            )

            # Weighted score
            step_score = (
                weights["classify"] * cls_score
                + weights["priority"] * pri_score
                + weights["route"] * rte_score
                + weights["response"] * rsp_score
            )

            # Feedback
            if weights["classify"] > 0:
                if cls_score == 1.0:
                    feedback_parts.append(f"Classification: Correct ({action.classify})")
                elif cls_score > 0:
                    feedback_parts.append(
                        f"Classification: Partially correct ({action.classify}, "
                        f"expected {gt['category']})"
                    )
                else:
                    feedback_parts.append(
                        f"Classification: Incorrect ({action.classify}, "
                        f"expected {gt['category']})"
                    )

            if weights["priority"] > 0:
                if pri_score == 1.0:
                    feedback_parts.append(f"Priority: Correct ({action.priority})")
                elif pri_score > 0:
                    feedback_parts.append(
                        f"Priority: Close ({action.priority}, expected {gt['priority']})"
                    )
                else:
                    feedback_parts.append(
                        f"Priority: Incorrect ({action.priority}, expected {gt['priority']})"
                    )

            if weights["route"] > 0:
                if rte_score == 1.0:
                    feedback_parts.append(f"Routing: Correct ({action.route_to})")
                elif rte_score > 0:
                    feedback_parts.append(
                        f"Routing: Partially correct ({action.route_to}, "
                        f"expected {gt['department']})"
                    )
                else:
                    feedback_parts.append(
                        f"Routing: Incorrect ({action.route_to}, "
                        f"expected {gt['department']})"
                    )

            if weights["response"] > 0:
                if rsp_score >= 0.7:
                    feedback_parts.append(f"Response: Good (score: {rsp_score:.2f})")
                elif rsp_score >= 0.3:
                    feedback_parts.append(f"Response: Acceptable (score: {rsp_score:.2f})")
                elif rsp_score > 0:
                    feedback_parts.append(f"Response: Needs improvement (score: {rsp_score:.2f})")
                else:
                    feedback_parts.append("Response: Missing or empty")

            # Streak bonus
            if step_score >= 0.8:
                self._streak += 1
                if self._streak >= 3:
                    bonus = 0.05 * min(self._streak - 2, 3)
                    step_score = min(step_score + bonus, 1.0)
                    feedback_parts.append(f"Streak bonus! ({self._streak} in a row)")
            else:
                self._streak = 0

            self._invalid_action_count = 0

        # Clamp score
        step_score = max(min(step_score, 1.0), -0.1)

        self._scores.append(step_score)
        self._cumulative_score = sum(self._scores) / len(self._emails)

        feedback_parts.append(f"Step score: {step_score:.3f}")
        feedback_parts.append(f"Cumulative: {self._cumulative_score:.3f}")

        # Move to next email
        self._current_idx += 1

        if self._current_idx >= len(self._emails):
            self._done = True
            feedback_parts.append("All emails processed. Episode complete.")

            return EmailTriageObservation(
                done=True,
                reward=step_score,
                feedback=" | ".join(feedback_parts),
                task_name=self._task,
                cumulative_score=self._cumulative_score,
                emails_processed=self._current_idx,
                total_emails=len(self._emails),
            )

        # Return next email
        next_email = self._emails[self._current_idx]
        return EmailTriageObservation(
            email_id=next_email["email_id"],
            sender=next_email["sender"],
            sender_name=next_email["sender_name"],
            subject=next_email["subject"],
            body=next_email["body"],
            timestamp=next_email["timestamp"],
            has_attachments=next_email["has_attachments"],
            is_reply=next_email["is_reply"],
            thread_count=next_email["thread_count"],
            task_name=self._task,
            task_description="",
            emails_remaining=len(self._emails) - self._current_idx - 1,
            emails_processed=self._current_idx,
            total_emails=len(self._emails),
            feedback=" | ".join(feedback_parts),
            cumulative_score=self._cumulative_score,
            done=False,
            reward=step_score,
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
