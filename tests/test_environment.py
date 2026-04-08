# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Email Triage Environment."""

import pytest

from models import EmailTriageAction, EmailTriageObservation
from server.email_triage_environment import (
    TASK_WEIGHTS,
    VALID_CATEGORIES,
    VALID_DEPARTMENTS,
    EmailTriageEnvironment,
    _score_classification,
    _score_priority,
    _score_response,
    _score_routing,
)


# ── Scoring function tests ────────────────────────────────────────────


class TestScoreClassification:
    """Tests for _score_classification."""

    def test_exact_match(self):
        for cat in VALID_CATEGORIES:
            assert _score_classification(cat, cat) == 1.0

    def test_none_returns_zero(self):
        assert _score_classification(None, "spam") == 0.0

    def test_invalid_category_returns_zero(self):
        assert _score_classification("invalid", "spam") == 0.0

    def test_related_spam_low_priority(self):
        assert _score_classification("spam", "low_priority") == 0.2
        assert _score_classification("low_priority", "spam") == 0.2

    def test_related_urgent_normal(self):
        assert _score_classification("urgent", "normal") == 0.3
        assert _score_classification("normal", "urgent") == 0.3

    def test_related_newsletter_low_priority(self):
        assert _score_classification("newsletter", "low_priority") == 0.4
        assert _score_classification("low_priority", "newsletter") == 0.4

    def test_related_newsletter_spam(self):
        assert _score_classification("newsletter", "spam") == 0.2
        assert _score_classification("spam", "newsletter") == 0.2

    def test_unrelated_categories_zero(self):
        assert _score_classification("spam", "urgent") == 0.0
        assert _score_classification("newsletter", "urgent") == 0.0

    def test_case_insensitive(self):
        assert _score_classification("SPAM", "spam") == 1.0
        assert _score_classification("Urgent", "urgent") == 1.0

    def test_whitespace_handling(self):
        assert _score_classification("  spam  ", "spam") == 1.0


class TestScorePriority:
    """Tests for _score_priority."""

    def test_exact_match(self):
        for p in range(1, 6):
            assert _score_priority(p, p) == 1.0

    def test_none_returns_zero(self):
        assert _score_priority(None, 3) == 0.0

    def test_out_of_range_returns_zero(self):
        assert _score_priority(0, 3) == 0.0
        assert _score_priority(6, 3) == 0.0
        assert _score_priority(-1, 3) == 0.0

    def test_off_by_one(self):
        assert _score_priority(2, 3) == 0.75
        assert _score_priority(4, 3) == 0.75

    def test_off_by_two(self):
        assert _score_priority(1, 3) == 0.5
        assert _score_priority(5, 3) == 0.5

    def test_off_by_three(self):
        assert _score_priority(1, 4) == 0.25

    def test_off_by_four(self):
        assert _score_priority(1, 5) == 0.0
        assert _score_priority(5, 1) == 0.0

    def test_linear_decay(self):
        """Scores should decrease linearly with distance."""
        scores = [_score_priority(i, 3) for i in range(1, 6)]
        assert scores == [0.5, 0.75, 1.0, 0.75, 0.5]


class TestScoreRouting:
    """Tests for _score_routing."""

    def test_exact_match(self):
        for dept in VALID_DEPARTMENTS:
            assert _score_routing(dept, dept) == 1.0

    def test_none_returns_zero(self):
        assert _score_routing(None, "engineering") == 0.0

    def test_invalid_department_returns_zero(self):
        assert _score_routing("invalid_dept", "engineering") == 0.0

    def test_related_engineering_support(self):
        assert _score_routing("engineering", "support") == 0.3
        assert _score_routing("support", "engineering") == 0.3

    def test_related_sales_marketing(self):
        assert _score_routing("sales", "marketing") == 0.3
        assert _score_routing("marketing", "sales") == 0.3

    def test_related_legal_executive(self):
        assert _score_routing("legal", "executive") == 0.3
        assert _score_routing("executive", "legal") == 0.3

    def test_related_hr_executive(self):
        assert _score_routing("hr", "executive") == 0.2
        assert _score_routing("executive", "hr") == 0.2

    def test_unrelated_departments_zero(self):
        assert _score_routing("engineering", "hr") == 0.0
        assert _score_routing("sales", "legal") == 0.0

    def test_case_insensitive(self):
        assert _score_routing("ENGINEERING", "engineering") == 1.0

    def test_whitespace_handling(self):
        assert _score_routing("  support  ", "support") == 1.0


class TestScoreResponse:
    """Tests for _score_response."""

    def test_none_returns_zero(self):
        assert _score_response(None, ["keyword"], "professional") == 0.0

    def test_empty_string_returns_zero(self):
        assert _score_response("", ["keyword"], "professional") == 0.0
        assert _score_response("   ", ["keyword"], "professional") == 0.0

    def test_none_tone_short_response(self):
        """When tone is 'none', short response gets 0.3."""
        score = _score_response("ok", [], "none")
        assert score == 0.3

    def test_none_tone_long_response_penalized(self):
        """When tone is 'none', long response gets 0.1."""
        long_resp = "x " * 50
        score = _score_response(long_resp, [], "none")
        assert score == 0.1

    def test_keyword_coverage_full(self):
        """All keywords present should give full keyword score."""
        response = "We are investigating the API errors and checking the trace for the account. We will escalat the team with an update on SLA."
        keywords = ["investigating", "API", "errors", "trace", "account", "escalat", "team", "update", "SLA"]
        score = _score_response(response, keywords, "professional")
        # Full keyword coverage = 0.6, plus length and tone
        assert score >= 0.6

    def test_keyword_coverage_partial(self):
        """Some keywords present should give partial score."""
        response = "We are investigating the errors."
        keywords = ["investigating", "API", "errors", "trace", "account"]
        score = _score_response(response, keywords, "professional")
        # 2/5 keywords = 0.24
        assert 0.2 <= score <= 0.6

    def test_length_appropriate(self):
        """Response of appropriate length gets length bonus."""
        response = " ".join(["word"] * 30)
        score = _score_response(response, [], "professional")
        # No keywords, but length bonus + possible tone bonus
        assert score > 0.0

    def test_very_short_response(self):
        """Very short response (<5 words) gets no length bonus."""
        score = _score_response("ok", ["ok"], "professional")
        # keyword: 0.6, length: 0.0, tone: 0.0
        assert score == pytest.approx(0.6, abs=0.1)

    def test_tone_markers_professional(self):
        """Professional tone markers should contribute to score."""
        response = "Thank you for reaching out. Please let us know if you need further assistance. Best regards."
        score = _score_response(response, [], "professional")
        assert score > 0.0

    def test_tone_markers_warning(self):
        """Warning tone markers should contribute to score."""
        response = "Do not click any links. This is suspicious and appears to be phishing. Please report it."
        score = _score_response(response, [], "warning")
        assert score > 0.0

    def test_score_capped_at_one(self):
        """Score should never exceed 1.0."""
        response = "Thank you please regards sincerely best " * 10
        keywords = ["thank", "please", "regards"]
        score = _score_response(response, keywords, "professional")
        assert score <= 1.0

    def test_empty_keywords_list(self):
        """Empty keywords list should not cause errors."""
        response = " ".join(["word"] * 30)
        score = _score_response(response, [], "professional")
        assert score >= 0.0


# ── Environment tests ──────────────────────────────────────────────────


class TestEmailTriageEnvironment:
    """Tests for the EmailTriageEnvironment class."""

    def test_init_default_task(self):
        env = EmailTriageEnvironment()
        assert env._task == "classification"

    def test_init_valid_task(self):
        for task in ["classification", "routing", "full_triage"]:
            env = EmailTriageEnvironment(task=task)
            assert env._task == task

    def test_init_invalid_task_defaults(self):
        env = EmailTriageEnvironment(task="invalid_task")
        assert env._task == "classification"

    def test_supports_concurrent_sessions(self):
        assert EmailTriageEnvironment.SUPPORTS_CONCURRENT_SESSIONS is True


class TestEnvironmentReset:
    """Tests for environment reset."""

    def test_reset_returns_observation(self):
        env = EmailTriageEnvironment()
        obs = env.reset()
        assert isinstance(obs, EmailTriageObservation)

    def test_reset_returns_first_email(self):
        env = EmailTriageEnvironment(task="classification")
        obs = env.reset()
        assert obs.email_id != ""
        assert obs.subject != ""
        assert obs.body != ""

    def test_reset_sets_task_metadata(self):
        env = EmailTriageEnvironment(task="routing")
        obs = env.reset()
        assert obs.task_name == "routing"
        assert obs.task_description != ""

    def test_reset_initial_state(self):
        env = EmailTriageEnvironment()
        obs = env.reset()
        assert obs.emails_processed == 0
        assert obs.total_emails == 5
        assert obs.emails_remaining == 4
        assert obs.cumulative_score == 0.0
        assert obs.done is False
        assert obs.reward == 0.0

    def test_reset_clears_previous_state(self):
        env = EmailTriageEnvironment()
        env.reset()
        # Step once
        env.step(EmailTriageAction(classify="spam", priority=5))
        # Reset should clear all state
        obs = env.reset()
        assert obs.emails_processed == 0
        assert obs.cumulative_score == 0.0
        assert obs.done is False

    def test_reset_generates_new_episode_id(self):
        env = EmailTriageEnvironment()
        env.reset()
        id1 = env._state.episode_id
        env.reset()
        id2 = env._state.episode_id
        assert id1 != id2


class TestEnvironmentStep:
    """Tests for environment step function."""

    def test_step_returns_observation(self):
        env = EmailTriageEnvironment(task="classification")
        env.reset()
        obs = env.step(EmailTriageAction(classify="spam", priority=5))
        assert isinstance(obs, EmailTriageObservation)

    def test_correct_classification_high_score(self):
        """Correct classification should yield a high score."""
        env = EmailTriageEnvironment(task="classification")
        env.reset()
        # e001 is spam, priority 5
        obs = env.step(EmailTriageAction(classify="spam", priority=5))
        assert obs.reward == 1.0

    def test_incorrect_classification_lower_score(self):
        env = EmailTriageEnvironment(task="classification")
        env.reset()
        # e001 is spam, classifying as "normal" should be low score
        obs = env.step(EmailTriageAction(classify="normal", priority=3))
        assert obs.reward < 1.0

    def test_step_advances_email(self):
        env = EmailTriageEnvironment(task="classification")
        env.reset()
        obs1 = env.step(EmailTriageAction(classify="spam", priority=5))
        assert obs1.emails_processed == 1
        assert obs1.emails_remaining == 3

    def test_step_increments_step_count(self):
        env = EmailTriageEnvironment()
        env.reset()
        env.step(EmailTriageAction(classify="spam", priority=5))
        assert env._state.step_count == 1
        env.step(EmailTriageAction(classify="urgent", priority=1))
        assert env._state.step_count == 2

    def test_episode_completes_after_all_emails(self):
        """Episode should be done after processing all emails."""
        env = EmailTriageEnvironment(task="classification")
        env.reset()
        for _ in range(5):
            obs = env.step(EmailTriageAction(classify="normal", priority=3))
        assert obs.done is True

    def test_step_after_done_returns_done(self):
        env = EmailTriageEnvironment(task="classification")
        env.reset()
        for _ in range(5):
            env.step(EmailTriageAction(classify="normal", priority=3))
        # Extra step after completion
        obs = env.step(EmailTriageAction(classify="spam", priority=5))
        assert obs.done is True
        assert obs.reward == 0.0
        assert "already completed" in obs.feedback.lower()

    def test_feedback_contains_info(self):
        env = EmailTriageEnvironment(task="classification")
        env.reset()
        obs = env.step(EmailTriageAction(classify="spam", priority=5))
        assert obs.feedback != ""
        assert "score" in obs.feedback.lower()

    def test_cumulative_score_accumulates(self):
        env = EmailTriageEnvironment(task="classification")
        env.reset()
        obs1 = env.step(EmailTriageAction(classify="spam", priority=5))
        score_after_1 = obs1.cumulative_score
        obs2 = env.step(EmailTriageAction(classify="urgent", priority=1))
        score_after_2 = obs2.cumulative_score
        # After two perfect scores, cumulative should be higher
        assert score_after_2 >= score_after_1

    def test_empty_action_no_fields(self):
        """Action with no fields should get zero score."""
        env = EmailTriageEnvironment(task="classification")
        env.reset()
        obs = env.step(EmailTriageAction())
        assert obs.reward == 0.0
        assert "no action" in obs.feedback.lower()

    def test_repeated_empty_actions_penalty(self):
        """Three or more empty actions should trigger penalty."""
        env = EmailTriageEnvironment(task="classification")
        env.reset()
        env.step(EmailTriageAction())
        env.step(EmailTriageAction())
        obs = env.step(EmailTriageAction())
        assert obs.reward == -0.1
        assert "penalty" in obs.feedback.lower()

    def test_valid_action_resets_invalid_count(self):
        """A valid action should reset the invalid action counter."""
        env = EmailTriageEnvironment(task="classification")
        env.reset()
        env.step(EmailTriageAction())  # invalid
        env.step(EmailTriageAction(classify="spam", priority=5))  # valid
        env.step(EmailTriageAction())  # invalid again
        env.step(EmailTriageAction())  # 2nd invalid, NOT 3rd
        obs = env.step(EmailTriageAction())  # 3rd invalid after reset
        assert obs.reward == -0.1


class TestEnvironmentRouting:
    """Tests for routing task."""

    def test_routing_weights(self):
        weights = TASK_WEIGHTS["routing"]
        assert weights["classify"] == 0.3
        assert weights["priority"] == 0.2
        assert weights["route"] == 0.5
        assert weights["response"] == 0.0

    def test_correct_routing_full_score(self):
        env = EmailTriageEnvironment(task="routing")
        env.reset()
        # e004: urgent, priority 1, engineering
        obs = env.step(
            EmailTriageAction(classify="urgent", priority=1, route_to="engineering")
        )
        assert obs.reward == 1.0

    def test_wrong_route_partial_score(self):
        env = EmailTriageEnvironment(task="routing")
        env.reset()
        # e004: department is engineering, support is related (0.3)
        obs = env.step(
            EmailTriageAction(classify="urgent", priority=1, route_to="support")
        )
        assert 0.0 < obs.reward < 1.0


class TestEnvironmentFullTriage:
    """Tests for full_triage task."""

    def test_full_triage_weights(self):
        weights = TASK_WEIGHTS["full_triage"]
        assert weights["classify"] == 0.15
        assert weights["priority"] == 0.1
        assert weights["route"] == 0.25
        assert weights["response"] == 0.5

    def test_full_triage_with_response(self):
        env = EmailTriageEnvironment(task="full_triage")
        env.reset()
        # e007: urgent, priority 1, legal
        obs = env.step(
            EmailTriageAction(
                classify="urgent",
                priority=1,
                route_to="legal",
                response_draft=(
                    "We acknowledge receipt of the DSAR case REG-2026-0891. "
                    "Our data protection team is reviewing the GDPR compliance "
                    "requirements and will respond within the 48 hours deadline."
                ),
            )
        )
        assert obs.reward > 0.5

    def test_full_triage_without_response_loses_half(self):
        """Missing response in full_triage loses 50% weight."""
        env = EmailTriageEnvironment(task="full_triage")
        env.reset()
        obs = env.step(
            EmailTriageAction(classify="urgent", priority=1, route_to="legal")
        )
        # Max possible without response: 0.15 + 0.1 + 0.25 = 0.5
        assert obs.reward <= 0.5


class TestEnvironmentStreak:
    """Tests for streak bonus."""

    def test_streak_bonus_after_three(self):
        """Streak bonus should activate after 3 consecutive high scores."""
        env = EmailTriageEnvironment(task="classification")
        env.reset()
        # e001: spam/5, e002: urgent/1, e003: newsletter/5
        env.step(EmailTriageAction(classify="spam", priority=5))  # 1.0
        env.step(EmailTriageAction(classify="urgent", priority=1))  # 1.0
        obs = env.step(EmailTriageAction(classify="newsletter", priority=5))  # 1.0 + bonus
        assert "streak" in obs.feedback.lower()

    def test_streak_breaks_on_low_score(self):
        """Streak should reset when score < 0.8."""
        env = EmailTriageEnvironment(task="classification")
        env.reset()
        env.step(EmailTriageAction(classify="spam", priority=5))  # 1.0
        env.step(EmailTriageAction(classify="normal", priority=3))  # wrong, streak breaks
        assert env._streak == 0


class TestEnvironmentState:
    """Tests for environment state property."""

    def test_state_has_episode_id(self):
        env = EmailTriageEnvironment()
        env.reset()
        assert env.state.episode_id is not None
        assert len(env.state.episode_id) > 0

    def test_state_step_count(self):
        env = EmailTriageEnvironment()
        env.reset()
        assert env.state.step_count == 0
        env.step(EmailTriageAction(classify="spam", priority=5))
        assert env.state.step_count == 1
