# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the email dataset."""

import pytest

from email_dataset import (
    EMAIL_DATASET,
    TASK_EASY_EMAILS,
    TASK_HARD_EMAILS,
    TASK_MEDIUM_EMAILS,
    get_emails_for_task,
)


VALID_CATEGORIES = {"spam", "urgent", "normal", "low_priority", "newsletter"}
VALID_DEPARTMENTS = {
    "engineering", "sales", "support", "hr",
    "legal", "marketing", "finance", "executive",
}
REQUIRED_EMAIL_FIELDS = {
    "email_id", "sender", "sender_name", "subject", "body",
    "timestamp", "has_attachments", "is_reply", "thread_count",
    "ground_truth",
}
REQUIRED_GT_FIELDS = {
    "category", "priority", "department",
    "response_keywords", "response_tone",
}


class TestEmailDatasetIntegrity:
    """Tests for email dataset structure and completeness."""

    def test_dataset_not_empty(self):
        assert len(EMAIL_DATASET) > 0

    def test_total_email_count(self):
        """Dataset should have 15 emails."""
        assert len(EMAIL_DATASET) == 15

    def test_unique_email_ids(self):
        ids = [e["email_id"] for e in EMAIL_DATASET]
        assert len(ids) == len(set(ids)), "Duplicate email IDs found"

    @pytest.mark.parametrize("email", EMAIL_DATASET, ids=lambda e: e["email_id"])
    def test_email_has_required_fields(self, email):
        missing = REQUIRED_EMAIL_FIELDS - set(email.keys())
        assert not missing, f"Email {email['email_id']} missing fields: {missing}"

    @pytest.mark.parametrize("email", EMAIL_DATASET, ids=lambda e: e["email_id"])
    def test_ground_truth_has_required_fields(self, email):
        gt = email["ground_truth"]
        missing = REQUIRED_GT_FIELDS - set(gt.keys())
        assert not missing, f"Email {email['email_id']} ground_truth missing: {missing}"

    @pytest.mark.parametrize("email", EMAIL_DATASET, ids=lambda e: e["email_id"])
    def test_ground_truth_category_valid(self, email):
        cat = email["ground_truth"]["category"]
        assert cat in VALID_CATEGORIES, f"Invalid category '{cat}' in {email['email_id']}"

    @pytest.mark.parametrize("email", EMAIL_DATASET, ids=lambda e: e["email_id"])
    def test_ground_truth_priority_in_range(self, email):
        pri = email["ground_truth"]["priority"]
        assert 1 <= pri <= 5, f"Priority {pri} out of range in {email['email_id']}"

    @pytest.mark.parametrize("email", EMAIL_DATASET, ids=lambda e: e["email_id"])
    def test_ground_truth_department_valid(self, email):
        dept = email["ground_truth"]["department"]
        assert dept in VALID_DEPARTMENTS, f"Invalid department '{dept}' in {email['email_id']}"

    @pytest.mark.parametrize("email", EMAIL_DATASET, ids=lambda e: e["email_id"])
    def test_email_body_not_empty(self, email):
        assert email["body"].strip(), f"Empty body in {email['email_id']}"

    @pytest.mark.parametrize("email", EMAIL_DATASET, ids=lambda e: e["email_id"])
    def test_email_subject_not_empty(self, email):
        assert email["subject"].strip(), f"Empty subject in {email['email_id']}"


class TestTaskEmailSubsets:
    """Tests for task-specific email subsets."""

    def test_easy_emails_count(self):
        assert len(TASK_EASY_EMAILS) == 5

    def test_medium_emails_count(self):
        assert len(TASK_MEDIUM_EMAILS) == 5

    def test_hard_emails_count(self):
        assert len(TASK_HARD_EMAILS) == 5

    def test_no_overlap_between_tasks(self):
        easy = set(TASK_EASY_EMAILS)
        medium = set(TASK_MEDIUM_EMAILS)
        hard = set(TASK_HARD_EMAILS)
        assert not easy & medium, "Easy and Medium overlap"
        assert not easy & hard, "Easy and Hard overlap"
        assert not medium & hard, "Medium and Hard overlap"

    def test_all_emails_assigned_to_task(self):
        all_task_ids = set(TASK_EASY_EMAILS + TASK_MEDIUM_EMAILS + TASK_HARD_EMAILS)
        dataset_ids = {e["email_id"] for e in EMAIL_DATASET}
        assert all_task_ids == dataset_ids, "Not all emails assigned to a task"

    def test_all_task_ids_exist_in_dataset(self):
        dataset_ids = {e["email_id"] for e in EMAIL_DATASET}
        for eid in TASK_EASY_EMAILS + TASK_MEDIUM_EMAILS + TASK_HARD_EMAILS:
            assert eid in dataset_ids, f"Task references missing email: {eid}"


class TestGetEmailsForTask:
    """Tests for the get_emails_for_task function."""

    def test_classification_returns_easy_emails(self):
        emails = get_emails_for_task("classification")
        ids = [e["email_id"] for e in emails]
        assert ids == TASK_EASY_EMAILS

    def test_routing_returns_medium_emails(self):
        emails = get_emails_for_task("routing")
        ids = [e["email_id"] for e in emails]
        assert ids == TASK_MEDIUM_EMAILS

    def test_full_triage_returns_hard_emails(self):
        emails = get_emails_for_task("full_triage")
        ids = [e["email_id"] for e in emails]
        assert ids == TASK_HARD_EMAILS

    def test_unknown_task_returns_easy(self):
        emails = get_emails_for_task("nonexistent_task")
        ids = [e["email_id"] for e in emails]
        assert ids == TASK_EASY_EMAILS

    def test_returned_emails_have_ground_truth(self):
        for task in ["classification", "routing", "full_triage"]:
            emails = get_emails_for_task(task)
            for email in emails:
                assert "ground_truth" in email
