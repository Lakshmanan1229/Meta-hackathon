# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Email dataset for the Email Triage environment.

Contains realistic emails with ground-truth labels for classification,
routing, priority, and expected response characteristics.
"""

from typing import Any, Dict, List

# Each email has ground-truth labels used by graders
EMAIL_DATASET: List[Dict[str, Any]] = [
    # --- EASY: Clear spam ---
    {
        "email_id": "e001",
        "sender": "winner@prize-notifications.biz",
        "sender_name": "Prize Committee",
        "subject": "YOU WON $1,000,000!!! Claim NOW!!!",
        "body": (
            "Congratulations! You have been selected as our GRAND PRIZE WINNER! "
            "Click here immediately to claim your $1,000,000 prize. "
            "Send us your bank details and social security number to process your winnings. "
            "ACT NOW - offer expires in 24 hours!!!"
        ),
        "timestamp": "2026-04-01T08:15:00Z",
        "has_attachments": False,
        "is_reply": False,
        "thread_count": 1,
        "ground_truth": {
            "category": "spam",
            "priority": 5,
            "department": "support",
            "response_keywords": ["spam", "ignore", "do not click", "phishing", "scam"],
            "response_tone": "warning",
        },
    },
    # --- EASY: Clear urgent from CEO ---
    {
        "email_id": "e002",
        "sender": "ceo@company.com",
        "sender_name": "Sarah Chen, CEO",
        "subject": "URGENT: Board meeting moved to tomorrow 9 AM",
        "body": (
            "Hi team,\n\n"
            "The quarterly board meeting has been moved to tomorrow at 9 AM "
            "due to a scheduling conflict. Please ensure all Q1 reports are "
            "finalized by end of day today. This is critical - the board needs "
            "to review our performance metrics before the investor call on Thursday.\n\n"
            "Please confirm receipt of this message.\n\n"
            "Best,\nSarah"
        ),
        "timestamp": "2026-04-01T14:30:00Z",
        "has_attachments": False,
        "is_reply": False,
        "thread_count": 1,
        "ground_truth": {
            "category": "urgent",
            "priority": 1,
            "department": "executive",
            "response_keywords": ["confirm", "receipt", "acknowledged", "reports", "ready", "prepared"],
            "response_tone": "professional",
        },
    },
    # --- EASY: Newsletter ---
    {
        "email_id": "e003",
        "sender": "digest@techweekly.io",
        "sender_name": "TechWeekly Digest",
        "subject": "This Week in Tech: AI Breakthroughs & More",
        "body": (
            "TechWeekly Digest - April 1, 2026\n\n"
            "Top Stories:\n"
            "1. New LLM achieves record benchmark scores\n"
            "2. Quantum computing milestone reached\n"
            "3. Open source AI tools roundup\n\n"
            "Read more at techweekly.io\n\n"
            "You are receiving this because you subscribed. Unsubscribe: techweekly.io/unsub"
        ),
        "timestamp": "2026-04-01T06:00:00Z",
        "has_attachments": False,
        "is_reply": False,
        "thread_count": 1,
        "ground_truth": {
            "category": "newsletter",
            "priority": 5,
            "department": "marketing",
            "response_keywords": [],
            "response_tone": "none",
        },
    },
    # --- MEDIUM: Customer support request ---
    {
        "email_id": "e004",
        "sender": "john.doe@clientcorp.com",
        "sender_name": "John Doe",
        "subject": "Integration API returning 500 errors since yesterday",
        "body": (
            "Hi Support Team,\n\n"
            "We've been experiencing intermittent 500 errors from your REST API "
            "since yesterday around 3 PM EST. Our integration pipeline processes "
            "about 10,000 records per hour and roughly 15% of requests are failing.\n\n"
            "Error payload:\n"
            '{"error": "Internal Server Error", "code": 500, "trace_id": "abc-123"}\n\n'
            "This is affecting our production data pipeline. Can someone look into "
            "this urgently? We have an SLA that requires 99.9% uptime.\n\n"
            "Our account ID is CLIENT-4521.\n\n"
            "Thanks,\nJohn Doe\nSenior Engineer, ClientCorp"
        ),
        "timestamp": "2026-04-01T10:22:00Z",
        "has_attachments": False,
        "is_reply": False,
        "thread_count": 1,
        "ground_truth": {
            "category": "urgent",
            "priority": 1,
            "department": "engineering",
            "response_keywords": [
                "investigating", "API", "errors", "trace", "account",
                "escalat", "team", "update", "SLA",
            ],
            "response_tone": "professional",
        },
    },
    # --- MEDIUM: HR policy question ---
    {
        "email_id": "e005",
        "sender": "maria.garcia@company.com",
        "sender_name": "Maria Garcia",
        "subject": "Question about parental leave policy",
        "body": (
            "Hi HR,\n\n"
            "I'm expecting my first child in August and wanted to understand "
            "the company's parental leave policy. Specifically:\n\n"
            "1. How many weeks of paid leave are available?\n"
            "2. Is there a minimum tenure requirement?\n"
            "3. Can the leave be split into multiple periods?\n"
            "4. What documentation do I need to provide?\n\n"
            "I've been with the company for 2 years in the engineering department.\n\n"
            "Thank you,\nMaria Garcia"
        ),
        "timestamp": "2026-04-01T09:45:00Z",
        "has_attachments": False,
        "is_reply": False,
        "thread_count": 1,
        "ground_truth": {
            "category": "normal",
            "priority": 3,
            "department": "hr",
            "response_keywords": [
                "parental leave", "policy", "weeks", "documentation",
                "HR", "congratulations",
            ],
            "response_tone": "warm",
        },
    },
    # --- MEDIUM: Sales inquiry ---
    {
        "email_id": "e006",
        "sender": "alex.kim@bigenterprise.com",
        "sender_name": "Alex Kim",
        "subject": "Enterprise pricing for 500+ seats",
        "body": (
            "Hello,\n\n"
            "I'm the VP of Engineering at BigEnterprise. We currently use your "
            "competitor's product but our contract is up for renewal in Q3.\n\n"
            "We're evaluating alternatives for our team of 500+ developers. "
            "Could you provide:\n"
            "1. Enterprise pricing for 500+ seats\n"
            "2. Volume discount structure\n"
            "3. Migration support availability\n"
            "4. SOC 2 compliance documentation\n\n"
            "Our annual budget for this category is $2M+. Would appreciate a "
            "call this week if possible.\n\n"
            "Best regards,\nAlex Kim\nVP Engineering, BigEnterprise"
        ),
        "timestamp": "2026-04-01T11:15:00Z",
        "has_attachments": False,
        "is_reply": False,
        "thread_count": 1,
        "ground_truth": {
            "category": "urgent",
            "priority": 2,
            "department": "sales",
            "response_keywords": [
                "pricing", "enterprise", "call", "schedule", "demo",
                "team", "migration",
            ],
            "response_tone": "professional",
        },
    },
    # --- HARD: Ambiguous legal/compliance ---
    {
        "email_id": "e007",
        "sender": "compliance@regulator.gov",
        "sender_name": "Office of Data Protection",
        "subject": "Data Subject Access Request - Case #REG-2026-0891",
        "body": (
            "Dear Data Protection Officer,\n\n"
            "Pursuant to Article 15 of the General Data Protection Regulation, "
            "we are forwarding a Data Subject Access Request (DSAR) filed by "
            "an individual regarding their personal data held by your organization.\n\n"
            "Case Reference: REG-2026-0891\n"
            "Data Subject: [REDACTED]\n"
            "Request Type: Full data export and deletion\n"
            "Compliance Deadline: 30 calendar days from receipt\n\n"
            "Please acknowledge receipt of this request within 48 hours and "
            "provide a timeline for compliance. Failure to comply within the "
            "statutory period may result in enforcement action.\n\n"
            "Regards,\n"
            "Office of Data Protection\n"
            "Data Protection Authority"
        ),
        "timestamp": "2026-04-01T08:00:00Z",
        "has_attachments": True,
        "is_reply": False,
        "thread_count": 1,
        "ground_truth": {
            "category": "urgent",
            "priority": 1,
            "department": "legal",
            "response_keywords": [
                "acknowledge", "DSAR", "GDPR", "compliance", "deadline",
                "48 hours", "data protection", "case", "REG-2026-0891",
            ],
            "response_tone": "formal",
        },
    },
    # --- HARD: Multi-department issue ---
    {
        "email_id": "e008",
        "sender": "press@technews.com",
        "sender_name": "Jamie Rivera, Tech Journalist",
        "subject": "Press inquiry: Security vulnerability disclosure",
        "body": (
            "Hi,\n\n"
            "I'm a journalist with TechNews. We've received a tip from a "
            "security researcher about a potential vulnerability in your "
            "product's authentication system. The researcher claims:\n\n"
            "1. Session tokens are not properly invalidated on logout\n"
            "2. Password reset tokens have no expiration\n"
            "3. MFA can be bypassed via API endpoint\n\n"
            "We plan to publish a story on this within 72 hours. Before we do, "
            "we want to give your team a chance to respond.\n\n"
            "Questions:\n"
            "- Are you aware of these vulnerabilities?\n"
            "- What steps are you taking to address them?\n"
            "- Have any users been affected?\n\n"
            "Please respond by Thursday 5 PM EST.\n\n"
            "Jamie Rivera\nSenior Tech Reporter, TechNews"
        ),
        "timestamp": "2026-04-01T13:00:00Z",
        "has_attachments": False,
        "is_reply": False,
        "thread_count": 1,
        "ground_truth": {
            "category": "urgent",
            "priority": 1,
            "department": "executive",
            "response_keywords": [
                "security", "investigating", "team", "respond", "vulnerability",
                "disclosure", "press", "statement",
            ],
            "response_tone": "careful",
        },
    },
    # --- HARD: Subtle phishing disguised as internal ---
    {
        "email_id": "e009",
        "sender": "it-department@c0mpany.com",
        "sender_name": "IT Security Team",
        "subject": "Action Required: Password Expiry - Update Within 24hrs",
        "body": (
            "Dear Employee,\n\n"
            "Our security system indicates your password will expire in 24 hours. "
            "To avoid disruption to your work, please update your credentials "
            "immediately by clicking the link below:\n\n"
            "https://c0mpany-secure-portal.com/password-reset\n\n"
            "You will need to provide:\n"
            "- Current password\n"
            "- New password\n"
            "- Employee ID\n"
            "- Last 4 digits of SSN for verification\n\n"
            "This is an automated security notification. Do not reply to this email.\n\n"
            "IT Security Team"
        ),
        "timestamp": "2026-04-01T07:30:00Z",
        "has_attachments": False,
        "is_reply": False,
        "thread_count": 1,
        "ground_truth": {
            "category": "spam",
            "priority": 2,
            "department": "support",
            "response_keywords": [
                "phishing", "suspicious", "do not click", "report",
                "verify", "IT department", "security",
            ],
            "response_tone": "warning",
        },
    },
    # --- MEDIUM: Internal project update ---
    {
        "email_id": "e010",
        "sender": "dev-team@company.com",
        "sender_name": "Dev Team Updates",
        "subject": "Sprint 42 Retrospective Notes",
        "body": (
            "Hi all,\n\n"
            "Here are the key takeaways from Sprint 42 retro:\n\n"
            "What went well:\n"
            "- Shipped the new dashboard feature on time\n"
            "- Zero critical bugs in production\n"
            "- Good cross-team collaboration on the API migration\n\n"
            "What needs improvement:\n"
            "- Code review turnaround still averaging 48hrs (target: 24hrs)\n"
            "- Test coverage dropped to 72% (target: 80%)\n"
            "- Documentation updates lagging behind feature releases\n\n"
            "Action items:\n"
            "1. @engineering: Set up automated PR reminders\n"
            "2. @qa: Create coverage improvement plan by Friday\n"
            "3. @all: Update docs within 2 days of feature merge\n\n"
            "Next sprint planning: Monday 10 AM\n\n"
            "Best,\nDev Team"
        ),
        "timestamp": "2026-04-01T16:00:00Z",
        "has_attachments": False,
        "is_reply": False,
        "thread_count": 1,
        "ground_truth": {
            "category": "normal",
            "priority": 3,
            "department": "engineering",
            "response_keywords": [
                "sprint", "noted", "action items", "planning", "retro",
            ],
            "response_tone": "professional",
        },
    },
    # --- EASY: Low priority vendor email ---
    {
        "email_id": "e011",
        "sender": "promotions@cloudvendor.com",
        "sender_name": "CloudVendor Promotions",
        "subject": "Spring Sale: 20% off all cloud services",
        "body": (
            "Hi there,\n\n"
            "Spring into savings with CloudVendor! For a limited time, "
            "enjoy 20% off all our cloud services:\n\n"
            "- Compute instances\n"
            "- Storage solutions\n"
            "- Database services\n\n"
            "Use code SPRING2026 at checkout. Offer valid through April 30.\n\n"
            "View our full catalog at cloudvendor.com/spring-sale\n\n"
            "Best,\nCloudVendor Team\n\n"
            "Unsubscribe: cloudvendor.com/unsub"
        ),
        "timestamp": "2026-04-01T05:00:00Z",
        "has_attachments": False,
        "is_reply": False,
        "thread_count": 1,
        "ground_truth": {
            "category": "newsletter",
            "priority": 5,
            "department": "marketing",
            "response_keywords": [],
            "response_tone": "none",
        },
    },
    # --- HARD: Contract negotiation thread ---
    {
        "email_id": "e012",
        "sender": "legal@partnerfirm.com",
        "sender_name": "Patricia Wells, Partner",
        "subject": "RE: Master Services Agreement - Revised Terms",
        "body": (
            "Dear Counsel,\n\n"
            "Thank you for the revised MSA draft. After review, we have "
            "the following concerns with the proposed changes:\n\n"
            "1. Section 7.2 (Liability Cap): The proposed cap of $5M is "
            "significantly below industry standard for agreements of this "
            "magnitude. We propose $15M or 2x annual contract value.\n\n"
            "2. Section 12.1 (IP Assignment): The broad IP assignment clause "
            "could capture our pre-existing IP. We need a carve-out for "
            "background IP and jointly developed materials.\n\n"
            "3. Section 15.3 (Termination): 30-day termination notice is "
            "insufficient for an enterprise deployment. We require 90 days.\n\n"
            "4. Exhibit B (SLA): The 99.5% uptime SLA needs to be 99.9% "
            "with specific remedies for breaches.\n\n"
            "We need resolution on these points before we can proceed. "
            "Can we schedule a call for Thursday?\n\n"
            "Regards,\nPatricia Wells\nPartner, Wells & Associates LLP"
        ),
        "timestamp": "2026-04-01T15:45:00Z",
        "has_attachments": True,
        "is_reply": True,
        "thread_count": 5,
        "ground_truth": {
            "category": "urgent",
            "priority": 2,
            "department": "legal",
            "response_keywords": [
                "MSA", "terms", "review", "liability", "IP", "SLA",
                "call", "schedule", "negotiate",
            ],
            "response_tone": "formal",
        },
    },
    # --- MEDIUM: Finance request ---
    {
        "email_id": "e013",
        "sender": "accounting@company.com",
        "sender_name": "Accounting Department",
        "subject": "Q1 Expense Reports Due by April 7",
        "body": (
            "Hi team,\n\n"
            "This is a reminder that all Q1 2026 expense reports are due "
            "by April 7. Please ensure you:\n\n"
            "1. Submit all receipts in the expense management system\n"
            "2. Categorize expenses correctly\n"
            "3. Get manager approval before submission\n"
            "4. Include project codes for client-billable expenses\n\n"
            "Late submissions may delay reimbursement. If you have questions "
            "about categorization, refer to the expense policy on the intranet "
            "or contact the finance team.\n\n"
            "Thank you,\nAccounting Department"
        ),
        "timestamp": "2026-04-01T09:00:00Z",
        "has_attachments": False,
        "is_reply": False,
        "thread_count": 1,
        "ground_truth": {
            "category": "normal",
            "priority": 3,
            "department": "finance",
            "response_keywords": [
                "expense", "submit", "receipts", "April 7", "noted",
            ],
            "response_tone": "professional",
        },
    },
    # --- EASY: Obvious normal email ---
    {
        "email_id": "e014",
        "sender": "teammate@company.com",
        "sender_name": "Lisa Park",
        "subject": "Team lunch Friday?",
        "body": (
            "Hey team!\n\n"
            "Anyone up for team lunch this Friday? I was thinking we could "
            "try that new Thai place on 5th Street. They have great reviews "
            "and it's walking distance from the office.\n\n"
            "Let me know if you're interested and I'll make a reservation.\n\n"
            "Lisa"
        ),
        "timestamp": "2026-04-01T12:00:00Z",
        "has_attachments": False,
        "is_reply": False,
        "thread_count": 1,
        "ground_truth": {
            "category": "low_priority",
            "priority": 4,
            "department": "hr",
            "response_keywords": [
                "lunch", "Friday", "sounds good", "interested",
            ],
            "response_tone": "casual",
        },
    },
    # --- HARD: Ambiguous multi-department request ---
    {
        "email_id": "e015",
        "sender": "vp-product@company.com",
        "sender_name": "Marcus Johnson, VP Product",
        "subject": "Competitive analysis needed ASAP - board prep",
        "body": (
            "Hi all,\n\n"
            "The board wants a competitive analysis deck by next Monday. "
            "I need contributions from multiple teams:\n\n"
            "Engineering: Technical comparison of our platform vs. CompetitorX "
            "and CompetitorY. Focus on performance benchmarks, feature parity, "
            "and architecture advantages.\n\n"
            "Sales: Win/loss analysis for Q1. Which deals did we lose to "
            "competitors and why? Include pipeline impact estimates.\n\n"
            "Marketing: Brand perception data and analyst reports. How do "
            "we compare in the latest Gartner/Forrester reports?\n\n"
            "Finance: Cost structure comparison. What's our unit economics "
            "vs. competitor pricing?\n\n"
            "Please send your sections by Thursday EOD so I can compile "
            "the final deck over the weekend.\n\n"
            "This is a board-level request so please prioritize accordingly.\n\n"
            "Marcus"
        ),
        "timestamp": "2026-04-01T17:30:00Z",
        "has_attachments": False,
        "is_reply": False,
        "thread_count": 1,
        "ground_truth": {
            "category": "urgent",
            "priority": 1,
            "department": "executive",
            "response_keywords": [
                "competitive analysis", "board", "sections", "Thursday",
                "priority", "deck", "contributions",
            ],
            "response_tone": "professional",
        },
    },
]

# Task definitions with email subsets
TASK_EASY_EMAILS = ["e001", "e002", "e003", "e011", "e014"]
TASK_MEDIUM_EMAILS = ["e004", "e005", "e006", "e010", "e013"]
TASK_HARD_EMAILS = ["e007", "e008", "e009", "e012", "e015"]


def get_emails_for_task(task_name: str) -> List[Dict[str, Any]]:
    """Get the email subset for a given task.

    Args:
        task_name: One of 'classification', 'routing', 'full_triage'

    Returns:
        List of email dictionaries for the task
    """
    if task_name == "classification":
        email_ids = TASK_EASY_EMAILS
    elif task_name == "routing":
        email_ids = TASK_MEDIUM_EMAILS
    elif task_name == "full_triage":
        email_ids = TASK_HARD_EMAILS
    else:
        email_ids = TASK_EASY_EMAILS

    id_to_email = {e["email_id"]: e for e in EMAIL_DATASET}
    return [id_to_email[eid] for eid in email_ids if eid in id_to_email]
