# Email Triage OpenEnv Environment

A real-world OpenEnv environment that simulates **email triage** — the process of classifying, prioritizing, routing, and responding to incoming emails. This is a genuine workplace task performed by millions of knowledge workers daily, making it an excellent benchmark for evaluating AI agent capabilities.

## Motivation

Email triage is one of the most common productivity tasks in the modern workplace. An effective email triage agent must:
- **Understand context** — distinguish spam from urgent executive communications
- **Classify accurately** — categorize emails into actionable buckets
- **Prioritize correctly** — identify time-sensitive items
- **Route intelligently** — direct emails to the right department
- **Respond appropriately** — draft contextually correct responses with proper tone

This environment provides a controlled, reproducible benchmark for training and evaluating AI agents on these real-world skills.

## Environment Overview

### Action Space

The agent interacts via `EmailTriageAction` with four optional fields:

| Field | Type | Description |
|-------|------|-------------|
| `classify` | `str` | Category: `spam`, `urgent`, `normal`, `low_priority`, `newsletter` |
| `priority` | `int` | Priority level: `1` (highest) to `5` (lowest) |
| `route_to` | `str` | Department: `engineering`, `sales`, `support`, `hr`, `legal`, `marketing`, `finance`, `executive` |
| `response_draft` | `str` | Draft response text for the email |

### Observation Space

The agent receives `EmailTriageObservation` containing:

| Field | Type | Description |
|-------|------|-------------|
| `email_id` | `str` | Unique email identifier |
| `sender` | `str` | Sender email address |
| `sender_name` | `str` | Sender display name |
| `subject` | `str` | Email subject line |
| `body` | `str` | Email body content |
| `timestamp` | `str` | When the email was received |
| `has_attachments` | `bool` | Whether email has attachments |
| `is_reply` | `bool` | Whether this is a reply |
| `thread_count` | `int` | Number of emails in thread |
| `task_name` | `str` | Current task name |
| `task_description` | `str` | What the agent should do |
| `emails_remaining` | `int` | Emails left to process |
| `emails_processed` | `int` | Emails already processed |
| `total_emails` | `int` | Total emails in episode |
| `feedback` | `str` | Feedback on previous action |
| `cumulative_score` | `float` | Running cumulative score |
| `valid_categories` | `List[str]` | Valid classification categories |
| `valid_departments` | `List[str]` | Valid routing departments |

## Tasks (3 Difficulty Levels)

### Task 1: Classification (Easy)
- **Objective**: Classify 5 emails by category and assign priority
- **Emails**: Clear-cut spam, obvious urgent messages, newsletters, simple internal emails
- **Scored fields**: `classify` (60%), `priority` (40%)
- **Expected difficulty**: Straightforward — clear signals in subject/body

### Task 2: Routing (Medium)  
- **Objective**: Classify, prioritize, AND route 5 emails to correct departments
- **Emails**: Customer support tickets, HR inquiries, sales opportunities, internal project updates
- **Scored fields**: `classify` (30%), `priority` (20%), `route_to` (50%)
- **Expected difficulty**: Moderate — requires understanding organizational structure

### Task 3: Full Triage (Hard)
- **Objective**: Complete triage including drafting appropriate responses
- **Emails**: GDPR compliance requests, press security inquiries, phishing attempts disguised as internal, legal negotiations, cross-department executive requests
- **Scored fields**: `classify` (15%), `priority` (10%), `route_to` (25%), `response_draft` (50%)
- **Expected difficulty**: Challenging — ambiguous emails, subtle phishing, tone-sensitive responses

## Reward Function

The reward function provides **dense, informative signals**:

- **Per-step scoring** (0.0 - 1.0): Weighted combination of component scores
- **Partial credit**: Close-but-wrong answers get partial scores
  - Related categories (e.g., `low_priority` vs `newsletter`): 0.2-0.4
  - Close priority (off by 1): 0.75; (off by 2): 0.5
  - Related departments (e.g., `engineering` vs `support`): 0.3
- **Response scoring**: Based on keyword coverage (60%), length (20%), tone (20%)
- **Streak bonus**: 3+ consecutive high scores (≥0.8) earn a bonus
- **Invalid action penalty**: Repeated empty actions receive -0.1

## Setup & Usage

### Prerequisites
- Python 3.10+
- Docker (for containerized deployment)
- `openenv-core` package

### Local Development

```bash
# Install dependencies
pip install openenv-core[core]

# Run the server locally
cd /path/to/this/repo
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
# Build
docker build -t email_triage_env:latest -f server/Dockerfile .

# Run
docker run -p 8000:8000 email_triage_env:latest
```

### Validate

```bash
openenv validate
```

### Run Inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4"
export HF_TOKEN="your-api-key"
python inference.py
```

## Baseline Scores

| Task | Difficulty | Expected Score Range |
|------|-----------|---------------------|
| classification | Easy | 0.70 - 0.95 |
| routing | Medium | 0.55 - 0.85 |
| full_triage | Hard | 0.35 - 0.70 |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment, returns first email |
| `/step` | POST | Submit triage action, returns next email + reward |
| `/state` | GET | Get current episode state |
| `/health` | GET | Health check |
| `/schema` | GET | Action/observation JSON schemas |
| `/ws` | WebSocket | Persistent session endpoint |

## Project Structure

```
├── openenv.yaml          # OpenEnv spec metadata
├── models.py             # Pydantic Action & Observation models
├── email_dataset.py      # Realistic email dataset with ground truth
├── client.py             # EnvClient implementation
├── inference.py          # Baseline inference script
├── pyproject.toml        # Python project configuration
├── __init__.py           # Package init
├── README.md             # This file
└── server/
    ├── __init__.py
    ├── app.py            # FastAPI application
    ├── email_triage_environment.py  # Core environment logic
    ├── requirements.txt  # Server dependencies
    └── Dockerfile        # Container configuration
```

## License

BSD-style license. See LICENSE file for details.