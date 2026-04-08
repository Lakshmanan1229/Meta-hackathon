#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Baseline inference script for the Email Triage OpenEnv environment.

Uses the OpenAI API client to run a model against the environment.
Reads API credentials from environment variables.
Produces reproducible baseline scores on all 3 tasks.

Required environment variables:
    API_BASE_URL  - The API endpoint for the LLM
    MODEL_NAME    - The model identifier to use for inference
    HF_TOKEN      - Your Hugging Face / API key

Usage:
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4"
    export HF_TOKEN="your-key"
    python inference.py
"""

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4")
API_KEY = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))

IMAGE_NAME = "email_triage_env:latest"
BENCHMARK = "email_triage_env"
MAX_STEPS = 10  # max steps per task (each task has 5 emails)
MAX_TOTAL_REWARD = 5.0  # 5 emails, max 1.0 each
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS = [
    {
        "name": "classification",
        "description": "Easy: Classify emails by category and priority",
        "max_steps": 6,
    },
    {
        "name": "routing",
        "description": "Medium: Classify, prioritize, and route emails",
        "max_steps": 6,
    },
    {
        "name": "full_triage",
        "description": "Hard: Full triage with response drafting",
        "max_steps": 6,
    },
]

# ── Logging helpers (required format) ──────────────────────────────────


def log_start(task: str, env: str, model: str) -> None:
    """Log the start of a task evaluation."""
    print(
        f"[START] task={task} env={env} model={model}",
        flush=True,
    )


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    """Log a single step."""
    error_str = f' error="{error}"' if error else ""
    print(
        f"[STEP] step={step} action={json.dumps(action)} reward={reward:.4f} done={done}{error_str}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    """Log the end of a task evaluation."""
    print(
        f"[END] success={success} steps={steps} score={score:.4f} rewards={json.dumps(rewards)}",
        flush=True,
    )


# ── LLM helper ─────────────────────────────────────────────────────────


def get_model_action(
    client: OpenAI,
    task_name: str,
    email_obs: Dict[str, Any],
    history: List[str],
    last_feedback: str,
    last_reward: float,
) -> Dict[str, Any]:
    """Use the LLM to decide on a triage action for the current email."""
    system_prompt = _build_system_prompt(task_name)
    user_prompt = _build_user_prompt(email_obs, history, last_feedback, last_reward)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        content = response.choices[0].message.content or ""
        return _parse_action(content, task_name)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return _fallback_action(task_name)


def _build_system_prompt(task_name: str) -> str:
    """Build the system prompt for the LLM."""
    base = (
        "You are an expert email triage assistant. You process emails and make "
        "triage decisions. Always respond with a valid JSON object.\n\n"
        "Valid categories: spam, urgent, normal, low_priority, newsletter\n"
        "Valid priority levels: 1 (highest) to 5 (lowest)\n"
        "Valid departments: engineering, sales, support, hr, legal, marketing, finance, executive\n\n"
    )

    if task_name == "classification":
        return base + (
            "Your task is to classify each email and assign priority.\n"
            'Respond with JSON: {"classify": "<category>", "priority": <1-5>}\n'
            "Guidelines:\n"
            "- spam: unsolicited, scam, phishing emails\n"
            "- urgent: time-sensitive, from executives, critical issues\n"
            "- normal: standard business communications\n"
            "- low_priority: social, non-urgent internal\n"
            "- newsletter: digests, subscriptions, promotional\n"
        )
    elif task_name == "routing":
        return base + (
            "Your task is to classify, prioritize, and route each email.\n"
            'Respond with JSON: {"classify": "<category>", "priority": <1-5>, "route_to": "<department>"}\n'
            "Route based on the email's content and who should handle it.\n"
        )
    else:  # full_triage
        return base + (
            "Your task is to fully triage each email: classify, prioritize, route, "
            "and draft a response.\n"
            'Respond with JSON: {"classify": "<category>", "priority": <1-5>, '
            '"route_to": "<department>", "response_draft": "<your response>"}\n'
            "Draft professional, context-appropriate responses. For spam/phishing, "
            "warn about the threat. For newsletters, no response needed (set to empty string).\n"
        )


def _build_user_prompt(
    email_obs: Dict[str, Any],
    history: List[str],
    last_feedback: str,
    last_reward: float,
) -> str:
    """Build the user prompt with current email and context."""
    parts = []

    if last_feedback:
        parts.append(f"Previous feedback: {last_feedback} (reward: {last_reward:.2f})")

    parts.append("Current email to triage:")
    parts.append(f"From: {email_obs.get('sender_name', '')} <{email_obs.get('sender', '')}>")
    parts.append(f"Subject: {email_obs.get('subject', '')}")
    parts.append(f"Date: {email_obs.get('timestamp', '')}")
    parts.append(f"Has attachments: {email_obs.get('has_attachments', False)}")
    parts.append(f"Is reply: {email_obs.get('is_reply', False)}")
    if email_obs.get("thread_count", 0) > 1:
        parts.append(f"Thread length: {email_obs.get('thread_count', 0)} emails")
    parts.append(f"\nBody:\n{email_obs.get('body', '')}")
    parts.append(
        f"\nProgress: {email_obs.get('emails_processed', 0)}/{email_obs.get('total_emails', 0)} "
        f"emails processed, {email_obs.get('emails_remaining', 0)} remaining"
    )

    parts.append("\nRespond with JSON only. No explanation needed.")

    return "\n".join(parts)


def _parse_action(content: str, task_name: str) -> Dict[str, Any]:
    """Parse the LLM response into an action dictionary."""
    # Try to extract JSON from the response
    content = content.strip()

    # Handle markdown code blocks
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(content[start:end])
            except json.JSONDecodeError:
                return _fallback_action(task_name)
        else:
            return _fallback_action(task_name)

    action = {}
    if "classify" in data:
        action["classify"] = str(data["classify"]).lower().strip()
    if "priority" in data:
        try:
            action["priority"] = int(data["priority"])
        except (ValueError, TypeError):
            action["priority"] = 3
    if "route_to" in data and task_name in ("routing", "full_triage"):
        action["route_to"] = str(data["route_to"]).lower().strip()
    if "response_draft" in data and task_name == "full_triage":
        action["response_draft"] = str(data["response_draft"])

    return action if action else _fallback_action(task_name)


def _fallback_action(task_name: str) -> Dict[str, Any]:
    """Return a safe fallback action."""
    action = {"classify": "normal", "priority": 3}
    if task_name in ("routing", "full_triage"):
        action["route_to"] = "support"
    if task_name == "full_triage":
        action["response_draft"] = "Thank you for your email. We will review and respond shortly."
    return action


# ── Main loop ──────────────────────────────────────────────────────────


async def run_task(task_config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single task and return results."""
    # Import here to allow the script to be parsed without openenv installed
    from client import EmailTriageEnv
    from models import EmailTriageAction

    task_name = task_config["name"]
    max_steps = task_config["max_steps"]

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Connect to the environment
    env = await EmailTriageEnv.from_docker_image(IMAGE_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        obs = result.observation
        last_feedback = obs.feedback if hasattr(obs, "feedback") else ""
        last_reward = 0.0

        for step in range(1, max_steps + 1):
            if result.done:
                break

            # Build observation dict for the LLM
            email_obs = {
                "email_id": obs.email_id,
                "sender": obs.sender,
                "sender_name": obs.sender_name,
                "subject": obs.subject,
                "body": obs.body,
                "timestamp": obs.timestamp,
                "has_attachments": obs.has_attachments,
                "is_reply": obs.is_reply,
                "thread_count": obs.thread_count,
                "task_name": obs.task_name,
                "emails_remaining": obs.emails_remaining,
                "emails_processed": obs.emails_processed,
                "total_emails": obs.total_emails,
            }

            # Get action from LLM
            action_dict = get_model_action(
                client, task_name, email_obs, history, last_feedback, last_reward
            )

            # Create action
            action = EmailTriageAction(
                classify=action_dict.get("classify"),
                priority=action_dict.get("priority"),
                route_to=action_dict.get("route_to"),
                response_draft=action_dict.get("response_draft"),
            )

            # Step the environment
            result = await env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_feedback = obs.feedback if hasattr(obs, "feedback") else ""
            last_reward = reward

            log_step(
                step=step,
                action=json.dumps(action_dict),
                reward=reward,
                done=done,
                error=error,
            )

            history.append(
                f"Step {step}: {json.dumps(action_dict)} -> reward {reward:+.2f}"
            )

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task": task_name,
        "score": score,
        "success": success,
        "steps": steps_taken,
        "rewards": rewards,
    }


async def main() -> None:
    """Run all tasks and report results."""
    print(f"[DEBUG] Starting Email Triage baseline inference", flush=True)
    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)
    print(f"[DEBUG] IMAGE_NAME={IMAGE_NAME}", flush=True)

    results = []
    for task_config in TASKS:
        print(f"\n{'='*60}", flush=True)
        print(f"[DEBUG] Running task: {task_config['name']} - {task_config['description']}", flush=True)
        print(f"{'='*60}", flush=True)

        result = await run_task(task_config)
        results.append(result)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("[DEBUG] === BASELINE RESULTS SUMMARY ===", flush=True)
    print(f"{'='*60}", flush=True)
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(
            f"[DEBUG] {r['task']:20s} | score: {r['score']:.4f} | "
            f"steps: {r['steps']} | {status}",
            flush=True,
        )

    overall = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"[DEBUG] {'Overall':20s} | score: {overall:.4f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
