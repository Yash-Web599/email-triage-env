"""
inference.py  —  Baseline inference script for Email Triage OpenEnv
MANDATORY: Must be in root directory
Uses OpenAI client as required by hackathon rules
"""

import os
import json
from openai import OpenAI
from env.environment import EmailTriageEnv
from env.models import Action

# ── Environment variables (mandatory) ───────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

# ── Constants ────────────────────────────────────────────────────
MAX_STEPS   = 10
TEMPERATURE = 0.2
MAX_TOKENS  = 300

TASKS = ["easy", "medium", "hard"]


def build_prompt(observation) -> str:
    """Build a prompt for the LLM from the current observation."""
    email = observation.emails[0] if observation.emails else None
    if not email:
        return "No emails to process."

    return f"""
You are an email triage assistant. Your goal: {observation.goal}

Email to process:
- ID: {email.id}
- Subject: {email.subject}
- Sender: {email.sender}
- Body: {email.body}
- Urgent: {email.is_urgent}

Respond ONLY with a valid JSON object like this:
{{
  "email_id": "{email.id}",
  "action_type": "classify",
  "classification": "spam or not-spam",
  "priority": "low or medium or high",
  "response_text": "optional reply text"
}}
No explanation. Just the JSON.
""".strip()


def parse_action(response_text: str, fallback_email_id: str) -> Action:
    """Parse LLM response into an Action object."""
    try:
        # Clean up response
        text = response_text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)
        return Action(
            email_id=data.get("email_id", fallback_email_id),
            action_type=data.get("action_type", "classify"),
            classification=data.get("classification"),
            priority=data.get("priority"),
            response_text=data.get("response_text"),
        )
    except Exception:
        # Fallback action if parsing fails
        return Action(
            email_id=fallback_email_id,
            action_type="classify",
            classification="not-spam",
        )


def run_task(client: OpenAI, task_id: str) -> float:
    """Run one task and return final score."""
    print(f"\n{'='*40}")
    print(f"Running task: {task_id.upper()}")
    print(f"{'='*40}")

    env = EmailTriageEnv(task_id=task_id)
    observation = env.reset()
    total_reward = 0.0
    steps = 0

    while not env._done and steps < MAX_STEPS:
        # Find first unprocessed email
        email = None
        for e in observation.emails:
            if e.id not in env._actions_taken:
                email = e
                break

        if email is None:
            break

        prompt = build_prompt(observation)

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert email triage assistant. Always respond with valid JSON only."},
                    {"role": "user",   "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as e:
            print(f"  LLM error: {e}. Using fallback.")
            response_text = json.dumps({
                "email_id": email.id,
                "action_type": "classify",
                "classification": "not-spam",
                "priority": "medium",
            })

        action = parse_action(response_text, email.id)
        result = env.step(action)

        print(f"  Step {steps+1}: {action.action_type} on {email.id} "
              f"→ reward {result.reward:+.2f}")

        total_reward += result.reward
        observation  = result.observation
        steps += 1

    final_score = round(
        min(1.0, total_reward / max(1, len(observation.emails))), 2
    )
    print(f"  Final score [{task_id}]: {final_score:.2f}")
    return final_score


def main():
    print("Email Triage OpenEnv — Baseline Inference")
    print("==========================================")

    if not API_KEY:
        print("⚠️  WARNING: No API key found.")
        print("   Set HF_TOKEN or OPENAI_API_KEY environment variable.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")

    scores = {}
    for task_id in TASKS:
        scores[task_id] = run_task(client, task_id)

    print("\n==========================================")
    print("FINAL SCORES:")
    for task_id, score in scores.items():
        print(f"  {task_id.upper():10s}: {score:.2f}")
    print(f"  AVERAGE   : {sum(scores.values())/len(scores):.2f}")
    print("==========================================")


if __name__ == "__main__":
    main()