import os
import sys
import json
from typing import Optional
from openai import OpenAI
from env.environment import EmailTriageEnv
from env.models import Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct:groq")
BENCHMARK = os.getenv("BENCHMARK", "email-triage")

MAX_STEPS = 10
TEMPERATURE = 0.2
MAX_TOKENS = 300
TASKS = ["easy", "medium", "hard"]


def build_prompt(observation) -> str:
    email = None
    for e in observation.emails:
        email = e
        break
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
  "response_text": ""
}}
No explanation. Just the JSON.
""".strip()


def parse_action(response_text: str, fallback_email_id: str) -> Action:
    try:
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
        return Action(
            email_id=fallback_email_id,
            action_type="classify",
            classification="not-spam",
        )


def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


def run_task(client: OpenAI, task_id: str) -> float:
    env = None
    observation = None
    total_reward = 0.0
    steps = 0
    rewards: list[float] = []
    success = False
    score = 0.0001
    error_message: Optional[str] = None

    log_start(task=task_id, env_name=BENCHMARK, model=MODEL_NAME)

    try:
        env = EmailTriageEnv(task_id=task_id)
        observation = env.reset()

        while not env._done and steps < MAX_STEPS:
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
                        {
                            "role": "system",
                            "content": "You are an expert email triage assistant. Always respond with valid JSON only.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as exc:
                response_text = json.dumps(
                    {
                        "email_id": email.id,
                        "action_type": "classify",
                        "classification": "not-spam",
                        "priority": "medium",
                    }
                )
                error_message = str(exc)

            action = parse_action(response_text, email.id)
            result = env.step(action)

            steps += 1
            reward = result.reward
            total_reward += reward
            rewards.append(reward)

            action_str = "|".join(
                filter(None, [action.action_type, action.classification, action.priority])
            )
            log_step(
                step=steps,
                action=action_str,
                reward=reward,
                done=result.done,
                error=error_message,
            )

            observation = result.observation

        score = max(0.0001, min(0.9999, total_reward / max(1, len(observation.emails) if observation else 1)))
        success = score > 0.0
        return score

    finally:
        if env is not None:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception as exc:
                    print(f"[DEBUG] env.close error: {exc}", file=sys.stderr, flush=True)
        log_end(success=success, steps=steps, score=score, rewards=rewards)


def main():
    if not API_KEY:
        raise RuntimeError("HF_TOKEN or OPENAI_API_KEY must be set for inference")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_id in TASKS:
        run_task(client, task_id)


if __name__ == "__main__":
    main()