import os
import json
from openai import OpenAI
from env.environment import EmailTriageEnv
from env.models import Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct:groq")

MAX_STEPS   = 10
TEMPERATURE = 0.2
MAX_TOKENS  = 300
TASKS       = ["easy", "medium", "hard"]


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


def run_task(client: OpenAI, task_id: str) -> float:
    print(f"[START] task={task_id}", flush=True)

    env = EmailTriageEnv(task_id=task_id)
    observation = env.reset()
    total_reward = 0.0
    steps = 0

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
                    {"role": "system", "content": "You are an expert email triage assistant. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as e:
            response_text = json.dumps({
                "email_id": email.id,
                "action_type": "classify",
                "classification": "not-spam",
                "priority": "medium",
            })

        action = parse_action(response_text, email.id)
        result = env.step(action)

        steps += 1
        total_reward += result.reward

        print(f"[STEP] step={steps} reward={result.reward:.2f}", flush=True)

        observation = result.observation

    final_score = max(0.0001, min(0.9999, total_reward / max(1, len(observation.emails))))

    print(f"[END] task={task_id} score={final_score:.4f} steps={steps}", flush=True)
    return final_score


def main():
    print("[START] task=all", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")

    scores = {}
    for task_id in TASKS:
        scores[task_id] = run_task(client, task_id)

    avg = sum(scores.values()) / len(scores)
    print(f"[END] task=all score={avg:.2f} steps={len(TASKS)}", flush=True)


if __name__ == "__main__":
    main()