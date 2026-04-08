from env.environment import EmailTriageEnv
from env.models import Action


def run_hard_task(env: EmailTriageEnv = None) -> float:
    """
    Hard Task: Triage full inbox of 5 emails.
    Classify + prioritize + action for each.
    Returns score between 0.0 and 1.0
    """
    if env is None:
        env = EmailTriageEnv(task_id="hard")

    observation = env.reset()
    total_reward = 0.0
    steps = 0

    while not env._done:
        acted = False
        for email in observation.emails:
            if email.id in env._actions_taken:
                continue

            spam_keywords = ["won", "prize", "click", "free",
                             "make money", "giveaway", "claim"]
            is_spam = any(w in email.subject.lower()
                          for w in spam_keywords)

            # Priority logic
            if email.is_urgent:
                priority = "high"
            elif not is_spam:
                priority = "medium"
            else:
                priority = "low"

            # Action logic
            if is_spam:
                action_type = "archive"
            elif email.is_urgent:
                action_type = "escalate"
            else:
                action_type = "reply"

            action = Action(
                email_id=email.id,
                action_type=action_type,
                classification="spam" if is_spam else "not-spam",
                priority=priority,
            )

            result = env.step(action)
            total_reward += result.reward
            observation = result.observation
            steps += 1
            acted = True
            break

        if not acted:
            break

    final_score = max(0.01, min(0.99, total_reward / max(1, len(observation.emails))))
    print(f"[HARD] Steps: {steps} | Score: {final_score:.2f}")
    return final_score


def grade_hard(actions: list, emails: list) -> float:
    """
    Deterministic grader for hard task.
    Checks classification + priority + action.
    Returns score 0.0 - 1.0
    """
    if not emails:
        return 0.0

    total = 0.0
    spam_keywords = ["won", "prize", "click", "free",
                     "make money", "giveaway", "claim"]

    for action, email in zip(actions, emails):
        is_spam = any(w in email.subject.lower()
                      for w in spam_keywords)

        expected_class = "spam" if is_spam else "not-spam"
        expected_priority = (
            "high" if email.is_urgent else
            "medium" if not is_spam else
            "low"
        )
        expected_action = (
            "archive" if is_spam else
            "escalate" if email.is_urgent else
            "reply"
        )

        score = 0.0
        if action.classification == expected_class:
            score += 0.4
        if action.priority == expected_priority:
            score += 0.2
        if action.action_type == expected_action:
            score += 0.4
        total += score

    return max(0.01, min(0.99, total / len(emails)))