from env.environment import EmailTriageEnv
from env.models import Action


def run_medium_task(env: EmailTriageEnv = None) -> float:
    """
    Medium Task: Classify email AND choose correct action.
    Returns score between 0.0 and 1.0
    """
    if env is None:
        env = EmailTriageEnv(task_id="medium")

    observation = env.reset()
    total_reward = 0.0
    steps = 0

    while not env._done:
        for email in observation.emails:
            if email.id in env._actions_taken:
                continue

            spam_keywords = ["won", "prize", "click", "free",
                             "make money", "giveaway", "claim"]
            is_spam = any(w in email.subject.lower()
                          for w in spam_keywords)

            # Decide action
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
            )

            result = env.step(action)
            total_reward += result.reward
            observation = result.observation
            steps += 1
            break

    final_score = max(0.0001, min(0.9999, total_reward / max(1, len(observation.emails))))
    print(f"[MEDIUM] Steps: {steps} | Score: {final_score:.4f}")
    return final_score


def grade_medium(actions: list, emails: list) -> float:
    """
    Deterministic grader for medium task.
    Checks classification AND action choice.
    Returns score strictly between 0 and 1
    """
    if not emails:
        return 0.0001

    total = 0.0
    spam_keywords = ["won", "prize", "click", "free",
                     "make money", "giveaway", "claim"]

    for action, email in zip(actions, emails):
        is_spam = any(w in email.subject.lower()
                      for w in spam_keywords)
        expected_class = "spam" if is_spam else "not-spam"
        expected_action = (
            "archive" if is_spam else
            "escalate" if email.is_urgent else
            "reply"
        )

        score = 0.0
        if action.classification == expected_class:
            score += 0.5
        if action.action_type == expected_action:
            score += 0.5
        total += score

    return max(0.0001, min(0.9999, total / len(emails)))