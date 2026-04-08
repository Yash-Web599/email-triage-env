from env.environment import EmailTriageEnv
from env.models import Action


def run_easy_task(env: EmailTriageEnv = None) -> float:
    """
    Easy Task: Classify a single email as spam or not-spam.
    Returns a score between 0.0 and 1.0
    """
    if env is None:
        env = EmailTriageEnv(task_id="easy")

    observation = env.reset()
    total_reward = 0.0
    steps = 0

    while not env._done:
        email = observation.emails[0]

        # Simple rule-based grader for baseline
        spam_keywords = ["won", "prize", "click", "free",
                         "make money", "giveaway", "claim"]
        is_spam = any(w in email.subject.lower()
                      for w in spam_keywords)

        action = Action(
            email_id=email.id,
            action_type="classify",
            classification="spam" if is_spam else "not-spam",
        )

        result = env.step(action)
        total_reward += result.reward
        observation = result.observation
        steps += 1

    final_score = max(0.0001, min(0.9999, total_reward))
    print(f"[EASY] Steps: {steps} | Score: {final_score:.4f}")
    return final_score


def grade_easy(actions: list, emails: list) -> float:
    """
    Deterministic grader: checks if each email was
    classified correctly. Returns score strictly between 0 and 1
    """
    if not emails:
        return 0.0001

    correct = 0
    spam_keywords = ["won", "prize", "click", "free",
                     "make money", "giveaway", "claim"]

    for action, email in zip(actions, emails):
        is_spam = any(w in email.subject.lower()
                      for w in spam_keywords)
        expected = "spam" if is_spam else "not-spam"
        if action.classification == expected:
            correct += 1

    return max(0.0001, min(0.9999, correct / len(emails)))