import random
from typing import Optional
from .models import (
    Observation, Action, Reward, StepResult, Email
)
from .data import generate_email, generate_inbox


TASK_GOALS = {
    "easy": "Classify each email as 'spam' or 'not-spam'.",
    "medium": "Classify each email AND choose the correct action: reply, archive, or escalate.",
    "hard": "Triage the full inbox: classify, prioritize (low/medium/high), and choose the correct action for all 5 emails.",
}

MAX_STEPS = {
    "easy": 3,
    "medium": 5,
    "hard": 10,
}


class EmailTriageEnv:
    def __init__(self, task_id: str = "easy"):
        assert task_id in ("easy", "medium", "hard"), \
            f"task_id must be easy/medium/hard, got {task_id}"
        self.task_id = task_id
        self._inbox: list = []
        self._step_count: int = 0
        self._actions_taken: dict = {}
        self._done: bool = False

    # ── reset() ─────────────────────────────────────────────────
    def reset(self) -> Observation:
        """Start a fresh episode."""
        if self.task_id == "easy":
            self._inbox = [generate_email("email_0")]
        elif self.task_id == "medium":
            self._inbox = [generate_email("email_0"),
                           generate_email("email_1")]
        else:
            self._inbox = generate_inbox(size=5)

        self._step_count = 0
        self._actions_taken = {}
        self._done = False
        return self._make_observation()

    # ── step() ──────────────────────────────────────────────────
    def step(self, action: Action) -> StepResult:
        """Take one action and return result."""
        if self._done:
            return StepResult(
                observation=self._make_observation(),
                reward=0.0,
                done=True,
                info={"error": "Episode already done"},
            )

        self._step_count += 1
        reward_obj = self._calculate_reward(action)
        self._actions_taken[action.email_id] = action

        # Episode ends when all emails are acted on OR max steps reached
        all_done = all(
            e.id in self._actions_taken for e in self._inbox
        )
        max_reached = self._step_count >= MAX_STEPS[self.task_id]
        self._done = all_done or max_reached

        return StepResult(
            observation=self._make_observation(last_action=action.action_type),
            reward=reward_obj.value,
            done=self._done,
            info={"reward_reason": reward_obj.reason,
                  "step": self._step_count},
        )

    # ── state() ─────────────────────────────────────────────────
    def state(self) -> dict:
        """Return full current state."""
        return {
            "task_id": self.task_id,
            "step": self._step_count,
            "done": self._done,
            "inbox_size": len(self._inbox),
            "actions_taken": {
                k: v.dict() for k, v in self._actions_taken.items()
            },
        }

    # ── reward logic ────────────────────────────────────────────
    def _calculate_reward(self, action: Action) -> Reward:
        email = next((e for e in self._inbox
                      if e.id == action.email_id), None)

        if email is None:
            return Reward(value=0.0, reason="Email ID not found")

        score = 0.0
        reasons = []
        is_spam = any(w in email.subject.lower() for w in
                      ["won", "prize", "click", "free", "urgent: your",
                       "make money", "giveaway", "claim"])

        # ── Classification score (all tasks) ────────────────────
        if action.classification:
            correct_class = "spam" if is_spam else "not-spam"
            if action.classification == correct_class:
                score += 0.4
                reasons.append("correct classification +0.4")
            else:
                score -= 0.2
                reasons.append("wrong classification -0.2")

        # ── Action score (medium + hard) ─────────────────────────
        if self.task_id in ("medium", "hard") and action.action_type:
            correct_action = self._expected_action(email, is_spam)
            if action.action_type == correct_action:
                score += 0.3
                reasons.append(f"correct action '{correct_action}' +0.3")
            elif action.action_type == "delete" and not is_spam:
                score -= 0.3
                reasons.append("deleted important email -0.3")
            else:
                reasons.append("suboptimal action +0.0")

        # ── Priority score (hard only) ───────────────────────────
        if self.task_id == "hard" and action.priority:
            correct_priority = "high" if email.is_urgent else (
                "medium" if not is_spam else "low"
            )
            if action.priority == correct_priority:
                score += 0.2
                reasons.append("correct priority +0.2")
            else:
                reasons.append("wrong priority +0.0")

        # ── Efficiency bonus ─────────────────────────────────────
        if self._step_count <= MAX_STEPS[self.task_id] // 2:
            score += 0.1
            reasons.append("efficiency bonus +0.1")

        # clamp to [0.001, 0.999]
        score = max(0.001, min(0.999, score))
        return Reward(value=round(score, 2),
                      reason=" | ".join(reasons) or "no action matched")

    def _expected_action(self, email: Email, is_spam: bool) -> str:
        if is_spam:
            return "archive"
        if email.is_urgent:
            return "escalate"
        return "reply"

    def _make_observation(self,
                          last_action: Optional[str] = None) -> Observation:
        return Observation(
            emails=self._inbox,
            current_step=self._step_count,
            max_steps=MAX_STEPS[self.task_id],
            task_id=self.task_id,
            goal=TASK_GOALS[self.task_id],
            last_action=last_action,
        )