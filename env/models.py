from pydantic import BaseModel
from typing import Optional, List

# ── What the agent SEES ──────────────────────────────────────────
class Email(BaseModel):
    id: str
    subject: str
    sender: str
    body: str
    timestamp: str
    is_urgent: bool = False

class Observation(BaseModel):
    emails: List[Email]
    current_step: int
    max_steps: int
    task_id: str
    goal: str
    last_action: Optional[str] = None
    last_action_error: Optional[str] = None

# ── What the agent DOES ──────────────────────────────────────────
class Action(BaseModel):
    email_id: str
    action_type: str        # "classify", "reply", "archive", "escalate", "delete", "flag"
    classification: Optional[str] = None   # "spam" or "not-spam"
    priority: Optional[str] = None         # "low", "medium", "high"
    response_text: Optional[str] = None

# ── What the agent GETS BACK ─────────────────────────────────────
class Reward(BaseModel):
    value: float            # 0.0 to 1.0
    reason: str

# ── Step result ──────────────────────────────────────────────────
class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict