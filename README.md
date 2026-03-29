---
title: Email Triage Env
emoji: 🌍
colorFrom: indigo
colorTo: red
sdk: docker
tags:
  - openenv
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


# 📧 Email Triage OpenEnv

A real-world OpenEnv environment where an AI agent learns to triage emails —
classifying, prioritizing, and acting on them just like a human would.

---

## 🌍 Environment Description

Email triage is one of the most common real-world tasks humans do daily.
This environment simulates a realistic email inbox where an AI agent must:
- Classify emails as spam or not-spam
- Choose the correct action (reply, archive, escalate, delete, flag)
- Prioritize emails by urgency (low, medium, high)

This environment is valuable for training and evaluating AI agents on
real-world decision-making with partial progress signals.

---

## 🎯 Tasks

| Task | Difficulty | Description | Max Steps |
|------|-----------|-------------|-----------|
| `easy` | 🟢 Easy | Classify a single email as spam or not-spam | 3 |
| `medium` | 🟡 Medium | Classify + choose correct action for 2 emails | 5 |
| `hard` | 🔴 Hard | Triage full inbox of 5 emails with priorities | 10 |

---

## 👁️ Observation Space

Each observation contains:
- `emails` — list of Email objects (id, subject, sender, body, timestamp, is_urgent)
- `current_step` — current step number
- `max_steps` — maximum steps allowed
- `task_id` — current task (easy/medium/hard)
- `goal` — natural language description of what agent must do
- `last_action` — last action taken by agent
- `last_action_error` — any error from last action

---

## ⚡ Action Space

Each action contains:
- `email_id` — ID of email to act on
- `action_type` — one of: `classify`, `reply`, `archive`, `escalate`, `delete`, `flag`
- `classification` — `spam` or `not-spam`
- `priority` — `low`, `medium`, or `high`
- `response_text` — optional reply text

---

## 🏆 Reward Function

Rewards are given at every step (not just end of episode):

| Action | Reward |
|--------|--------|
| Correct classification | +0.4 |
| Correct action chosen | +0.3 |
| Correct priority assigned | +0.2 |
| Efficiency bonus (fast completion) | +0.1 |
| Wrong classification | -0.2 |
| Deleting important email | -0.3 |

---

## 📊 Baseline Scores

Scores achieved by `meta-llama/Llama-3.3-70B-Instruct` via HuggingFace router:

| Task | Score |
|------|-------|
| Easy | 0.50 |
| Medium | 1.00 |
| Hard | 1.00 |
| **Average** | **0.83** |

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.10+
- Docker
- HuggingFace account with API token

### Local Setup

**1. Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/email-triage-env.git
cd email-triage-env
```

**2. Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Run the server:**
```bash
uvicorn app:app --reload --port 8000
```

**5. Visit the API docs:**
```
http://localhost:8000/docs
```

---

### Run Baseline Inference
```bash
export HF_TOKEN="your_huggingface_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct:groq"
python inference.py
```

---

### Docker Setup
```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```

---

## 📁 Project Structure
```
email-triage-env/
├── inference.py        # Baseline inference script
├── app.py              # FastAPI server
├── openenv.yaml        # OpenEnv metadata
├── Dockerfile          # Container setup
├── requirements.txt    # Dependencies
├── README.md           # This file
├── env/
│   ├── models.py       # Pydantic models
│   ├── data.py         # Email data generator
│   └── environment.py  # Core environment
└── tasks/
    ├── task_easy.py    # Easy task + grader
    ├── task_medium.py  # Medium task + grader
    └── task_hard.py    # Hard task + grader
```

---

## 🔗 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Server status |
| GET | `/tasks` | List all tasks |
| POST | `/reset/{task_id}` | Start new episode |
| POST | `/step/{task_id}` | Take one action |
| GET | `/state/{task_id}` | Get current state |