from fastapi import FastAPI, HTTPException
from env.environment import EmailTriageEnv
from env.models import Action

app = FastAPI(title="Email Triage OpenEnv")

envs = {
    "easy":   EmailTriageEnv(task_id="easy"),
    "medium": EmailTriageEnv(task_id="medium"),
    "hard":   EmailTriageEnv(task_id="hard"),
}

@app.get("/")
def root():
    return {"message": "Email Triage OpenEnv is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ── OpenEnv standard endpoints ───────────────────────────────────

@app.post("/reset")
def reset_default():
    """Default reset — uses easy task"""
    obs = envs["easy"].reset()
    return obs

@app.post("/reset/{task_id}")
def reset(task_id: str):
    if task_id not in envs:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}")
    obs = envs[task_id].reset()
    return obs

@app.post("/step")
def step_default(action: Action):
    """Default step — uses easy task"""
    result = envs["easy"].step(action)
    return result

@app.post("/step/{task_id}")
def step(task_id: str, action: Action):
    if task_id not in envs:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}")
    result = envs[task_id].step(action)
    return result

@app.get("/state")
def state_default():
    """Default state — uses easy task"""
    return envs["easy"].state()

@app.get("/state/{task_id}")
def state(task_id: str):
    if task_id not in envs:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}")
    return envs[task_id].state()

@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": "easy",   "difficulty": "easy"},
            {"id": "medium", "difficulty": "medium"},
            {"id": "hard",   "difficulty": "hard"},
        ]
    }