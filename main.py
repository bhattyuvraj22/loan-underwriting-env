"""
FastAPI application for Loan Underwriting OpenEnv.
Implements the full OpenEnv HTTP interface:
  GET  /          → health check (required for HF Space ping)
  GET  /health    → health check
  GET  /tasks     → list all tasks
  POST /reset     → start new episode, returns Observation
  POST /step      → submit decisions, returns StepResult
  GET  /state     → get current state without advancing episode
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from env.models import AgentAction, ResetRequest
from env.state import SessionManager
from typing import Optional


app = FastAPI(
    title="Loan Underwriting OpenEnv",
    version="1.0.0",
    description=(
        "A real-world OpenEnv environment for loan underwriting. "
        "An AI agent evaluates mortgage applicants and makes approve/reject/escalate decisions."
    ),
)
sessions = SessionManager()


# ── Root endpoint — required for HF Space automated ping (must return 200) ────
@app.get("/", summary="Health check and environment info")
def root():
    return {
        "status": "ok",
        "env": "loan-underwriting-env",
        "version": "1.0.0",
        "tasks": ["task_1_easy", "task_2_medium", "task_3_hard"],
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "tasks": "GET /tasks",
        },
    }


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "env": "loan-underwriting-env", "version": "1.0.0"}


@app.get("/tasks", summary="List all available tasks")
def list_tasks():
    return sessions.list_tasks()


@app.post("/step", summary="Submit agent decisions")
def step(action: AgentAction):
    """
    Submit decisions for all applicants in the current episode.
    Returns reward, done=True, and detailed grader breakdown.
    """
    try:
        result = sessions.step(action)
        return {
            "observation": result.observation.model_dump(),
            "reward": result.reward,
            "done": result.done,
            "info": result.info,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", summary="Get current episode state")
def state(task_id: str):
    """
    Get the current state of an active episode without advancing it.
    Useful for inspecting state between steps.
    """
    try:
        obs = sessions.get_state(task_id)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

def main():
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()