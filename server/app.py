"""
FastAPI app for mechinterp-env.
create_app() generates all OpenEnv-compliant HTTP endpoints automatically.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from models import MechInterpAction, MechInterpObservation
from server.environment import MechInterpEnvironment

# One global environment instance (sessions are reset on each /reset call)
env = MechInterpEnvironment()

app = FastAPI(
    title="mechinterp-env",
    description="RL environment for mechanistic interpretability circuit debugging.",
    version="1.0.0",
)


@app.post("/reset")
async def reset(request: Request):
    """Reset the environment. Optionally pass task_id in request body."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    task_id = body.get("task_id", "head-identification")
    obs = env.reset(task_id=task_id)
    return obs.model_dump()


@app.post("/step")
async def step(request: Request):
    """Take one action in the environment."""
    try:
        body = await request.json()
        action = MechInterpAction(**body)
    except (ValidationError, Exception) as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid action: {str(e)}"}
        )

    obs = env.step(action)
    return obs.model_dump()


@app.get("/state")
async def state():
    """Return current environment state (does not advance episode)."""
    return env.state


@app.get("/health")
async def health():
    return {"status": "ok", "environment": "mechinterp-env", "version": "1.0.0"}


@app.get("/")
async def root():
    return {
        "name":        "mechinterp-env",
        "description": "RL environment for mechanistic interpretability circuit debugging.",
        "tasks":       ["head-identification", "circuit-localization", "full-hypothesis"],
        "endpoints":   ["/reset", "/step", "/state", "/health"],
    }


def main():
    """Entry point for the server. Called by the [project.scripts] entry point."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, workers=1)


if __name__ == "__main__":
    main()
