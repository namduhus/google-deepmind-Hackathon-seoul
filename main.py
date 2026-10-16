"""
FastAPI Backend for Sim-to-Action 3D VLA Co-Pilot & Kinematics Verifier
Serves:
  - GET  /                    -> Interactive 3D WebGPU / Three.js Co-Pilot UI
  - GET  /api/health          -> Health & environment status
  - GET  /api/scene           -> Default 3D scene & preset scenarios
  - POST /api/plan-and-verify -> Runs Gemini 4 Argon + Antigravity Self-Repair Loop
  - POST /api/verify-custom   -> Verifies custom joint waypoints in the sandbox
"""

from __future__ import annotations
import copy
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from kinematics_verifier import DEFAULT_SCENE, JOINT_LIMITS_DEG, verify_trajectory
from vla_orchestrator import run_autonomous_verification_loop

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Sim-to-Action 3D VLA Co-Pilot & Kinematics Verifier",
    description="Track 2: Gemini 4 Argon in Hard Tech — Closed-Loop 6-DoF Trajectory Planning & Collision Self-Repair",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PRESET_SCENARIOS = {
    "standard_pick_place": {
        "title": "Scenario A: Tall Cylinder Bypass Pick & Place",
        "instruction": "Pick up the red can, avoid the tall central cylinder obstacle via a 52cm overhead arch, and place it into the green goal basket.",
        "scene": copy.deepcopy(DEFAULT_SCENE),
    },
    "narrow_corridor": {
        "title": "Scenario B: Dual-Obstacle High-Clearance Slalom",
        "instruction": "S-curve zig-zag slalom mode: Pick up the red can, swing far forward toward the front edge, cut across the center, and sweep far out to the back-left before placing it into the basket.",
        "scene": {
            "table": {"min": [-0.6, -0.6, -0.05], "max": [0.6, 0.6, 0.0]},
            "target_can": {"pos": [0.32, 0.15, 0.06], "radius": 0.035, "height": 0.12},
            "obstacle_cylinder": {"pos": [0.07, 0.24, 0.145], "radius": 0.055, "height": 0.29},
            "secondary_obstacle": {"pos": [-0.13, 0.24, 0.12], "radius": 0.045, "height": 0.24},
            "goal_basket": {"pos": [-0.28, 0.28, 0.05], "radius": 0.09, "height": 0.10},
            "clearance_margin": 0.035,
        },
    },
    "precision_sort": {
        "title": "Scenario C: Far-Reach Singularity-Safe Transfer",
        "instruction": "Pick up the far-reach red can without violating joint limits or outer workspace singularities, bypass the obstacles at 54cm altitude, and place it into the basket.",
        "scene": {
            "table": {"min": [-0.6, -0.6, -0.05], "max": [0.6, 0.6, 0.0]},
            "target_can": {"pos": [0.35, 0.18, 0.06], "radius": 0.035, "height": 0.12},
            "obstacle_cylinder": {"pos": [0.08, 0.25, 0.135], "radius": 0.055, "height": 0.27},
            "secondary_obstacle": {"pos": [-0.14, 0.25, 0.11], "radius": 0.045, "height": 0.22},
            "goal_basket": {"pos": [-0.32, 0.25, 0.05], "radius": 0.09, "height": 0.10},
            "clearance_margin": 0.035,
        },
    },
}


class PlanRequest(BaseModel):
    instruction: str = Field(
        default="Pick up the red can, avoid the tall central cylinder obstacle via a 52cm overhead arch, and place it into the green goal basket."
    )
    scenario_id: str = Field(default="standard_pick_place")
    scene: dict[str, Any] | None = None
    system_prompt_override: str | None = None
    kinematic_overrides: dict[str, Any] | None = None
    auto_self_repair: bool = True


class VerifyCustomRequest(BaseModel):
    waypoints: list[dict[str, Any]]
    scene: dict[str, Any] | None = None


@app.get("/api/health")
async def health() -> dict[str, Any]:
    has_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    return {
        "status": "ok",
        "project": "Sim-to-Action 3D VLA Co-Pilot & Kinematics Verifier",
        "track": "Track 2: Gemini 4 Argon in hard tech",
        "gemini_api_configured": has_key,
        "sandbox_env": "antigravity-preview-09-2026",
        "joint_limits_deg": JOINT_LIMITS_DEG,
    }


@app.get("/api/scene")
async def get_scene_presets() -> dict[str, Any]:
    return {
        "default_scene": DEFAULT_SCENE,
        "presets": PRESET_SCENARIOS,
        "joint_limits_deg": JOINT_LIMITS_DEG,
    }


@app.post("/api/plan-and-verify")
async def plan_and_verify(req: PlanRequest) -> dict[str, Any]:
    preset = PRESET_SCENARIOS.get(req.scenario_id, PRESET_SCENARIOS["standard_pick_place"])
    active_scene = req.scene if req.scene else copy.deepcopy(preset["scene"])
    instruction = req.instruction.strip() or preset["instruction"]
    result = run_autonomous_verification_loop(
        instruction=instruction,
        scene=active_scene,
        system_prompt_override=req.system_prompt_override,
        kinematic_overrides=req.kinematic_overrides,
        auto_self_repair=req.auto_self_repair,
    )
    return result


@app.post("/api/verify-custom")
async def verify_custom(req: VerifyCustomRequest) -> dict[str, Any]:
    return verify_trajectory(req.waypoints, scene=req.scene or DEFAULT_SCENE)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>Sim-to-Action 3D VLA Co-Pilot Loading...</h1>")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
