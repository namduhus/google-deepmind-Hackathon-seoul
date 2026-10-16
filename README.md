# Sim-to-Action 3D VLA Co-Pilot & Kinematics Verifier

> **Track 2**: Gemini 4 Argon in Hard Tech  
> **Stack**: Python 3.13 (`uv`), FastAPI, Google GenAI SDK (`gemini-4-argon` with `thinking_level="high"`), NumPy/SciPy 6-DoF Inverse Kinematics & 3D Collision Verifier, Three.js / WebGPU 3D Digital Twin.

---

## Overview

Vision-Language-Action (VLA) models and LLM trajectory generators frequently produce robot arm trajectories that violate **joint limits**, pass near **kinematic singularities**, or **collide with 3D obstacles** when deployed directly to simulators (MuJoCo / Isaac Sim) or physical hardware.

**Sim-to-Action 3D VLA Co-Pilot** solves this with a closed-loop **Antigravity Verification & Self-Repair Loop**:
1. **3D Spatial Reasoning (`gemini-4-argon`, `thinking_level="high"`)**: Synthesizes 6-DoF manipulator trajectories from natural language instructions and 3D scene specifications.
2. **Antigravity Kinematics & Collision Sandbox ([`kinematics_verifier.py`](kinematics_verifier.py))**:
   - Computes Forward & Inverse Kinematics (SLSQP with top-down orientation & workspace constraints).
   - Verifies hardware joint limits ($\theta_i \in [\theta_{\min}, \theta_{\max}]$) and outer-reach singularities.
   - Performs continuous 3D line-segment vs. cylinder & table collision detection across interpolated motion steps.
3. **Autonomous Self-Repair ([`vla_orchestrator.py`](vla_orchestrator.py))**: Feeds exact collision coordinates and joint limit violations back into Gemini 4 Argon to synthesize an elevated obstacle-bypass trajectory until all checks pass.
4. **Interactive 3D Digital Twin ([`static/index.html`](static/index.html))**: Real-time Three.js 6-DoF robot arm visualization comparing the Iteration 1 failed path against the Iteration 2 self-repaired trajectory, complete with E-Stop safety interlock and ROS 2 JSON export.

---

## Quick Start (Local with `uv` + Python 3.13)

```bash
# 1. Install dependencies (Python 3.13)
uv sync

# 2. Configure API key in .env
# GEMINI_API_KEY=your_api_key_here

# 3. Start the FastAPI + 3D Web Server
uv run uvicorn main:app --host 0.0.0.0 --port 8080
```

Open **http://localhost:8080** in your browser.

---

## Cloud Run Deployment

```bash
gcloud run deploy sim-to-action-vla \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="GEMINI_API_KEY=${GEMINI_API_KEY}"
```
