# Track 2: Gemini 4 Argon in Hard Tech — Project Proposal

## Project Title: **Sim-to-Action 3D VLA Co-Pilot & Kinematics Verifier**
> **Target Persona**: Robotics / Vision-Language-Action (VLA) / Physical AI Researchers & Engineers  
> **Core Value**: Synthesizes 6-DoF robotic manipulator trajectories from natural-language prompts and 3D scene specifications, executes **autonomous inverse kinematics & 3D collision verification in an isolated sandbox**, and visualizes the self-repaired motion in a real-time **Three.js 3D Digital Twin**.

---

## 1. Problem Statement & Solution

### Pain Points
* **High Cost of Trial-and-Error in VLA & Motion Planning**: Standard Vision-Language Models frequently output robot trajectories that violate hardware joint limits, enter kinematic singularities, or collide with 3D obstacles when executed in physics simulators (MuJoCo, Isaac Sim) or on physical hardware.
* **Lack of Closed-Loop Autonomous Verification**: Existing LLM coding assistants generate one-shot trajectories without running deterministic Forward/Inverse Kinematics (FK/IK) and continuous 3D collision verification in a closed feedback loop.

### Solution
* **Gemini 4 Argon (`gemini-4-argon`, `thinking_level="high"`)**: Performs deep 3D spatial reasoning and diagnoses exact mathematical failure logs to self-repair failed trajectories.
* **Deterministic 6-DoF Kinematics & Collision Sandbox ([`kinematics_verifier.py`](kinematics_verifier.py))**: Executes SLSQP 6-DoF Inverse Kinematics, hardware joint-limit checks, reach-singularity checks, and continuous 3D line-segment vs. cylinder collision verification.
* **Interactive 3D Digital Twin ([`static/index.html`](static/index.html))**: Renders the 6-DoF robot arm in Three.js with real-time Prompt & 6-DoF Studio controls, 5 distinct behavioral modes, E-Stop safety interlock, and one-click ROS 2 `JointTrajectory` JSON export.

---

## 2. System Architecture

```text
[User Input (Web UI)]
  - Natural-Language Prompt (5 Distinct Modes + Custom Altitude/Arc/Wrist Angles)
  - 3D Scene Metadata (Table bounds, Target Can, Central Cylinder Obstacles, Goal Basket)
        │
        ▼
[Orchestrator: Gemini 4 Argon (thinking_level="high")]
  - Parses behavioral mode & 6-DoF kinematic directives
  - Synthesizes candidate 6-DoF trajectory waypoints
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ Antigravity Kinematics & Collision Sandbox                      │
│ 1. Run Python Kinematics/Collision Verifier                     │
│    - Joint Limit Check: theta_i in [theta_min, theta_max]       │
│    - Singularity Check: Max radial reach envelope               │
│    - 3D Collision Check: Link segments vs. Obstacle Cylinders   │
│ 2. Evaluate Pass vs. Fail                                       │
└───────┬─────────────────────────────────────────────────────────┘
        │
        ├─── [FAIL] Exact Violation Logs (e.g., "J4 limit exceeded", "Link 3 collision")
        │     └──> Fed back into Gemini 4 Argon for Autonomous Self-Repair (Iteration 2)
        │
        └─── [PASS] Verified Trajectory JSON & 6-DoF Telemetry returned to Web UI
              │
              ▼
[Frontend: Interactive 3D Digital Twin (Three.js + TailwindCSS)]
  - Real-time 6-DoF robot arm animation (J1–J6 + Parallel Gripper Physics)
  - Side-by-side comparison of Iteration 1 Failed Path vs. Iteration 2 Verified Path
  - Live Joint Telemetry Bars & ROS 2 JointTrajectory Export
```