# Sim-to-Action 3D VLA Co-Pilot & Kinematics Verifier

> **Closed-Loop Spatial Reasoning, Kinematics Safety Verification, and Real-Time 3D Digital Twin for Autonomous Manipulators**  
> Powered by **Google Gemini 2.5**, **FastAPI**, **Three.js WebGL**, and **Google Cloud Run**.

---

## 🚀 Overview

Vision-Language-Action (VLA) models and spatial LLMs are transforming robotic autonomy. However, executing raw generative waypoints directly on physical 6-DoF robot arms presents severe safety hazards:
- **Kinematic Singularities & Joint Over-extension**: Generative models lack deterministic knowledge of robotic mechanical limits.
- **3D Workspace Collisions**: Unverified trajectories frequently collide with workcell obstacles (e.g., tools, mugs, fixtures) and the ground plane.
- **Sim-to-Real Reality Gap**: Small spatial hallucination errors translate into catastrophic hardware damage.

**Sim-to-Action 3D VLA Co-Pilot** bridges this critical gap with a **Closed-Loop Verification & Self-Healing Architecture**:
1. **Multimodal VLA Orchestrator (Google Gemini 2.5 Flash / Pro)**: Translates natural language directives and live user-injected prompt constraints into structured 6-DoF robotic manipulation sequences (`approach`, `grasp`, `lift`, `transport`, `release`).
2. **Analytical Kinematics & 3D Collision Verifier**: Evaluates 100% of interpolated waypoints against joint limits ($\theta_i \in [\theta_{\min}, \theta_{\max}]$), table boundary clearance ($Z \ge 0.015\text{ m}$), reach envelope ($[0.12\text{ m}, 0.78\text{ m}]$), and continuous cylinder obstacle collision geometry.
3. **Autonomous Self-Healing Loop**: If verification fails (e.g., collision detected with the obstacle), the exact 3D violation coordinates and vector normals are fed back to Gemini to synthesize a corrected, obstacle-bypassing path.
4. **Interactive 3D WebGL Digital Twin (Three.js)**: Visualizes the 6-DoF articulated robotic manipulator, live gripper status, dense trajectory splines, and step-by-step verification badges in real time.

---

## 🏛️ System Architecture

```
                      +---------------------------------------+
                      |       User Interface / Web Client     |
                      |  - 3D Digital Twin (Three.js)         |
                      |  - Interactive Prompt Injection UI    |
                      |  - 5 Behavioral Preset Selectors      |
                      +-------------------+-------------------+
                                          | HTTP POST /api/plan-and-verify
                                          v
                      +---------------------------------------+
                      |    FastAPI VLA Orchestrator Backend   |
                      +-------------------+-------------------+
                                          |
                                          v
+------------------------+ Prompt with Scene   +-----------------------------+
| Closed-Loop Self-Repair|<--------------------+ Google Gemini 2.5 Flash/Pro |
| Feedback Loop          |                     | Structured Waypoint Plan    |
+-----------^------------+                     +--------------+--------------+
            |                                                 |
            | Violation Diagnostics                           | Raw 6-DoF Waypoints
            | (Collision Points, Joint Bounds)                v
+-----------+-------------------------------------------------+--------------+
|                Analytical 3D Kinematics & Physics Verifier                 |
| - SLSQP Inverse Kinematics                                                 |
| - Hardware Joint Limits Checking (6-DoF: theta_1 .. theta_6)               |
| - Continuous 3D Line-Segment vs. Cylinder Collision Detection              |
| - Workspace Reach Envelope & Table Boundary Clearance Checks               |
+-------------------------------------------------------------+--------------+
                                                              |
                                           Passed Trajectory  | (100% Collision-Free)
                                                              v
                                              +---------------+--------------+
                                              | Real-Time 3D WebGL Execution |
                                              | & ROS 2 JointTrajectory JSON |
                                              +------------------------------+
```

---

## 🎯 Key Features

### 1. Interactive Prompt Steering & Dynamic Constraints
Users can inject custom natural-language spatial constraints directly into the Web UI:
- *"Maintain at least 15cm clearance above all obstacles during transport."*
- *"Execute a curved bypass around the right side of the red mug."*
- *"Perform a smooth descending arc directly to the basket center."*

### 2. 5 Distinct Behavioral Preset Modes
Test and evaluate diverse 6-DoF robotic behaviors with one click:
| Preset Mode | Description | Verification Highlight |
|---|---|---|
| **🔍 1. 360° Scan & Return** | Inspects target cylinder with full rotation, returning to origin | High-yaw rotation & joint limit check |
| **⏪ 2. Reverse Fetch** | Fetches object from goal basket back to source location | Multi-phase approach & inverse trajectory |
| **🐍 3. S-Curve Slalom** | Executes sinusoidal obstacle evasion trajectory | Dense waypoint curvature & clearance |
| **🎯 4. 45cm Air-Drop** | High-altitude release with parachute hover sequence | Elevated reach limit & safety envelope |
| **🚀 5. Standard Overhead Arch** | Classical pick-and-place with obstacle arch bypass | Closed-loop self-repair baseline |

### 3. Industrial Safety & Verification Pipeline
- **Continuous Obstacle Collision Checking**: Mathematical line-segment to cylinder distance testing with customizable safety margins ($r = 0.08\text{ m}$).
- **Table Plane Boundary Interlock**: Strict $Z \ge 0.015\text{ m}$ safety guard prevents end-effector surface crashes.
- **Joint Velocity & Reach Feasibility**: Enforces radial distance bounds ($0.12\text{ m} \le R \le 0.78\text{ m}$) across all kinematic segments.
- **Hardware E-Stop Safety Interlock**: Immediate visual and programmatic execution halt on emergency stop trigger.
- **ROS 2 Trajectory Export**: Ready-to-use `trajectory_msgs/JointTrajectory` JSON format for physical robot hardware and ROS 2 controllers.

---

## 🛠️ Tech Stack

- **AI Model**: Google Gemini 2.5 (`gemini-2.5-flash`, `gemini-2.5-pro` via `google-genai` SDK)
- **Backend**: Python 3.13, FastAPI, Uvicorn, NumPy, SciPy (SLSQP optimization)
- **Frontend**: Three.js (WebGL 3D Digital Twin), Vanilla JS, CSS3 Modern Dark UI
- **Deployment**: Google Cloud Run (Fully managed serverless container), Docker

---

## ⚡ Quick Start

### Option A: Local Development with `uv`

```bash
# 1. Clone repository
git clone https://github.com/your-username/sim-to-action-vla.git
cd sim-to-action-vla

# 2. Install dependencies
uv sync

# 3. Create .env file with your Google Gemini API Key
cp .env.example .env
# Edit .env and set: GEMINI_API_KEY="your-gemini-api-key"

# 4. Start development server
uv run uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```
Visit **http://localhost:8080** in your browser.

---

### Option B: Local Development with `pip`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-gemini-api-key"
python3 -m uvicorn main:app --host 0.0.0.0 --port 8080
```

---

### Option C: Docker Container

```bash
# Build Docker image
docker build -t sim-to-action-vla .

# Run container locally
docker run -p 8080:8080 -e GEMINI_API_KEY="your-gemini-api-key" sim-to-action-vla
```

---

## ☁️ Google Cloud Run Deployment

Automated deployment script is included in [`deploy.sh`](deploy.sh).

### 1. Prerequisites
- Google Cloud SDK (`gcloud`) installed and authenticated (`gcloud auth login`)
- Target GCP project with billing enabled

### 2. One-Command Deploy

```bash
# Configure your GCP Project ID
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export GEMINI_API_KEY="your-gemini-api-key"

# Make deploy script executable and run
chmod +x deploy.sh
./deploy.sh
```

Or deploy directly via `gcloud`:
```bash
gcloud run deploy sim-to-action-vla \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --set-env-vars="GEMINI_API_KEY=${GEMINI_API_KEY}"
```

---

## 📡 REST API Reference

### `GET /api/scene`
Retrieves the default 3D workspace configuration including robot base, obstacle cylinder, target can, and goal basket coordinates.

**Response**:
```json
{
  "status": "ok",
  "scene": {
    "robot_base": [0.0, 0.0, 0.0],
    "can": [0.28, -0.22, 0.06],
    "basket": [0.32, 0.28, 0.08],
    "obstacle": {
      "center": [0.30, 0.03],
      "radius": 0.055,
      "height": 0.24
    }
  }
}
```

### `POST /api/plan-and-verify`
Executes spatial trajectory generation, kinematics checks, and closed-loop self-repair.

**Request Payload**:
```json
{
  "goal": "Pick the blue can and place it safely in the basket while clearing the obstacle.",
  "prompt_constraints": "Maintain at least 10cm vertical clearance over the obstacle.",
  "preset_mode": "standard_arch"
}
```

**Response Payload**:
```json
{
  "status": "success",
  "iterations": [
    {
      "iteration": 1,
      "passed": false,
      "violations": ["Obstacle collision detected at waypoint 4"],
      "waypoints": [...]
    },
    {
      "iteration": 2,
      "passed": true,
      "violations": [],
      "waypoints": [...]
    }
  ],
  "final_trajectory": [...],
  "ros2_joint_trajectory": {
    "joint_names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    "points": [...]
  }
}
```

---

## 🔒 Security & Best Practices

- **Zero Hardcoded Secrets**: Secrets and credentials are managed strictly via environment variables (`GEMINI_API_KEY` / `GOOGLE_API_KEY`).
- **Deterministic Offline Fallback**: In the event of network disruption or missing API keys, the backend gracefully activates mathematical trajectory synthesis to guarantee reliable runtime safety testing.
- **Git Hygiene**: Sensitive local files (`.env`, credentials, local caches) are excluded via [`.gitignore`](.gitignore).

---

## 📄 License

Distributed under the Apache 2.0 License. See `LICENSE` for details.
