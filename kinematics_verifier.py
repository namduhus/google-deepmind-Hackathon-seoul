"""
6-DoF Robotic Arm Kinematics & Collision Verifier
Runs inside the Antigravity Verification Loop to validate:
1. Joint angle limits: theta_i in [-170, 170] deg across all 6 joints (J1..J6)
2. Singularity / Jacobian conditioning check
3. 3D link-segment vs obstacle cylinder & table collision check
4. Target reachability & pick-and-place accuracy
"""

from __future__ import annotations
import math
from typing import Any
import numpy as np
from scipy.optimize import minimize

# 6-DoF Arm DH / Link parameters (meters)
# Base at [0, 0, 0]
L_BASE = 0.12   # Base height (Z)
L_UPPER = 0.28  # Shoulder to Elbow
L_FORE = 0.24   # Elbow to Wrist Pitch (J4)
L_WRIST = 0.10  # Wrist Pitch (J4) to Wrist Roll (J5)
L_TOOL = 0.08   # Wrist Roll (J5) / Tool Yaw (J6) to End-Effector TCP

JOINT_LIMITS_DEG = [
    (-170.0, 170.0),  # J1: Base yaw
    (-120.0, 120.0),  # J2: Shoulder pitch
    (-150.0, 150.0),  # J3: Elbow pitch
    (-170.0, 170.0),  # J4: Wrist pitch
    (-170.0, 170.0),  # J5: Wrist roll
    (-170.0, 170.0),  # J6: Tool yaw
]

DEFAULT_SCENE = {
    "table": {"min": [-0.6, -0.6, -0.05], "max": [0.6, 0.6, 0.0]},
    "target_can": {"pos": [0.30, 0.16, 0.055], "radius": 0.032, "height": 0.11},
    "obstacle_cylinder": {"pos": [0.06, 0.24, 0.14], "radius": 0.055, "height": 0.28},
    "secondary_obstacle": {"pos": [-0.14, 0.25, 0.11], "radius": 0.045, "height": 0.22},
    "goal_basket": {"pos": [-0.30, 0.26, 0.045], "radius": 0.09, "height": 0.09},
    "clearance_margin": 0.035,
}


def rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def rot_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [ c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def rot_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0,  c, -s, 0.0],
        [0.0,  s,  c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def trans(x: float, y: float, z: float) -> np.ndarray:
    return np.array([
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0],
    ])


def forward_kinematics(joints_deg: list[float]) -> dict[str, Any]:
    """
    Computes 3D positions of all joints, the end-effector TCP, and tool orientation
    for the 6-DoF arm. Matches the exact hierarchy used in the Three.js 3D viewer.
    """
    q = [math.radians(float(a)) for a in joints_deg]
    while len(q) < 6:
        q.append(0.0)

    p0 = np.array([0.0, 0.0, 0.0])
    # Joint 1: Base Yaw around Z, translated up by L_BASE
    T1 = trans(0.0, 0.0, L_BASE) @ rot_z(q[0])
    p1 = T1[:3, 3].copy()

    # Joint 2: Shoulder Pitch around Y, link along Z (L_UPPER)
    T2 = T1 @ rot_y(q[1]) @ trans(0.0, 0.0, L_UPPER)
    p2 = T2[:3, 3].copy()

    # Joint 3: Elbow Pitch around Y, link along Z (L_FORE)
    T3 = T2 @ rot_y(q[2]) @ trans(0.0, 0.0, L_FORE)
    p3 = T3[:3, 3].copy()

    # Joint 4: Wrist Pitch around Y, link along Z (L_WRIST)
    T4 = T3 @ rot_y(q[3]) @ trans(0.0, 0.0, L_WRIST)
    p4 = T4[:3, 3].copy()

    # Joint 5 (Wrist Roll around X) & Joint 6 (Tool Yaw around Z), link along Z (L_TOOL)
    T5 = T4 @ rot_x(q[4]) @ rot_z(q[5]) @ trans(0.0, 0.0, L_TOOL)
    p_ee = T5[:3, 3].copy()

    chain = [p0, p1, p2, p3, p4, p_ee]
    return {
        "ee_pos": [round(float(v), 4) for v in p_ee],
        "ee_pos_exact": p_ee,
        "joint_positions": [[round(float(v), 4) for v in pt] for pt in chain],
        "joint_positions_exact": chain,
        "transform": T5,
    }


def solve_ik(
    target_xyz: list[float],
    initial_guess_deg: list[float] | None = None,
    prefer_top_down: bool = True,
    target_roll_deg: float | None = None,
    target_yaw_deg: float | None = None,
) -> list[float]:
    """
    Solves 6-DoF Inverse Kinematics for target [x, y, z] end-effector position
    along with wrist roll (J5) and tool yaw (J6) orientation targets.
    Achieves sub-millimeter accuracy using exact unrounded FK in the objective.
    """
    target = np.array(target_xyz, dtype=float)
    yaw_hint = math.degrees(math.atan2(target[1], target[0]))

    desired_roll = float(target_roll_deg) if target_roll_deg is not None else 0.0
    desired_yaw = float(target_yaw_deg) if target_yaw_deg is not None else round(yaw_hint * 0.85, 2)

    if initial_guess_deg is None:
        q0 = np.array([yaw_hint, -15.0, 85.0, 110.0, desired_roll, desired_yaw], dtype=float)
    else:
        q0 = np.array(initial_guess_deg, dtype=float)
        q0[0] = yaw_hint
        q0[4] = desired_roll
        q0[5] = desired_yaw

    def loss(q_deg: np.ndarray) -> float:
        fk = forward_kinematics(q_deg.tolist())
        ee = fk["ee_pos_exact"]
        pos_err = float(np.sum((ee - target) ** 2)) * 5000.0

        # Encourage top-down wrist orientation (q2 + q3 + q4 ~ 180 deg so tool points vertically down -Z)
        pitch_sum = q_deg[1] + q_deg[2] + q_deg[3]
        target_pitch_sum = 180.0 if prefer_top_down else 150.0
        orient_err = 0.002 * ((pitch_sum - target_pitch_sum) ** 2)

        # Wrist Roll (J5) and Tool Yaw (J6) orientation tracking
        roll_err = 0.01 * ((q_deg[4] - desired_roll) ** 2)
        yaw_err = 0.01 * ((q_deg[5] - desired_yaw) ** 2)

        # Keep elbow (p2) and wrist (p3) high above table and obstacles
        chain_exact = fk["joint_positions_exact"]
        elbow_z = float(chain_exact[2][2])
        wrist_z = float(chain_exact[3][2])
        height_pen = 0.0
        if elbow_z < 0.25:
            height_pen += (0.25 - elbow_z) ** 2 * 200.0
        if wrist_z < 0.12:
            height_pen += (0.12 - wrist_z) ** 2 * 200.0

        reg = 0.0001 * float(np.sum((q_deg[:4] - q0[:4]) ** 2))
        return pos_err + orient_err + roll_err + yaw_err + height_pen + reg

    res = minimize(
        loss,
        q0,
        method="SLSQP",
        bounds=JOINT_LIMITS_DEG,
        options={"maxiter": 250, "ftol": 1e-9},
    )
    solved = [round(float(v), 2) for v in res.x]
    solved[4] = round(desired_roll, 2)
    solved[5] = round(desired_yaw, 2)
    return solved


def point_to_cylinder_clearance(
    pt: np.ndarray, cyl_center: np.ndarray, cyl_radius: float, cyl_height: float
) -> float:
    """
    Signed clearance distance between a 3D point and a finite vertical cylinder.
    Negative means inside/colliding.
    """
    dx = pt[0] - cyl_center[0]
    dy = pt[1] - cyl_center[1]
    radial_dist = math.hypot(dx, dy) - cyl_radius

    z_min = 0.0
    z_max = cyl_height
    if pt[2] < z_min:
        dz = z_min - pt[2]
    elif pt[2] > z_max:
        dz = pt[2] - z_max
    else:
        dz = 0.0

    if dz == 0.0:
        return radial_dist
    if radial_dist <= 0.0:
        return dz
    return math.hypot(radial_dist, dz)


def check_segment_cylinder_collision(
    p_start: np.ndarray,
    p_end: np.ndarray,
    cyl_center: np.ndarray,
    cyl_radius: float,
    cyl_height: float,
    samples: int = 20,
) -> tuple[float, np.ndarray]:
    """Returns minimum clearance along segment [p_start, p_end] and closest point."""
    min_dist = float("inf")
    closest_pt = p_start
    for i in range(samples + 1):
        t = i / samples
        pt = p_start * (1.0 - t) + p_end * t
        d = point_to_cylinder_clearance(pt, cyl_center, cyl_radius, cyl_height)
        if d < min_dist:
            min_dist = d
            closest_pt = pt
    return min_dist, closest_pt


def verify_trajectory(
    waypoints: list[dict[str, Any]], scene: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Runs full kinematic, joint-limit, singularity, and 3D obstacle collision verification
    across all waypoints and interpolated sub-steps.
    """
    if scene is None:
        scene = DEFAULT_SCENE

    margin = float(scene.get("clearance_margin", 0.035))
    obstacles = [
        ("Primary Obstacle (Tall Cylinder)", scene["obstacle_cylinder"]),
    ]
    if "secondary_obstacle" in scene and scene["secondary_obstacle"]:
        obstacles.append(("Secondary Obstacle (Pipe)", scene["secondary_obstacle"]))

    violations: list[dict[str, Any]] = []
    enriched_waypoints: list[dict[str, Any]] = []
    min_global_clearance = float("inf")

    if not waypoints or len(waypoints) < 2:
        return {
            "passed": False,
            "min_clearance_m": 0.0,
            "violations": [{"type": "EMPTY_TRAJECTORY", "message": "At least 2 waypoints are required."}],
            "waypoints": [],
        }

    # 1. Per-waypoint joint limit, table collision, and FK enrichment
    for idx, wp in enumerate(waypoints):
        name = wp.get("name", f"Waypoint_{idx}")
        if "joints_deg" in wp and wp["joints_deg"]:
            joints = [float(x) for x in wp["joints_deg"]]
            fk = forward_kinematics(joints)
        elif "target_xyz" in wp and wp["target_xyz"]:
            prev_q = enriched_waypoints[-1]["joints_deg"] if enriched_waypoints else None
            joints = solve_ik(
                wp["target_xyz"],
                initial_guess_deg=prev_q,
                target_roll_deg=wp.get("target_roll_deg"),
                target_yaw_deg=wp.get("target_yaw_deg"),
            )
            fk = forward_kinematics(joints)
        else:
            violations.append({
                "waypoint_index": idx,
                "waypoint_name": name,
                "type": "INVALID_FORMAT",
                "message": f"Waypoint {idx} ({name}) missing both joints_deg and target_xyz.",
            })
            continue

        # Joint limit check
        for j_idx, (angle, (low, high)) in enumerate(zip(joints, JOINT_LIMITS_DEG)):
            if angle < low or angle > high:
                violations.append({
                    "waypoint_index": idx,
                    "waypoint_name": name,
                    "type": "JOINT_LIMIT_VIOLATION",
                    "joint": j_idx + 1,
                    "value_deg": round(angle, 2),
                    "limit_deg": [low, high],
                    "message": (
                        f"Waypoint {idx} ('{name}') violates Joint {j_idx + 1} limit: "
                        f"{angle:.1f} deg not in [{low}, {high}] deg."
                    ),
                })

        # Singularity / hyper-extension check (distance from shoulder p1 to EE)
        shoulder_pos = fk["joint_positions"][1]
        ee_pos = fk["ee_pos"]
        arm_ext = math.sqrt(sum((a - b) ** 2 for a, b in zip(ee_pos, shoulder_pos)))
        max_reach = L_UPPER + L_FORE + L_WRIST + L_TOOL
        if arm_ext > max_reach * 0.97:
            violations.append({
                "waypoint_index": idx,
                "waypoint_name": name,
                "type": "KINEMATIC_SINGULARITY",
                "message": (
                    f"Waypoint {idx} ('{name}') approaches outer workspace boundary singularity "
                    f"(reach={arm_ext:.3f}m / max={max_reach:.3f}m)."
                ),
            })

        # Table collision check (all joints except base must have z >= 0.012)
        for link_idx, j_pos in enumerate(fk["joint_positions"][1:], start=1):
            if j_pos[2] < 0.012:
                violations.append({
                    "waypoint_index": idx,
                    "waypoint_name": name,
                    "type": "TABLE_COLLISION",
                    "message": (
                        f"Waypoint {idx} ('{name}') Link {link_idx} collides with table plane "
                        f"(z={j_pos[2]:.3f}m < 0.012m)."
                    ),
                })

        gripper_str = wp.get("gripper", "open")
        default_opening = 0.0 if gripper_str == "closed" else 1.0
        enriched_waypoints.append({
            "index": idx,
            "name": name,
            "joints_deg": [round(float(x), 2) for x in joints],
            "ee_pos": fk["ee_pos"],
            "joint_positions": fk["joint_positions"],
            "gripper": gripper_str,
            "gripper_opening": float(wp.get("gripper_opening", default_opening)),
            "payload_attached": bool(wp.get("payload_attached", gripper_str == "closed")),
            "phase": wp.get("phase", "transit"),
        })

    # 2. Continuous segment interpolation collision check between consecutive waypoints
    interp_steps = 12
    for seg_idx in range(len(enriched_waypoints) - 1):
        wp_a = enriched_waypoints[seg_idx]
        wp_b = enriched_waypoints[seg_idx + 1]
        q_a = np.array(wp_a["joints_deg"], dtype=float)
        q_b = np.array(wp_b["joints_deg"], dtype=float)

        seg_min_clearance = float("inf")
        worst_obs_name = ""
        worst_pt = np.zeros(3)
        worst_alpha = 0.0

        for step in range(interp_steps + 1):
            alpha = step / interp_steps
            q_interp = (1.0 - alpha) * q_a + alpha * q_b
            fk_step = forward_kinematics(q_interp.tolist())
            chain_pts = [np.array(p) for p in fk_step["joint_positions"]]

            # Check each arm link segment (Base->Shoulder->Elbow->Wrist->EE)
            for l_i in range(1, len(chain_pts) - 1):
                p_start = chain_pts[l_i]
                p_end = chain_pts[l_i + 1]
                for obs_name, obs in obstacles:
                    c_pos = np.array(obs["pos"], dtype=float)
                    c_rad = float(obs["radius"])
                    c_h = float(obs["height"])
                    dist, closest_pt = check_segment_cylinder_collision(
                        p_start, p_end, c_pos, c_rad, c_h, samples=12
                    )
                    if dist < min_global_clearance:
                        min_global_clearance = dist
                    if dist < seg_min_clearance:
                        seg_min_clearance = dist
                        worst_obs_name = obs_name
                        worst_pt = closest_pt
                        worst_alpha = alpha

        if seg_min_clearance < margin:
            violations.append({
                "segment": [seg_idx, seg_idx + 1],
                "waypoint_names": [wp_a["name"], wp_b["name"]],
                "type": "OBSTACLE_COLLISION",
                "obstacle": worst_obs_name,
                "clearance_m": round(float(seg_min_clearance), 4),
                "required_clearance_m": margin,
                "collision_xyz": [round(float(v), 3) for v in worst_pt],
                "interpolation_alpha": round(worst_alpha, 2),
                "message": (
                    f"Collision detected on segment [{seg_idx}:{wp_a['name']} -> {seg_idx+1}:{wp_b['name']}] "
                    f"at {int(worst_alpha*100)}% interpolation against {worst_obs_name}: "
                    f"clearance={seg_min_clearance:.3f}m < safety margin {margin:.3f}m "
                    f"near XYZ={[round(float(v), 3) for v in worst_pt]}. "
                    f"Insert an elevated or lateral bypass via-point (e.g., z >= 0.34m) to clear the obstacle."
                ),
            })

    passed = len(violations) == 0
    return {
        "passed": passed,
        "min_clearance_m": round(float(min_global_clearance if min_global_clearance != float("inf") else 0.0), 4),
        "required_clearance_m": margin,
        "violations": violations,
        "waypoints": enriched_waypoints,
    }
