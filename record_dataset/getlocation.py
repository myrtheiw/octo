#!/usr/bin/env python3
"""
getlocation.py

Utilities to locate side stems and compute grasp points for the tomato plant.

Provided API (used by oracle_dynamic.py / helpers.py):
    - get_side_stem_origins_and_quats(model, data)
    - get_side_stem_grasp_points(model, data, s=0.66)

Both functions are robust to “inverted” stem frames and missing bodies. The
grasp point is computed along the straight line from a stem origin body to its
paired truss tip body, at fraction `s` (0 -> at the stem origin, 1 -> at the tip).
"""

from __future__ import annotations
import os
import numpy as np
import mujoco


# ------------------------- small MuJoCo helpers -------------------------

def _bid(model, name: str, kind) -> int:
    """Return id for a named object; -1 if missing."""
    return mujoco.mj_name2id(model, kind, name)

def _body_pos(model, data, body_name):
    bid = _bid(model, body_name, mujoco.mjtObj.mjOBJ_BODY)
    if bid < 0:
        raise KeyError(f"Body '{body_name}' not found")
    return data.xpos[bid].copy(), bid

def _body_quat(model, data, body_name):
    """Return (w,x,y,z) quaternion for a body from xmat (row-major)."""
    bid = _bid(model, body_name, mujoco.mjtObj.mjOBJ_BODY)
    if bid < 0:
        raise KeyError(f"Body '{body_name}' not found")
    R = data.xmat[bid].reshape(3, 3)
    # robust mat->quat
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=float)
    n = np.linalg.norm(q)
    return q / (n if n > 1e-9 else 1.0)

def _plant_center_xy(model, data, plant_body="tomato_plant"):
    try:
        c, _ = _body_pos(model, data, plant_body)
    except KeyError:
        # fallback: average side_stem positions
        pts = []
        for i in range(1, 7):
            name = f"side_stem{i}"
            bid = _bid(model, name, mujoco.mjtObj.mjOBJ_BODY)
            if bid >= 0:
                pts.append(data.xpos[bid].copy())
        if pts:
            c = np.mean(np.stack(pts, axis=0), axis=0)
        else:
            c = np.array([0.5, 0.0, 0.4], dtype=float)
    c[2] = 0.0
    return c


# ------------------------- public API functions -------------------------

def get_side_stem_origins_and_quats(model, data):
    """
    Returns:
        dict: {stem_name: {"pos": (3,), "quat": (4,), "inverted": bool}}
    Notes:
        - Some stems have their local frames flipped; we detect those by looking
          at the sign of the local Y axis relative to the plant center and mark
          them as inverted=True. This matches the “inverted stems” printout you saw.
    """
    out = {}
    plant_xy = _plant_center_xy(model, data)

    inverted = []
    for i in range(1, 7):
        stem = f"side_stem{i}"
        bid = _bid(model, stem, mujoco.mjtObj.mjOBJ_BODY)
        if bid < 0:
            continue

        pos = data.xpos[bid].copy()
        quat = _body_quat(model, data, stem)

        # Detect “inversion” by checking world +y of the stem versus plant center.
        R = data.xmat[bid].reshape(3, 3)
        y_world = R[:, 1]  # local +y in world
        vec_xy = pos.copy()
        vec_xy[2] = 0.0
        to_plant = plant_xy - vec_xy
        to_plant[2] = 0.0

        # Heuristic: if local +y points toward the plant center, call it inverted
        inv = float(np.dot(y_world[:2], to_plant[:2])) > 0.0
        if inv:
            inverted.append(stem)

        out[stem] = {"pos": pos, "quat": quat, "inverted": bool(inv)}

    if inverted:
        print("Inverted stems:", inverted)
    return out


def get_side_stem_grasp_points(model, data, s: float = 0.66):
    """
    For each available side stem i:
      - origin  = side_stem{i} body position
      - tip     = paired truss{i} body position (if present)
      - grasp   = origin + s * (tip - origin)
      - approach_dir_xy = unit vector from plant center to grasp, with z=0

    Returns:
        dict: {
          stem_name: {
            "grasp_pos": (3,),
            "approach_dir_xy": (3,),    # z=0
            "tip_body": "truss<i>" or None,
          }, ...
        }
    """
    res = {}
    plant_xy = _plant_center_xy(model, data)

    # Pre-resolve body ids to avoid repeated mj_name2id calls
    stem_bids = {}
    truss_bids = {}
    for i in range(1, 7):
        stem_bids[i] = _bid(model, f"side_stem{i}", mujoco.mjtObj.mjOBJ_BODY)
        truss_bids[i] = _bid(model, f"truss{i}", mujoco.mjtObj.mjOBJ_BODY)

    for i in range(1, 7):
        stem = f"side_stem{i}"
        bid_stem = stem_bids[i]
        if bid_stem < 0:
            continue

        p0 = data.xpos[bid_stem].copy()

        bid_tip = truss_bids[i]
        if bid_tip >= 0:
            p1 = data.xpos[bid_tip].copy()
            tip_body = f"truss{i}"
        else:
            # Fallback: small offset along local +x of the stem
            R = data.xmat[bid_stem].reshape(3, 3)
            p1 = p0 + 0.10 * R[:, 0]   # 10 cm along local +x
            tip_body = None

        # Ensure the segment points outward: pick the endpoint farther from plant center as "tip"
        p0_xy = p0.copy(); p0_xy[2] = 0.0
        p1_xy = p1.copy(); p1_xy[2] = 0.0
        if np.linalg.norm(p0_xy - plant_xy) > np.linalg.norm(p1_xy - plant_xy):
            p0, p1 = p1, p0  # swap so that p1 is more “outside” than p0

        grasp = p0 + float(s) * (p1 - p0)

        # Lateral approach direction (XY unit vector from plant center to grasp)
        g_xy = grasp.copy(); g_xy[2] = 0.0
        n = g_xy - plant_xy
        n[2] = 0.0
        nrm = np.linalg.norm(n)
        approach_dir_xy = (n / nrm) if nrm > 1e-9 else np.array([1.0, 0.0, 0.0], dtype=float)

        res[stem] = {
            "grasp_pos": grasp,
            "approach_dir_xy": approach_dir_xy,
            "tip_body": tip_body,
        }

    return res


# ----------------------------- quick self-test -----------------------------

if __name__ == "__main__":
    """
    Minimal runtime check. If you want to test standalone, set MODEL_XML to your
    scene xml path (env var or tweak below).
    """
    MODEL_XML = os.environ.get(
        "MODEL_XML",
        "/home/myrtheiw/octo_ws/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie/franka_emika_panda/scene_dynamic.xml",
    )
    if not os.path.exists(MODEL_XML):
        print("[WARN] MODEL_XML not found; adjust the path above to your scene xml.")
    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    stems = get_side_stem_origins_and_quats(model, data)
    inv_list = [k for k, v in stems.items() if v.get("inverted", False)]
    print("Inverted stems:", inv_list)

    for name in sorted(stems.keys()):
        p = stems[name]["pos"]; q = stems[name]["quat"]
        print(f"{name}: pos=[{p[0]:.2f} {p[1]:.2f} {p[2]:.2f}], "
              f"quat=[{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}]")

    gp = get_side_stem_grasp_points(model, data, s=0.66)
    for name, d in sorted(gp.items()):
        p = d["grasp_pos"]; a = d["approach_dir_xy"]; tip = d["tip_body"]
        print(f"{name}: grasp={p.round(3)}  approach_xy={a.round(3)}  tip_body={tip}")
