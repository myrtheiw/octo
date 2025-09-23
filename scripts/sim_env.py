# ============================ sim_env.py ============================
#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import mujoco
from mujoco import viewer
import dm_env
from dm_env import specs, TimeStep

START_JOINTS = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0], dtype=float)
PRIMARY_CAM_CANDIDATES = ("third_person_cam", "camera", "cam", "third_person", "overhead")
WRIST_CAM_CANDIDATES   = ("gripper_cam", "wrist_cam", "wrist", "hand_cam")
IMG_H, IMG_W = 256, 256

def _id_or_minus_one(model, kind, name: str) -> int:
    try:
        return mujoco.mj_name2id(model, kind, name)
    except Exception:
        return -1

def resolve_camera_name(model, candidates, fallback_id: int = 0) -> str:
    for name in (candidates or ()): 
        cid = _id_or_minus_one(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cid >= 0:
            return name
    N = int(getattr(model, "ncam", 0))
    if N == 0:
        return ""
    cid = int(np.clip(fallback_id, 0, N-1))
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cid) or ""

def auto_ee_ref(model, data):
    sid = _id_or_minus_one(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    if sid >= 0:
        return ("site", "tcp")
    bid = _id_or_minus_one(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    if bid >= 0:
        return ("body", "hand")
    bid = _id_or_minus_one(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    if bid >= 0:
        return ("body", "link7")
    return ("body", "hand")

def find_gripper_actuator(model):
    for aid in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or ""
        if any(k in aname.lower() for k in ("grip", "finger", "hand")):
            return int(aid)
    return -1

def build_arm_mapping_from_model(model, prefer_position=True):
    arm_body_names = [f"link{i}" for i in range(1, 8)]
    arm_joint_ids = []
    for j in range(model.njnt):
        if int(model.jnt_type[j]) != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        b = int(model.jnt_bodyid[j])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if bname in arm_body_names:
            arm_joint_ids.append(j)

    per_joint_candidates = {j: [] for j in arm_joint_ids}
    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid][0])
        if jid not in per_joint_candidates:
            continue
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or ""
        if any(k in aname.lower() for k in ("grip", "finger", "hand")):
            continue
        gaintype = int(model.actuator_gaintype[aid])  # 3=position, 0=general/torque
        qadr     = int(model.jnt_qposadr[jid])
        jname    = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        bname    = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,  int(model.jnt_bodyid[jid])) or ""
        per_joint_candidates[jid].append((aid, jid, qadr, jname, aname, gaintype, bname))

    chosen = []
    for jid in arm_joint_ids:
        cands = per_joint_candidates.get(jid, [])
        if not cands:
            continue
        if prefer_position:
            cands.sort(key=lambda t: (0 if t[5] == 3 else 1, t[2]))
        else:
            cands.sort(key=lambda t: (0 if t[5] != 3 else 1, t[2]))
        chosen.append(cands[0])

    chosen = [c for c in chosen if c[6] in arm_body_names]
    by_jid = {}
    for c in chosen:
        by_jid.setdefault(c[1], c)
    chosen = sorted(by_jid.values(), key=lambda t: t[2])

    if len(chosen) != 7:
        sample = [(c[6], c[3], c[4], c[5], c[2]) for c in chosen]
        raise RuntimeError(
            f"Could not find 7 unique arm actuators (got {len(chosen)}). "
            f"Chosen sample: {sample}"
        )

    arm_act_ids  = np.array([c[0] for c in chosen], dtype=int)
    arm_qpos_adr = np.array([c[2] for c in chosen], dtype=int)
    return arm_act_ids, arm_qpos_adr

def build_arm_dof_indices(model, arm_act_ids):
    pairs = []
    for aid in arm_act_ids:
        jid = int(model.actuator_trnid[aid][0])
        dof = next(d for d in range(model.nv) if int(model.dof_jntid[d]) == jid)
        qadr = int(model.jnt_qposadr[jid])
        pairs.append((qadr, dof))
    pairs.sort(key=lambda t: t[0])
    return np.array([dof for (_, dof) in pairs], dtype=int)

class PandaSimEnv(dm_env.Environment):
    """dm_env around MuJoCo Panda using PD torque control to track joint targets.
    Actions are interpreted as *joint deltas* (7,), applied each outer step.
    """
    def __init__(self, model, data, arm_act_ids, arm_qpos_addr, ee_ref,
                 substeps=40, gripper_idx=-1, arm_dof_idx=None,
                 kp=120.0, kd=None, action_scale=0.05,
                 primary_cam=None, wrist_cam=None):
        self.model = model; self.data = data
        self.arm_act_ids = np.asarray(arm_act_ids, int)
        self.arm_qpos_addr = np.asarray(arm_qpos_addr, int)
        self.arm_dof_idx = np.asarray(arm_dof_idx, int) if arm_dof_idx is not None else None
        if self.arm_dof_idx is None:
            raise ValueError("arm_dof_idx is required")
        self.ee_ref = ee_ref
        self.substeps = int(substeps)
        self.gripper_idx = int(gripper_idx) if gripper_idx is not None else -1
        self.kp = float(kp)
        self.kd = float(2.0 * np.sqrt(self.kp) if kd is None else kd)
        self.action_scale = float(action_scale)
        self._t = 0
        self._capture_images = True

        self.cam_primary = primary_cam or resolve_camera_name(model, PRIMARY_CAM_CANDIDATES, fallback_id=0)
        self.cam_wrist   = wrist_cam   or resolve_camera_name(model, WRIST_CAM_CANDIDATES,   fallback_id=min(1, max(0, int(getattr(model, "ncam", 1)) - 1)))
        self._r_primary = mujoco.Renderer(self.model, height=IMG_H, width=IMG_W)
        self._r_wrist   = mujoco.Renderer(self.model, height=IMG_H, width=IMG_W)

    def set_capture_images(self, enabled: bool):
        self._capture_images = bool(enabled)

    def _render_images(self):
        if not getattr(self, "_capture_images", True):
            z = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
            return z, z
        if self.cam_primary:
            self._r_primary.update_scene(self.data, camera=self.cam_primary)
            img_primary = self._r_primary.render().copy()
        else:
            img_primary = np.zeros((IMG_H, IMG_W, 3), np.uint8)
        if self.cam_wrist:
            self._r_wrist.update_scene(self.data, camera=self.cam_wrist)
            img_wrist = self._r_wrist.render().copy()
        else:
            img_wrist = np.zeros((IMG_H, IMG_W, 3), np.uint8)
        return img_primary, img_wrist

    def _clamp_to_limits(self, q_target):
        q_clamped = q_target.copy()
        for i, dof in enumerate(self.arm_dof_idx):
            jid = int(self.model.dof_jntid[dof])
            if int(self.model.jnt_limited[jid]) == 1:
                lo, hi = self.model.jnt_range[jid]
                if hi > lo + 1e-6:
                    q_clamped[i] = np.clip(q_clamped[i], lo, hi)
        return q_clamped

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qvel[:] = 0.0
        self.data.qpos[self.arm_qpos_addr] = START_JOINTS
        mujoco.mj_forward(self.model, self.data)
        self._t = 0
        img_primary, img_wrist = self._render_images()
        return TimeStep(
            dm_env.StepType.FIRST,
            reward=np.float32(0.0),
            discount=np.float32(1.0),
            observation={
                "proprio": self.data.qpos.astype(np.float32).copy(),
                "image_primary": img_primary,
                "image_wrist": img_wrist,
                "timestep": np.int32(self._t),
            },
        )

    def step(self, action):
        a = np.asarray(action, np.float32)
        if a.ndim == 2 and a.shape[-1] == 7:
            a = a[0]
        if a.shape != (7,):
            raise ValueError(f"Expected action shape (7,), got {a.shape}")
        q  = self.data.qpos[self.arm_qpos_addr].copy()
        q_target = self._clamp_to_limits(q + self.action_scale * a)

        for _ in range(self.substeps):
            q  = self.data.qpos[self.arm_qpos_addr].copy()
            qd = self.data.qvel[self.arm_dof_idx].copy()
            u  = self.kp * (q_target - q) - self.kd * qd
            self.data.ctrl[self.arm_act_ids] = u
            if 0 <= self.gripper_idx < self.model.nu:
                self.data.ctrl[self.gripper_idx] = self.data.ctrl[self.gripper_idx]
            mujoco.mj_step(self.model, self.data)

        self._t += 1
        img_primary, img_wrist = self._render_images()
        return TimeStep(
            dm_env.StepType.MID,
            reward=np.float32(0.0),
            discount=np.float32(1.0),
            observation={
                "proprio": self.data.qpos.astype(np.float32).copy(),
                "image_primary": img_primary,
                "image_wrist": img_wrist,
                "timestep": np.int32(self._t),
            },
        )

    def observation_spec(self):
        return {
            "proprio": specs.Array(shape=(self.model.nq,), dtype=np.float32, name="proprio"),
            "image_primary": specs.Array(shape=(IMG_H, IMG_W, 3), dtype=np.uint8, name="image_primary"),
            "image_wrist":   specs.Array(shape=(IMG_H, IMG_W, 3), dtype=np.uint8, name="image_wrist"),
            "timestep":      specs.Array(shape=(), dtype=np.int32, name="timestep"),
        }

    def action_spec(self):
        return specs.Array(shape=(7,), dtype=np.float32, name="action")
