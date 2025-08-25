# collision_checker.py
"""
MuJoCo-based state and edge validity checks for a 7‑DoF Panda arm.
This *is* your planner "map": a function that says if a joint config or an
edge (interpolated sequence of configs) is collision-free in the current scene.


Design notes
------------
- Works purely from the loaded `MjModel`/`MjData`: no external meshes needed.
- Uses MuJoCo's contact pipeline. We call `mj_forward` after setting qpos and
then inspect `data.ncon` and each contact's geoms. Visual geoms are already
non-colliding in your model; we don't have to filter them out.
- Lets you whitelist benign contacts (e.g., finger pads with the tomato) at the
very end of motion by a user-supplied predicate.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Optional
import numpy as np
import mujoco


AllowContactFn = Callable[[int, int], bool]


@dataclass
class CollisionChecker:
    model: mujoco.MjModel
    data: mujoco.MjData
    arm_qpos_addr: np.ndarray # indices of the 7 arm joints in qpos
    step_rad: float = 0.02 # edge discretization (rad L2 per segment)
    allow_contact: Optional[AllowContactFn] = None # optional whitelist


    def _set_q(self, q: np.ndarray):
        self.data.qpos[self.arm_qpos_addr] = q
        mujoco.mj_forward(self.model, self.data)


    def _has_blocking_contact(self) -> bool:
        """Return True if any contact is NOT whitelisted by `allow_contact`."""
        ncon = int(self.data.ncon)
        if ncon == 0:
            return False
        if self.allow_contact is None:
            return True
        # check every contact pair
        for i in range(ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if not self.allow_contact(g1, g2):
                return True
        return False


    def is_state_valid(self, q: np.ndarray) -> bool:
        q_backup = self.data.qpos.copy()
        try:
            self._set_q(q)
            return not self._has_blocking_contact()
        finally:
            self.data.qpos[:] = q_backup
            mujoco.mj_forward(self.model, self.data)


    def is_edge_valid(self, qa: np.ndarray, qb: np.ndarray) -> bool:
        """Interpolate qa→qb and ensure every waypoint is collision-free."""
        qa, qb = np.asarray(qa), np.asarray(qb)
        dist = float(np.linalg.norm(qb - qa))
        steps = max(1, int(dist / self.step_rad))
        q_backup = self.data.qpos.copy()
        try:
            for i in range(1, steps + 1):
               q = qa + (i / steps) * (qb - qa)
               self._set_q(q)
               if self._has_blocking_contact():
                    return False
            return True
        finally:
            self.data.qpos[:] = q_backup
            mujoco.mj_forward(self.model, self.data)


# Convenience helpers ----------------------------------------------------
def geom_name(self, gid: int) -> str:
    return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""


def geom_group(self, gid: int) -> int:
    return int(self.model.geom_group[gid])




def default_allow_contact_factory(model: mujoco.MjModel) -> AllowContactFn:
    """Return a predicate whitelisting *only* harmless fingertip contacts.


    You can customize this depending on how your tomato geoms are named/grouped.
    As a conservative default, we do **not** whitelist anything and thus treat
    any contact as blocking. Edit this to, e.g., allow finger pads touching the
    tomato once you're within a small EEF radius of the goal.
    """
    def allow(g1: int, g2: int) -> bool:
        _ = model # currently unused
        return False
    return allow