from __future__ import annotations
import math
from dataclasses import dataclass 
import numpy as np 
from typing import List, Optional, Callable

from collision_checker import CollisionChecker 

@dataclass
class RRTStarConfig:
    step_rad: float = 0.25 # steering step in joint L2 (rad)
    neighborhood_radius: float = 0.6 # rewiring/search radius
    goal_sample_prob: float = 0.1 # goal bias
    max_iterations: int = 5000 
    goal_tol_rad: float = 0.05 # in joint space L2

class RRTStar:
    def __init__(self, limits_lo: np.ndarray, limits_hi: np.ndarray,
                 checker: CollisionChecker, cfg: RRTStarConfig = RRTStarConfig()):
        self.lo = np.asarray(limits_lo)
        self.hi = np.asarray(limits_hi)
        self.dim = int(self.lo.size)
        self.checker = checker
        self.cfg = cfg
        # tree state
        self.nodes: List[np.ndarray] = []
        self.parent: List[int] = []
        self.cost: List[float] = []

    # -------------------- Core ops --------------------
    def sample(self, q_goal: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.cfg.goal_sample_prob:
            return q_goal.copy()
        return self.lo + np.random.rand(self.dim) * (self.hi - self.lo)


    def nearest(self, q: np.ndarray) -> int:
        dists = [np.linalg.norm(q - n) for n in self.nodes]
        return int(np.argmin(dists))


    def steer(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        v = q_to - q_from
        d = float(np.linalg.norm(v))
        if d <= self.cfg.step_rad:
            q_new = q_to
        else:
            q_new = q_from + (self.cfg.step_rad / d) * v
        return np.clip(q_new, self.lo, self.hi)


    def neighbors(self, q: np.ndarray) -> List[int]:
        idxs = []
        for i, n in enumerate(self.nodes):
            if np.linalg.norm(q - n) <= self.cfg.neighbor_radius:
                idxs.append(i)
        return idxs


# -------------------- Planning --------------------
    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> Optional[List[np.ndarray]]:
        if not self.checker.is_state_valid(q_start):
            print("Start is in collision.")
            return None
        if not self.checker.is_state_valid(q_goal):
            print("Goal is in collision.")
    
    # We still try to reach a nearby goal; but warn the caller.


        self.nodes = [q_start.copy()]
        self.parent = [-1]
        self.cost = [0.0]


        best_goal_idx: Optional[int] = None


        for it in range(self.cfg.max_iters):
            q_rand = self.sample(q_goal)
            j = self.nearest(q_rand)
            q_new = self.steer(self.nodes[j], q_rand)
            if not self.checker.is_edge_valid(self.nodes[j], q_new):
                continue


            # Choose parent that minimizes cost + edge cost among neighbors
            neigh = self.neighbors(q_new)
            q_parent = j
            c_parent = self.cost[j] + np.linalg.norm(q_new - self.nodes[j])
            for i in neigh:
                if self.checker.is_edge_valid(self.nodes[i], q_new):
                    c = self.cost[i] + np.linalg.norm(q_new - self.nodes[i])
                    if c < c_parent:
                        q_parent, c_parent = i, c


            # Insert node
            self.nodes.append(q_new)
            self.parent.append(q_parent)
            self.cost.append(c_parent)
            new_idx = len(self.nodes) - 1


            # Rewire neighbors
            for i in neigh:
                new_cost = self.cost[new_idx] + np.linalg.norm(self.nodes[i] - q_new)
                if new_cost + 1e-6 < self.cost[i] and self.checker.is_edge_valid(q_new, self.nodes[i]):
                    self.parent[i] = new_idx
                    self.cost[i] = new_cost


            # Goal check (in joint space)
            if np.linalg.norm(q_new - q_goal) <= self.cfg.goal_tol_rad:
                best_goal_idx = new_idx
                break


        if best_goal_idx is None:
            # pick closest-to-goal node if no exact hit
            dists = [np.linalg.norm(n - q_goal) for n in self.nodes]
            best_goal_idx = int(np.argmin(dists))


        # Reconstruct path
        path: List[np.ndarray] = []
        i = best_goal_idx
        while i >= 0:
            path.append(self.nodes[i])
            i = self.parent[i]
        path.reverse()
        return path


# -------------------- Utilities --------------------


def shortcut_smooth(path: List[np.ndarray], checker: CollisionChecker, attempts: int = 200) -> List[np.ndarray]:
    if len(path) <= 2:
        return path
    path = [p.copy() for p in path]
    for _ in range(attempts):
        if len(path) <= 2:
            break
        i = np.random.randint(0, len(path) - 2)
        j = np.random.randint(i + 2, len(path))
        if checker.is_edge_valid(path[i], path[j]):
            path = path[: i + 1] + path[j:]
    return path