#!/usr/bin/env python3
"""
Realistic tomato plant generator (v10-style) for MuJoCo.

- Structure mirrors tomato_plant_v10.xml:
  * Pot at (0.4, 0, 0.075)
  * Main plant body "tomato_plant" at (0.4, 0, 0.15)
  * Main stem: capsule from z=0.0 to z≈0.6
  * Side stems "side_stem{i}" at plausible heights with alternating ±x direction
  * Each side stem carries "truss{i}" with a central rachis and 4–6 laterals
    bearing tomatoes (spheres, radius ≈0.02)
  * A few "leaf{i}" bodies at varying heights
- Names use side_stem*, truss*, leaf* (no side_branch*).

Call:
    generate_tomato_plant_xml("/path/to/tomato_plant.xml", seed=123)

This file writes a complete <mujoco> subtree containing just the pot and plant.
It is meant to be <include>d into your scene.
"""

from __future__ import annotations
import math
import random
from typing import List, Tuple, Optional

def _capsule_fromto(x1,y1,z1, x2,y2,z2, size, rgba):
    return f'<geom type="capsule" fromto="{x1:.3f} {y1:.3f} {z1:.3f} {x2:.3f} {y2:.3f} {z2:.3f}" size="{size:.4f}" rgba="{rgba}"/>\n'

def _sphere(pos, r, rgba):
    x,y,z = pos
    return f'<geom type="sphere" pos="{x:.3f} {y:.3f} {z:.3f}" size="{r:.3f}" rgba="{rgba}"/>\n'

def _body(name, pos, inner_xml):
    x,y,z = pos
    return f'<body name="{name}" pos="{x:.3f} {y:.3f} {z:.3f}">\n{inner_xml}</body>\n'

def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def _rand_u(a,b):  # uniform
    return random.uniform(a,b)

def _rng_choice(seq):
    return random.choice(seq)

def _quat_from_yaw_deg(yaw_deg: float):
    import math
    half = math.radians(yaw_deg) * 0.5
    # yaw about +Z: (w, x, y, z)
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _make_truss_xml(dir_sign: int,
                    tomato_count_range=(3,7),
                    distal_span_m: Tuple[float, float]=(0.020, 0.035)) -> str:

    """
    Tomatoes distributed along the *distal* (last) 2–3.5 cm of the side stem,
    measured BACK from the tip (where this truss body is attached).
    This avoids floating: pedicels always start on the side_stem centerline.
    """
    xml = []

    # number of fruits
    n_min, n_max = tomato_count_range
    n_tom = _rng_choice(list(range(n_min, n_max + 1)))

    # how far back from the tip to spread fruit
    span = _rand_u(*distal_span_m)  # meters from tip back toward the base

    # distances from tip along the stem (0 at tip, up to span back)
    if n_tom > 1:
        dists = [i * (span / (n_tom - 1)) for i in range(n_tom)]
    else:
        dists = [0.5 * span]

    for d in dists:
        # base of pedicel lies ON the side_stem centerline:
        # tip frame is at (0,0,0); moving back toward the base is -dir_sign along x
        x_base = -dir_sign * d
        y_base = 0.0
        z_base = 0.0  # centerline at the tip frame (no offset -> no floating)

        # tiny sideways wiggle and downward drop for the pedicel
        y_off = _rand_u(-0.010, 0.010)
        z_drop = _rand_u(0.015, 0.035)     # hang below the stem

        # pedicel: short stub starting ON the stem, ending slightly below & sideways
        x_tip = x_base + _rand_u(-0.003, 0.003)  # tiny axial jitter
        y_tip = y_base + y_off
        z_tip = z_base - z_drop

        xml.append(_capsule_fromto(x_base, y_base, z_base,
                                   x_tip,  y_tip,  z_tip,
                                   0.0012, "0 0.6 0 1"))

        # tomato at the pedicel end (a few mm further down so it really "hangs")
        r = _rand_u(0.018, 0.023)  # 1.8–2.3 cm
        xml.append(_sphere((x_tip, y_tip, z_tip - _rand_u(0.002, 0.004)),
                           r, "1 0 0 1"))

    xml.insert(0, f'<!-- truss: {n_tom} tomatoes, span={span:.3f} m from tip -->\n')
    return "".join(xml)

def _make_side_stem_with_truss(i: int,
                               height_z: float,
                               dir_sign: int,
                               side_length: float = 0.060,
                               side_r: float = 0.0025,
                               stem_drop_range=(0.020, 0.055)) -> str:
    """
    side_stem{i} with a naturally hanging curvature (3 capsule segments).
    A truss is attached EXACTLY at the curved tip (x3, 0, z3), so every side
    stem bears fruit. No straight stems exist in this generator.
    """
    xml = []

    # Total reach and overall droop for this lateral
    x_out = dir_sign * side_length
    z_drop = _rand_u(*stem_drop_range)  # 2–5.5 cm downward sag

    # 3-segment arc: (prox) -> (mid1) -> (mid2) -> (tip)
    x0, z0 = 0.0, 0.0
    x1, z1 = dir_sign * (side_length * 0.40), -z_drop * 0.15
    x2, z2 = dir_sign * (side_length * 0.70), -z_drop * 0.55
    x3, z3 = x_out,                         -z_drop

    # Curved stem (three capsules)
    xml.append(_capsule_fromto(x0, 0, z0,  x1, 0, z1, side_r, "0 0.6 0 1"))
    xml.append(_capsule_fromto(x1, 0, z1,  x2, 0, z2, side_r, "0 0.6 0 1"))
    xml.append(_capsule_fromto(x2, 0, z2,  x3, 0, z3, side_r, "0 0.6 0 1"))

    # Truss sits at the **exact curved tip**
    truss_inner = _make_truss_xml(dir_sign)
    xml.append(f'<!-- side_stem{i} tip=({x3:.3f},{0.0:.3f},{z3:.3f}) -->\n')
    xml.append(_body(f"truss{i}", (x3, 0.0, z3), truss_inner))

    # Wrap side stem body at its world z height
    return _body(f"side_stem{i}", (0.0, 0.0, height_z), "".join(xml))

def random_plant_params(rng: np.random.Generator):
    # Jitter pot/plant around nominal (0.4, 0.0, 0.15)
    x = rng.uniform(0.36, 0.44)
    y = rng.uniform(-0.05, 0.05)
    z = 0.15  # keep same height so pot sits on the table
    base_pos = (float(x), float(y), float(z))

    # Stem height: vary within a reachable band
    stem_height = float(rng.uniform(0.40, 0.52))

    # Sometimes allow unreachable top stems
    reachable_top_only = bool(rng.random() < 0.7)

    return dict(base_pos=base_pos, stem_height=stem_height, reachable_top_only=reachable_top_only)

def generate_tomato_plant_xml(output_file: str,
                              seed: Optional[int] = None,
                              base_pos: Tuple[float, float, float] = (0.4, 0.0, 0.15),
                              stem_height: float = 0.45,   # shorter default
                              reachable_top_only: bool = True):
    """
    Write a realistic tomato plant (pot + plant) to `output_file`.

    Changes:
      - No leaves are generated anywhere.
      - Main stem ends a few cm above the highest truss (no tall bare tip).
      - Every side stem has a truss whose tomatoes hang downward.
    """
    if seed is not None:
        random.seed(int(seed))

    bx, by, bz = base_pos
    xml = []

    # Header
    xml.append('<mujoco model="tomato_plant">\n')
    xml.append('  <worldbody>\n')

    # Pot (fixed, v10-like)
    xml.append(
        f'    <geom name="pot" type="cylinder" pos="{bx:.3f} {by:.3f} {0.075:.3f}" '
        f'size="0.100 0.075" rgba="0.4 0.2 0 1" contype="0" conaffinity="0"/>\n'
    )

    # Assemble plant contents, then wrap into <body name="tomato_plant">
    plant_inner = []

    # ---------------- Side stems + trusses ----------------
    n_side = _rng_choice([4, 5, 6])

    # Internode heights (kept realistic & reachable)
    base_min, base_max = 0.14, min(0.45, stem_height - 0.10)
    raw_heights = sorted([_rand_u(base_min, base_max) for _ in range(n_side)])

    if reachable_top_only and n_side >= 2:
        top_band_lo = max(0.32, base_min + 0.05)
        top_band_hi = min(0.44, stem_height - 0.08)
        if top_band_hi > top_band_lo:
            raw_heights[-1] = _rand_u(top_band_lo + 0.05, min(top_band_hi, stem_height - 0.05))
            raw_heights[-2] = _rand_u(top_band_lo, raw_heights[-1] - 0.04)
        raw_heights = sorted(raw_heights)

    for i, hz in enumerate(raw_heights, start=1):
        dir_sign = +1 if (i % 2 == 1) else -1
        side_len = _clamp(_rand_u(0.055, 0.065), 0.050, 0.075)
        plant_inner.append(_make_side_stem_with_truss(i, hz, dir_sign, side_length=side_len))

    # ---------------- Main vertical stem (short, ends just above top truss) ---
    top_truss_z = raw_heights[-1] if raw_heights else 0.38
    stem_tip_z = max(0.30, float(top_truss_z + 0.03))  # ~3 cm above top truss
    plant_inner.insert(0, _capsule_fromto(0, 0, 0.0, 0, 0, stem_tip_z, 0.0040, "0 0.6 0 1"))

    # Wrap plant body
    plant = _body("tomato_plant", (bx, by, bz), "".join(plant_inner))
    xml.append("    " + plant.replace("\n", "\n    "))

    # Footer
    xml.append("  </worldbody>\n</mujoco>\n")

    with open(output_file, "w") as f:
        f.write("".join(xml))
