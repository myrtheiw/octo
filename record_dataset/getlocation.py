import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_side_stem_midpoints_and_quats(model, data, prefix="side_stem"):
    """Return dict: body_name -> (world midpoint (3,), world orientation quaternion (4,))"""
    mujoco.mj_forward(model, data)  # make sure transforms are up to date

    # Build: body_id -> body_name
    body_names = {bid: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
                  for bid in range(model.nbody)}

    # Collect geoms per body
    geoms_by_body = {}
    for gid in range(model.ngeom):
        bid = model.geom_bodyid[gid]
        geoms_by_body.setdefault(bid, []).append(gid)

    results = {}
    for bid, bname in body_names.items():
        if not bname or not bname.startswith(prefix):
            continue

        # Get all capsule geoms for this body
        capsule_gids = [gid for gid in geoms_by_body.get(bid, [])
                        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CAPSULE]
        if not capsule_gids:
            continue

        # Pick the longest capsule (by half-length in geom_size[1])
        def half_len(gid):
            return model.geom_size[gid][1] if model.geom_size.shape[1] > 1 else 0.0
        gid_best = max(capsule_gids, key=half_len)

        # World position of capsule midpoint
        pos_world = data.geom_xpos[gid_best].copy()

        # World orientation as rotation matrix
        rot_mat = data.geom_xmat[gid_best].reshape(3, 3)
        quat_wxyz = R.from_matrix(rot_mat).as_quat()  # returns (x, y, z, w) by default
        # Convert to (w, x, y, z)
        quat_wxyz = np.roll(quat_wxyz, 1)

        results[bname] = (pos_world, quat_wxyz)

    return results

# ---- Example usage ----
model = mujoco.MjModel.from_xml_path("/home/myrtheiw/octo_ws/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie/franka_emika_panda/scene.xml"
)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

stems = get_side_stem_midpoints_and_quats(model, data)
for name, (pos, quat) in sorted(stems.items()):
    print(f"{name}: pos={pos}, quat={quat}")
