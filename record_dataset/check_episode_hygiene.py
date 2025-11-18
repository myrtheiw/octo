import argparse, math, numpy as np, tensorflow as tf

BIN_TOL = 0.05          # tolerance for gripper ~binary check
RAD_Q_ABS_WARN = 4*math.pi   # q beyond this looks non-radian
RAD_DQ_STEP_WARN = 0.2       # per-step delta beyond this is suspicious

def _get_float_arr(f, key):
    return np.array(f[key].float_list.value, dtype=np.float32) if key in f else None

def _get_int_arr(f, key):
    return np.array(f[key].int64_list.value, dtype=np.int64) if key in f else None

def _get_bytes(f, key):
    if key in f and f[key].bytes_list.value:
        return f[key].bytes_list.value[0].decode("utf-8")
    return None

def _get_float(f, key, default=None):
    if key in f and f[key].float_list.value:
        return float(f[key].float_list.value[0])
    return default

def pct(x, p):
    return float(np.percentile(x, p)) if x.size else float('nan')

def looks_binary(x, tol=BIN_TOL):
    if x.size == 0: return False
    xr = np.round(x)
    return np.all(np.abs(x - xr) <= tol) and set(np.unique(xr).tolist()).issubset({0.0, 1.0})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--dof", type=int, default=7)
    ap.add_argument("--clamp_rad", type=float, default=0.02)
    ap.add_argument("--show_bad", action="store_true")
    # New semantics & scale expectations (can be None to skip strictness)
    ap.add_argument("--expect_action_type", type=str, default="joint_delta_executed")
    ap.add_argument("--expect_action_scale", type=float, default=1.0)
    args = ap.parse_args()

    ds = tf.data.TFRecordDataset(args.shard)

    n_total = 0
    n_short = 0
    n_missing_first = 0
    n_missing_terminal = 0
    n_nan = 0
    n_inf = 0

    # Aggregate tails
    all_action_mag = []
    all_execdq_mag = []

    # Semantics & scale tallies
    n_bad_type = 0
    n_bad_scale = 0
    n_bad_dof = 0
    n_bad_gripper = 0
    n_units_q = 0
    n_units_dq = 0

    bad_eps = []

    for idx, raw in enumerate(ds):
        n_total += 1
        ex = tf.train.Example.FromString(raw.numpy())
        f = ex.features.feature

        # Metadata (per-example)
        meta_type  = _get_bytes(f, "metadata/action_type")
        meta_scale = _get_float(f, "metadata/action_scale", None)
        meta_dof   = _get_float(f, "metadata/action_dof", None)
        meta_inc_g = _get_float(f, "metadata/include_gripper", None)  # 0.0/1.0 sometimes

        prop = _get_float_arr(f, "steps/observation/proprio")
        act  = _get_float_arr(f, "steps/action")
        isf  = _get_int_arr(f, "steps/is_first")
        ist  = _get_int_arr(f, "steps/is_terminal") if "steps/is_terminal" in f else None

        # Basic presence
        if prop is None or isf is None or isf.size == 0:
            n_short += 1
            bad_eps.append((idx, "missing_required_fields"))
            continue

        T = int(isf.size)
        Dp = prop.size // T
        proprio = prop.reshape(T, Dp)

        Da = 0
        action = None
        if act is not None and act.size:
            Da = act.size // T
            action = act.reshape(T, Da)

        is_first = isf.astype(bool)
        has_first = bool(is_first[0])
        if not has_first:
            n_missing_first += 1

        if ist is not None and ist.size == T:
            is_term = ist.astype(bool)
            if not bool(is_term[-1]):
                n_missing_terminal += 1

        # Short/empty episode?
        if T < 2:
            n_short += 1
            bad_eps.append((idx, "too_short"))
            continue

        # NaN/Inf checks
        flags = []
        def _check_bad(name, arr):
            nonlocal n_nan, n_inf
            if arr is None:
                return False
            bad_nan = np.isnan(arr).any()
            bad_inf = np.isinf(arr).any()
            if bad_nan: n_nan += 1
            if bad_inf: n_inf += 1
            if bad_nan or bad_inf:
                flags.append(f"{name}:nan={bad_nan}|inf={bad_inf}")
                return True
            return False

        _check_bad("proprio", proprio)
        if action is not None:
            _check_bad("action", action)

        # ===== Semantics & Scale checks =====
        # 1) action_type
        exp_type = args.expect_action_type
        if exp_type is not None:
            if meta_type != exp_type:
                n_bad_type += 1
                flags.append(f"bad_action_type:{meta_type}")

        # 2) action_scale
        exp_scale = args.expect_action_scale
        if exp_scale is not None:
            if meta_scale is None or not np.isfinite(meta_scale) or abs(meta_scale - exp_scale) > 1e-6:
                n_bad_scale += 1
                flags.append(f"bad_action_scale:{meta_scale}")

        # 3) DOF (+gripper) and gripper binary check
        # Determine expected length
        dof = int(args.dof)
        include_gripper = None
        if meta_inc_g is not None:
            include_gripper = (float(meta_inc_g) >= 0.5)

        # If metadata missing, infer by dimension and binary last column
        if include_gripper is None and action is not None and Da in (dof, dof+1):
            if Da == dof+1 and looks_binary(action[:, -1]):
                include_gripper = True
            elif Da == dof:
                include_gripper = False

        if action is not None and Da > 0:
            exp_len = dof + (1 if include_gripper else 0 if include_gripper is not None else 0)
            # If include_gripper unknown and Da not in {dof,dof+1}, flag
            if include_gripper is None and Da not in (dof, dof+1):
                n_bad_dof += 1
                flags.append(f"bad_action_dim:{Da}")
            else:
                # If include_gripper decided, enforce exact match
                if include_gripper is not None and Da != exp_len:
                    n_bad_dof += 1
                    flags.append(f"bad_action_dim:{Da}_expected:{exp_len}")
                # If we have a gripper column, check it's ~binary
                if include_gripper and Da >= dof+1:
                    g = action[:, -1]
                    if not looks_binary(g):
                        n_bad_gripper += 1
                        flags.append("gripper_not_binary")

        # 4) Units heuristics
        #   q in radians: warn if any |q| > 4π
        if np.any(np.abs(proprio[:, :dof]) > RAD_Q_ABS_WARN):
            n_units_q += 1
            flags.append("q_exceeds_4pi")
        #   executed Δq not crazy: |Δq| > 0.2 (per-step) suspicious
        dq = proprio[1:, :dof] - proprio[:-1, :dof]
        mask = ~is_first[:-1]
        dq = dq[mask]
        if dq.size and np.any(np.abs(dq) > RAD_DQ_STEP_WARN):
            n_units_dq += 1
            flags.append("dq_step_gt_0.2rad")

        # ===== Original hygiene magnitude summaries =====
        if dq.size:
            all_execdq_mag.append(np.linalg.norm(dq, axis=1))
        if action is not None and Da > 0:
            a = action[:-1]
            a = a[mask] if a.shape[0] == mask.shape[0] else a
            if a.size:
                all_action_mag.append(np.linalg.norm(a[:, :min(dof, a.shape[1])], axis=1))

        if (not has_first) or flags:
            bad_eps.append((idx, "|".join(flags) if flags else "flags"))

    # Aggregate reporting
    all_action_mag = np.concatenate(all_action_mag) if len(all_action_mag) else np.array([])
    all_execdq_mag = np.concatenate(all_execdq_mag) if len(all_execdq_mag) else np.array([])

    def summarise(name, arr):
        if arr.size == 0:
            return f"{name}: n=0"
        return (f"{name}: n={arr.size}, "
                f"p50={pct(arr,50):.5f}, p90={pct(arr,90):.5f}, "
                f"p99={pct(arr,99):.5f}, p99.9={pct(arr,99.9):.5f}, max={np.max(arr):.5f}")

    print("=== EPISODE HYGIENE + SEMANTICS SUMMARY ===")
    print(f"file: {args.shard}")
    print(f"episodes: {n_total}  short(<2): {n_short}  missing_first: {n_missing_first}  "
          f"missing_terminal(if_present): {n_missing_terminal}")
    print(f"NaN episodes (any): {n_nan}  Inf episodes (any): {n_inf}")
    print(summarise("|action| per-step", all_action_mag))
    print(summarise("|exec Δq| per-step", all_execdq_mag))
    print(f"clamp_rad: {args.clamp_rad}  dof: {args.dof}")
    print("\n-- Semantics & Scale checks --")
    print(f"bad_action_type: {n_bad_type}  (expected: {args.expect_action_type})")
    print(f"bad_action_scale: {n_bad_scale}  (expected: {args.expect_action_scale})")
    print(f"bad_action_dim: {n_bad_dof}  bad_gripper_binary: {n_bad_gripper}")
    print(f"units_warn_q>|4π|: {n_units_q}  units_warn_Δq>0.2: {n_units_dq}")

    if bad_eps and args.show_bad:
        print("\nBad episodes (index: reason):")
        for i, reason in bad_eps[:200]:
            print(f"  {i}: {reason}")
        if len(bad_eps) > 200:
            print(f"... and {len(bad_eps)-200} more")

if __name__ == "__main__":
    main()
