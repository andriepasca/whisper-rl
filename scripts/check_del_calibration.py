"""Provenance check: compare auto-calibrated DEL refs with the shared fixed constants.

Builds the env with get_default_config(seed=S) (auto-calibration, default damage CSV)
and prints calibrated del_flap_ref / del_edge_ref next to DEFAULT_DEL_*_REF.

Usage (from the whisper-rl directory):
    PYTHONPATH=. python scripts/check_del_calibration.py
    PYTHONPATH=. python scripts/check_del_calibration.py --seeds 42 100 2024
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from whisper_env import (
    get_default_config,
    OffshoreMaintenanceEnv,
    DEFAULT_DEL_FLAP_REF,
    DEFAULT_DEL_EDGE_REF,
)


def check(seed, tol):
    config = get_default_config(seed=seed)
    env = OffshoreMaintenanceEnv(config)
    cfg = env.damage_solver.config
    ok_all = True
    print(f"--- seed={seed} ---")
    for name, calibrated, fixed in (
        ("del_flap_ref", cfg.del_flap_ref, DEFAULT_DEL_FLAP_REF),
        ("del_edge_ref", cfg.del_edge_ref, DEFAULT_DEL_EDGE_REF),
    ):
        rel = abs(calibrated - fixed) / abs(fixed)
        ok = rel <= tol
        ok_all = ok_all and ok
        print(f"{name}: calibrated={calibrated!r}  fixed={fixed!r}  "
              f"rel_diff={rel:.3e}  {'PASS' if ok else 'FAIL'} (tol={tol:g})")
    return ok_all


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--tol", type=float, default=1e-3)
    args = p.parse_args()
    results = {s: check(s, args.tol) for s in args.seeds}
    print("SUMMARY:", {s: ('PASS' if r else 'FAIL') for s, r in results.items()})
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
