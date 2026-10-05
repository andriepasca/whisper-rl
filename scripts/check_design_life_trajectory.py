"""Design-life trajectory check for DEL references (PI's definition of a correct DEL_ref).

CRITERION (PI definition, NOT derived from the paper's specification by Ada):
    With all-zero (NO-OP) actions, initial HI = 1 and no randomization, the STOCHASTIC
    blade-damage trajectory must reach end-of-life (HI <= 0, i.e. cumulative combined
    damage = 1) around the design life (20 y). Operationally: the MEDIAN end-of-life
    time over N independent episodes must lie within +/- tol (default 10 %) of
    design_life_years (default 20 y). PASS/FAIL is reported per reference set.

Reference sets compared:
    fixed       : make_fixed_damage_solver_config() constants (config.DEFAULT_DEL_*_REF)
    calibrated  : both refs None -> OffshoreMaintenanceEnv._calibrate_del_refs() (auto)

Notes / technical choices (no change to physics, rewards or constants):
  * Overrides applied via dataclasses.replace: max_simulation_years (horizon, default 60),
    randomize_initial_hi=False, decision_interval_months=1 (so HI is observed every month;
    damage per month is identical to the 6-month stepping; no-op => no maintenance).
  * Env tiles ONE wind draw across all turbines, so turbines within an episode are
    perfectly correlated; default n_turbines=1 and each EPISODE is one independent sample.
  * HI is combined (mean of flap and edge damage); per-component lifetimes are not
    exposed by the env and are not reported.
  * End-of-life = first elapsed month with HI <= 0. Episodes not failing within the horizon
    are right-censored (lifetime recorded = horizon, flagged in CSV; statistics are then
    lower bounds).
  * Calibration uses env RNG seeded with --calib-seed (one env per ref set; episodes use
    reset(seed=...)).

Usage (from the whisper-rl directory):
    PYTHONPATH=. python scripts/check_design_life_trajectory.py
    PYTHONPATH=. python scripts/check_design_life_trajectory.py --n-episodes 50 --tol 0.1
    PYTHONPATH=. python scripts/check_design_life_trajectory.py --seeds 1 2 3 --out lifetime_check.csv
"""
import argparse
import csv
import dataclasses
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from whisper_env.config import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DamageSolverConfig,
    get_default_config,
    make_fixed_damage_solver_config,
)
from whisper_env.environment import OffshoreMaintenanceEnv  # noqa: E402
from whisper_env.physics import DamageSolver  # noqa: E402


def build_env(ref_set, csv_path, n_turbines, calib_seed, horizon_years):
    config = get_default_config(
        n_turbines=n_turbines, seed=calib_seed
    )
    overrides = dict(
        max_simulation_years=horizon_years,
        randomize_initial_hi=False,
        decision_interval_months=1,
        enable_logging=False,
    )
    if ref_set == "fixed":
        overrides["damage_solver"] = DamageSolver(make_fixed_damage_solver_config(csv_path))
    elif ref_set == "calibrated":
        # both refs None -> env auto-calibrates in __init__ (uses the given CSV)
        overrides["damage_solver"] = DamageSolver(DamageSolverConfig(csv_path=csv_path))
    else:
        raise ValueError(ref_set)
    config = dataclasses.replace(config, **overrides)
    return OffshoreMaintenanceEnv(config)


def run_episode(env, seed, design_months):
    """Returns (lifetime_months[n_turb] (nan if censored), HI at design life[n_turb])."""
    env.reset(seed=seed)
    n = env.unwrapped.n_turbines
    life = np.full(n, np.nan)
    hi_design = np.full(n, np.nan)
    action = np.zeros(n, dtype=np.int32)
    truncated = False
    while not truncated:
        _, _, terminated, truncated, _ = env.step(action)
        t = env.unwrapped.elapsed_month
        hi = env.unwrapped.HI
        newly = np.isnan(life) & (hi <= 0.0)
        life[newly] = t
        if t == design_months:
            hi_design = hi.copy()
        if not np.isnan(life).any() and t >= design_months:
            break
    return life, hi_design


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Explicit episode seeds (overrides --n-episodes).")
    p.add_argument("--n-turbines", type=int, default=1)
    p.add_argument("--horizon-years", type=int, default=60)
    p.add_argument("--design-life", type=float, default=20.0)
    p.add_argument("--tol", type=float, default=0.10, help="Relative tolerance on median lifetime.")
    p.add_argument("--calib-seed", type=int, default=42)
    p.add_argument("--damage-csv", type=str, default=os.path.join(DEFAULT_DATA_DIR, "response_surface.csv"))
    p.add_argument("--out", type=str, default="lifetime_check.csv")
    args = p.parse_args()

    if not os.path.exists(args.damage_csv):
        sys.exit(f"ERROR: damage CSV not found: {args.damage_csv}")
    seeds = args.seeds if args.seeds else list(range(args.n_episodes))
    design_months = int(round(args.design_life * 12))
    if args.horizon_years * 12 <= design_months:
        sys.exit("ERROR: --horizon-years must exceed the design life.")

    rows, verdicts = [], {}
    for ref_set in ("fixed", "calibrated"):
        env = build_env(ref_set, args.damage_csv, args.n_turbines, args.calib_seed, args.horizon_years)
        cfg = env.damage_solver.config
        print(f"=== {ref_set}: del_flap_ref={cfg.del_flap_ref!r} del_edge_ref={cfg.del_edge_ref!r} "
              f"m={cfg.m_coef} design_life={cfg.design_life_years} ===")
        lifetimes, hi20, cens = [], [], 0
        for s in seeds:
            life, hid = run_episode(env, s, design_months)
            for j in range(len(life)):
                censored = bool(np.isnan(life[j]))
                cens += censored
                yrs = (args.horizon_years if censored else life[j] / 12.0)
                lifetimes.append(yrs)
                hi20.append(hid[j])
                rows.append(dict(ref_set=ref_set, del_flap_ref=cfg.del_flap_ref, del_edge_ref=cfg.del_edge_ref,
                                 seed=s, turbine=j, lifetime_years=yrs, censored=censored,
                                 hi_at_design_life=float(hid[j])))
        L = np.asarray(lifetimes)
        H = np.asarray(hi20)
        med = float(np.median(L))
        rel = (med - args.design_life) / args.design_life
        ok = abs(rel) <= args.tol
        verdicts[ref_set] = ok
        print(f"lifetime [y]: mean={L.mean():.3f} median={med:.3f} std={L.std(ddof=1) if len(L) > 1 else float('nan'):.3f} "
              f"min={L.min():.3f} max={L.max():.3f}  (n={len(L)}, censored={cens})")
        print(f"fraction failing before {args.design_life:g} y: {np.mean(L < args.design_life):.3f}")
        print(f"HI at {args.design_life:g} y: mean={np.nanmean(H):.4f} median={np.nanmedian(H):.4f}")
        print(f"median rel. deviation from design life: {rel:+.3%} (tol={args.tol:g}) -> {'PASS' if ok else 'FAIL'}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out}")
    print("SUMMARY:", {k: ('PASS' if v else 'FAIL') for k, v in verdicts.items()})
    sys.exit(0 if all(verdicts.values()) else 1)


if __name__ == "__main__":
    main()
