"""
Minimal one-off evaluator (NEW method)
-------------------------------------
Usage:
    python eval_single_params.py --config-py NiCl2_config.py

This loads your material-specific config, uses its RIXS_runner instance (rixs_funV)
and experimental reference map (rixs_ref), and evaluates the positive L1 distance
between L1-normalized simulation and reference at a single parameter set.
"""

import argparse
import importlib.util
import json
import numpy as np
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Dict, Any

def load_config_module(path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("cfg_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_params_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Params JSON not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {p}: {e}") from e
    if not isinstance(data, dict):
        raise TypeError(f"Top-level JSON must be an object/dict, got: {type(data).__name__}")
    # If user wrapped under {"params": {...}}, allow it.
    if "params" in data and isinstance(data["params"], dict):
        data = data["params"]
    # Ensure values are JSON scalars; numpy will handle numeric casting later.
    for k, v in list(data.items()):
        if isinstance(v, (list, dict)):
            raise TypeError(f"Parameter '{k}' must be a scalar (number/string), got {type(v).__name__}")
    return data

"""
def l1_distance(sim: np.ndarray, ref: np.ndarray) -> float:
    sim = sim.astype(float)
    ref = ref.astype(float)
    # global L1 normalization (matches original objective)
    sim_norm = sim / np.sum(np.abs(sim))
    ref_norm = ref / np.sum(np.abs(ref))
    return float(np.sum(np.abs(sim_norm - ref_norm)))
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-py", required=True, help="Path to material config (e.g., NiCl2_config.py)")
    ap.add_argument("--params", required=True, help="Path to JSON file containing the parameter set")
    args = ap.parse_args()

    cfg = load_config_module(args.config_py)

    # --- Required objects from config ---
    # rixs_funV: an instance of RIXS_runner configured for this material
    # rixs_ref: experimental reference map (2D numpy array)
    try:
        greedy_bounds = cfg.greedy_bounds
        true_values = cfg.true_values              # dict: {param: value}
        funGreedy = cfg.funGreedy                  # callable with ordered args
    except AttributeError as e:
        raise RuntimeError(
            "Config must define 'greedy_bounds' (dict), 'true_values' (dict), and 'funGreedy' (callable)."
        ) from e
    if not isinstance(greedy_bounds, dict) or not isinstance(true_values, dict):
        raise TypeError("'greedy_bounds' and 'true_values' must both be dicts.")

    # --- Load parameter set from JSON ---
    params = load_params_json(args.params)
    #params: Dict[str, Any] = load_params_json(args.params)

    # If your config exposes default fixed params, use them to fill in any missing keys
    """
    for name in ("fixed_params_greedy", "fixed_params"):
        if hasattr(cfg, name):
            fixed = getattr(cfg, name)
            if isinstance(fixed, dict):
                for k, v in fixed.items():
                    params.setdefault(k, v)

    # Run the new path (RIXS_runner)
    #sim_map = rixs_runner(params)  # shape matches cfg target (usually [N_eloss, N_ominc])

    # Compute positive L1 distance versus the reference
    #dist = l1_distance(sim_map, rixs_ref)
    """
    # --- Key validation against greedy_bounds ---
    ordered_keys = list(greedy_bounds.keys())  # preserves intended argument order
    bounds_set = set(ordered_keys)
    json_set = set(params.keys())



    # (c) Any mismatching (extra) keys -> error
    extra = sorted(json_set - bounds_set)
    if extra:
        raise KeyError(
            "JSON contains keys not present in greedy_bounds: "
            + ", ".join(extra)
        )

    # (b) Missing keys -> warn and fill from true_values
    missing = sorted(bounds_set - json_set)
    if missing:
        # ensure true_values supplies them all
        missing_missing = [k for k in missing if k not in true_values]
        if missing_missing:
            raise KeyError(
                "Missing keys not found in true_values for backfill: "
                + ", ".join(missing_missing)
            )
        warnings.warn(
            "Missing keys in JSON were filled from true_values: "
            + ", ".join(missing),
            RuntimeWarning,
        )
        for k in missing:
            params[k] = true_values[k]

    # (a) Keys now match; build ordered positional args for funGreedy
    ordered_args = [params[k] for k in ordered_keys]

    # --- Call funGreedy ---
    result = funGreedy(*ordered_args)

    # Print results
    for k in ordered_keys:
        print(f"  {k:>8s} = {params[k]}")
    print(f"\nfunGreedy result: {result}")

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    sys.exit(main())