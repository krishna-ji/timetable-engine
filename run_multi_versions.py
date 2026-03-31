#!/usr/bin/env python3
"""Run multiple solver configurations to minimise hard constraint violations.

Each version uses a different strategy. Results are collected and compared
at the end in a summary table.

Usage:
    python run_multi_versions.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_version(label: str, **kwargs) -> dict:
    """Run one AdaptiveExperiment with given params, return results dict."""
    # Reset logging handlers between runs to avoid stale file handles
    import logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()
    for name in list(logging.Logger.manager.loggerDict):
        lg = logging.getLogger(name)
        for handler in lg.handlers[:]:
            lg.removeHandler(handler)
            handler.close()

    from src.experiments import AdaptiveExperiment

    kwargs.setdefault("export_pdf", False)
    kwargs.setdefault("verbose", True)

    exp = AdaptiveExperiment(**kwargs)
    t0 = time.time()
    try:
        results = exp.run()
    except Exception as e:
        print(f"  [FAILED] {label}: {e}")
        return {"label": label, "error": str(e), "best_hard": 9999, "best_cv": 9999}
    elapsed = time.time() - t0

    out = {
        "label": label,
        "best_hard": results.get("best_hard", 9999),
        "best_soft": results.get("best_soft", 0),
        "best_cv": results.get("best_cv", 9999),
        "elapsed_s": round(elapsed, 1),
        "n_feasible": results.get("n_feasible", 0),
        "output_dir": str(exp.output_dir),
    }

    # Extract per-constraint breakdown from final G
    final_G = results.get("final_G")
    if final_G:
        import numpy as np
        G = np.array(final_G)
        from src.pipeline.scheduling_problem import _TOLERATED_HARD_COLS
        _strict = [i for i in range(G.shape[1]) if i not in _TOLERATED_HARD_COLS]
        cv = G[:, _strict].sum(axis=1).clip(0)
        best_idx = int(np.argmin(cv))
        constraint_names = ["CTE", "FTE", "SRE", "FPC", "FFC", "FCA", "CQF", "ICTD"]
        breakdown = {}
        for j in range(min(len(constraint_names), G.shape[1])):
            val = float(G[best_idx, j])
            if val > 0:
                breakdown[constraint_names[j]] = int(val)
        out["breakdown"] = breakdown

    convergence = results.get("convergence_hard", [])
    if convergence:
        out["convergence_last5"] = convergence[-5:]

    return out


def print_summary(all_results: list[dict]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print("  MULTI-VERSION COMPARISON REPORT")
    print("=" * 90)

    # Sort by best_hard ascending
    ranked = sorted(all_results, key=lambda r: r.get("best_cv", 9999))

    print(f"\n{'#':<3} {'Version':<35} {'Hard':<7} {'CV':<7} {'Soft':<8} {'Time(s)':<8} {'Breakdown'}")
    print("-" * 90)

    for i, r in enumerate(ranked, 1):
        bd = r.get("breakdown", {})
        bd_str = "  ".join(f"{k}={v}" for k, v in sorted(bd.items())) if bd else r.get("error", "N/A")
        print(
            f"{i:<3} {r['label']:<35} "
            f"{r.get('best_hard', '?'):<7} "
            f"{r.get('best_cv', '?'):<7} "
            f"{r.get('best_soft', '?'):<8} "
            f"{r.get('elapsed_s', '?'):<8} "
            f"{bd_str}"
        )

    best = ranked[0]
    print(f"\n  BEST: {best['label']}  =>  hard={best.get('best_hard', '?')}, cv={best.get('best_cv', '?')}")
    if best.get("output_dir"):
        print(f"  Output: {best['output_dir']}")
    print("=" * 90)


def main():
    all_results = []

    versions = [
        # ----- V1: Baseline (seed=42, pop=100, gens=300) -----
        {
            "label": "V1: Baseline (p100/g300/s42)",
            "pop_size": 100, "ngen": 300, "seed": 42,
        },
        # ----- V2: Different seed -----
        {
            "label": "V2: Alt seed (p100/g300/s123)",
            "pop_size": 100, "ngen": 300, "seed": 123,
        },
        # ----- V3: Different seed -----
        {
            "label": "V3: Alt seed (p100/g300/s7)",
            "pop_size": 100, "ngen": 300, "seed": 7,
        },
        # ----- V4: Larger population -----
        {
            "label": "V4: Large pop (p200/g300/s42)",
            "pop_size": 200, "ngen": 300, "seed": 42,
        },
        # ----- V5: More generations -----
        {
            "label": "V5: More gens (p100/g500/s42)",
            "pop_size": 100, "ngen": 500, "seed": 42,
        },
        # ----- V6: Large pop + more gens -----
        {
            "label": "V6: Big run (p200/g500/s42)",
            "pop_size": 200, "ngen": 500, "seed": 42,
        },
        # ----- V7: Higher mutation -----
        {
            "label": "V7: High mut (p100/g300/s42/mut=0.10)",
            "pop_size": 100, "ngen": 300, "seed": 42,
            "mutation_event_prob": 0.10,
        },
        # ----- V8: Lower crossover, higher mutation -----
        {
            "label": "V8: Tuned ops (cx=0.3/mut=0.08/s42)",
            "pop_size": 100, "ngen": 300, "seed": 42,
            "crossover_prob": 0.3, "mutation_event_prob": 0.08,
        },
    ]

    total = len(versions)
    for i, ver in enumerate(versions, 1):
        label = ver.pop("label")
        print(f"\n{'='*60}")
        print(f"  [{i}/{total}] Running: {label}")
        print(f"{'='*60}")

        result = run_version(label, **ver)
        all_results.append(result)

        print(f"  => hard={result.get('best_hard', '?')}, "
              f"cv={result.get('best_cv', '?')}, "
              f"elapsed={result.get('elapsed_s', '?')}s")

    # Save raw results
    out_path = PROJECT_ROOT / "output" / "multi_version_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print_summary(all_results)
    print(f"\nRaw results saved to: {out_path}")


if __name__ == "__main__":
    main()
