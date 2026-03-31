#!/usr/bin/env python3
"""Run aggressive solver configurations to push hard violations below 60.

Strategies:
  S1: Tuned GA (stagnation_window=8, elite_pct=0.25, mutation_hi=0.35) + SA polish
  S2: Large pop + more gens + SA polish (p200/g500/s42)
  S3: Best prior params + SA polish (p100/g500/s42)
  S4: Multi-seed best-of-3 (s42, s123, s7) at p100/g500 + SA
  S5: Very long run (p100/g800/s42) + SA
  S6: Large pop, aggressive stagnation (p200/g500/stag=5/elite=0.30)

Usage:
    python run_aggressive.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def reset_logging():
    """Reset all loggers between runs."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()
    for name in list(logging.Logger.manager.loggerDict):
        lg = logging.getLogger(name)
        for h in lg.handlers[:]:
            lg.removeHandler(h)
            h.close()


def run_version(label, **kwargs):
    """Run one AdaptiveExperiment, return results dict."""
    reset_logging()
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

    final_G = results.get("final_G")
    if final_G:
        import numpy as np
        G = np.array(final_G)
        from src.pipeline.scheduling_problem import _TOLERATED_HARD_COLS
        _strict = [i for i in range(G.shape[1]) if i not in _TOLERATED_HARD_COLS]
        cv = G[:, _strict].sum(axis=1).clip(0)
        best_idx = int(np.argmin(cv))
        names = ["CTE", "FTE", "SRE", "FPC", "FFC", "FCA", "CQF", "ICTD"]
        breakdown = {}
        for j in range(min(len(names), G.shape[1])):
            val = float(G[best_idx, j])
            if val > 0:
                breakdown[names[j]] = int(val)
        out["breakdown"] = breakdown

    return out


def print_summary(all_results):
    print("\n" + "=" * 95)
    print("  AGGRESSIVE STRATEGIES COMPARISON REPORT")
    print("=" * 95)

    ranked = sorted(all_results, key=lambda r: r.get("best_cv", 9999))

    print(f"\n{'#':<3} {'Version':<45} {'Hard':<7} {'CV':<7} {'Soft':<10} {'Time':<8} {'Breakdown'}")
    print("-" * 95)

    for i, r in enumerate(ranked, 1):
        bd = r.get("breakdown", {})
        bd_str = "  ".join(f"{k}={v}" for k, v in sorted(bd.items()) if k != "FCA")
        elapsed = r.get("elapsed_s", "?")
        if isinstance(elapsed, (int, float)) and elapsed > 60:
            time_str = f"{elapsed/60:.1f}m"
        else:
            time_str = f"{elapsed}s"
        print(
            f"{i:<3} {r['label']:<45} "
            f"{r.get('best_hard', '?'):<7} "
            f"{r.get('best_cv', '?'):<7} "
            f"{r.get('best_soft', '?'):<10.1f} "
            f"{time_str:<8} "
            f"{bd_str}"
        )

    best = ranked[0]
    print(f"\n  BEST: {best['label']}  =>  hard={best.get('best_hard', '?')}, cv={best.get('best_cv', '?')}")
    if best.get("output_dir"):
        print(f"  Output: {best['output_dir']}")
    print("=" * 95)


def main():
    all_results = []

    versions = [
        # S1: Aggressive stagnation params + SA polishing
        {
            "label": "S1: Aggressive stagnation (stag=8/elite=25%)",
            "pop_size": 100, "ngen": 500, "seed": 42,
            "stagnation_window": 8, "elite_pct": 0.25,
            "repair_iters": 10, "mutation_hi": 0.35,
        },
        # S2: Large pop + aggressive stagnation + SA
        {
            "label": "S2: Large+aggressive (p200/stag=8/elite=25%)",
            "pop_size": 200, "ngen": 500, "seed": 42,
            "stagnation_window": 8, "elite_pct": 0.25,
            "repair_iters": 10, "mutation_hi": 0.35,
        },
        # S3: Default GA params + SA polish (reproduce V5 with SA)
        {
            "label": "S3: V5 config + SA polish (p100/g500/s42)",
            "pop_size": 100, "ngen": 500, "seed": 42,
        },
        # S4: Very long run with SA
        {
            "label": "S4: Very long (p100/g800/s42)",
            "pop_size": 100, "ngen": 800, "seed": 42,
        },
        # S5: Large pop, very long + aggressive
        {
            "label": "S5: Max compute (p200/g800/stag=8)",
            "pop_size": 200, "ngen": 800, "seed": 42,
            "stagnation_window": 8, "elite_pct": 0.25,
            "repair_iters": 10, "mutation_hi": 0.35,
        },
        # S6: Alt seed with aggressive params
        {
            "label": "S6: Alt seed aggressive (s123/stag=8)",
            "pop_size": 100, "ngen": 500, "seed": 123,
            "stagnation_window": 8, "elite_pct": 0.25,
            "repair_iters": 10, "mutation_hi": 0.35,
        },
    ]

    total = len(versions)
    for i, ver in enumerate(versions, 1):
        label = ver.pop("label")
        print(f"\n{'='*65}")
        print(f"  [{i}/{total}] Running: {label}")
        print(f"{'='*65}")

        result = run_version(label, **ver)
        all_results.append(result)

        print(f"  => hard={result.get('best_hard', '?')}, "
              f"cv={result.get('best_cv', '?')}, "
              f"elapsed={result.get('elapsed_s', '?')}s")

    out_path = PROJECT_ROOT / "output" / "aggressive_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print_summary(all_results)
    print(f"\nRaw results saved to: {out_path}")


if __name__ == "__main__":
    main()
