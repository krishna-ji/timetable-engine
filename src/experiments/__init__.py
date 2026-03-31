"""Experiment runners — clean OOP wrappers for GA experiments.

Each experiment is configured via constructor kwargs, then executed
with ``experiment.run()``.  Logging, output directories, timing, and
JSON result export are handled by the base class.
"""

from .ga_experiment import AdaptiveExperiment, GAExperiment

__all__ = [
    "AdaptiveExperiment",
    "GAExperiment",
]
