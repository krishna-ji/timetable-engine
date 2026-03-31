"""
Schedule Engine — University Course Scheduling with NSGA-II

A production-grade genetic algorithm scheduling engine that optimizes
university timetables using NSGA-II with adaptive stagnation-aware
mutation escalation and constraint-guided repair.

Package Structure:
    - domain/:       Domain models (Course, Group, Instructor, Room)
    - constraints/:  Hard and soft constraint definitions & evaluator
    - ga/:           Genetic algorithm operators, repair, metrics
    - io/:           Data loading, export (JSON, PDF, CSV)
    - pipeline/:     pymoo integration (encoding, evaluation, repair)
    - experiments/:  Experiment runner (AdaptiveExperiment)
    - config/:       Configuration loader
    - utils/:        Logging, console, parallel workers

Usage:
    python solve.py --gens 300 --pop 100 --seed 42

Authors:
    Krishna Acharya, Dinanath Padhya, Bipul Dahal

License:
    MIT
"""

__version__ = "2.0.0"
__license__ = "MIT"

__all__ = [
    "__author__",
    "__license__",
    "__version__",
]
