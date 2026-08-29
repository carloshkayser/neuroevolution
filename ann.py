"""Legacy module re-exporting classes from neuroevolution package for backwards compatibility."""

from neuroevolution import (
    CartPoleNet,
    GymOptimizationProblem,
    PyTorchGeneticTrainer,
)

__all__ = [
    "CartPoleNet",
    "GymOptimizationProblem",
    "PyTorchGeneticTrainer",
]
