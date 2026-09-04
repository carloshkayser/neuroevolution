"""Top-level package initialization for neuroevolution."""

__version__ = "0.1.0"

try:
    import gymnasium_hardmaze
except ImportError:
    pass

try:
    import minigrid
except ImportError:
    pass

from .models import CartPoleNet, NeuralNetwork, FeedForwardNet
from .trainer import (
    GymOptimizationProblem,
    PyTorchGeneticTrainer,
    GymSingleObjectiveOutput,
    CarRacingObsWrapper,
    make_env,
)
from .agent import NeuralNetworkAgent, evaluate, load_model
from .tracker import TrainingJob, create_job, list_jobs

__all__ = [
    "__version__",
    "CartPoleNet",
    "NeuralNetwork",
    "FeedForwardNet",
    "GymOptimizationProblem",
    "PyTorchGeneticTrainer",
    "GymSingleObjectiveOutput",
    "CarRacingObsWrapper",
    "NeuralNetworkAgent",
    "evaluate",
    "load_model",
    "make_env",
    "TrainingJob",
    "create_job",
    "list_jobs",
]
