"""Tests for package imports and public API."""

import unittest


class TestImports(unittest.TestCase):
    def test_package_exports(self):
        import neuroevolution as ne

        self.assertTrue(hasattr(ne, "__version__"))
        self.assertTrue(hasattr(ne, "CartPoleNet"))
        self.assertTrue(hasattr(ne, "PyTorchGeneticTrainer"))
        self.assertTrue(hasattr(ne, "GymOptimizationProblem"))
        self.assertTrue(hasattr(ne, "GymSingleObjectiveOutput"))
        self.assertTrue(hasattr(ne, "NeuralNetworkAgent"))
        self.assertTrue(hasattr(ne, "evaluate"))
        self.assertTrue(hasattr(ne, "load_model"))
        self.assertTrue(hasattr(ne, "make_env"))
        self.assertTrue(hasattr(ne, "CarRacingObsWrapper"))

    def test_submodule_exports(self):
        from neuroevolution.models import CartPoleNet, FeedForwardNet, NeuralNetwork
        from neuroevolution.trainer import (
            GymOptimizationProblem,
            PyTorchGeneticTrainer,
            GymSingleObjectiveOutput,
            CarRacingObsWrapper,
            make_env,
        )
        from neuroevolution.agent import NeuralNetworkAgent, evaluate, load_model
        from neuroevolution.cli import parse_args, main

        self.assertIsNotNone(CartPoleNet)
        self.assertIsNotNone(FeedForwardNet)
        self.assertIsNotNone(NeuralNetwork)
        self.assertIsNotNone(GymOptimizationProblem)
        self.assertIsNotNone(PyTorchGeneticTrainer)
        self.assertIsNotNone(GymSingleObjectiveOutput)
        self.assertIsNotNone(CarRacingObsWrapper)
        self.assertIsNotNone(NeuralNetworkAgent)
        self.assertIsNotNone(evaluate)
        self.assertIsNotNone(load_model)
        self.assertIsNotNone(make_env)
        self.assertIsNotNone(parse_args)
        self.assertIsNotNone(main)


if __name__ == "__main__":
    unittest.main()
