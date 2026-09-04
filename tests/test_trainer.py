"""Tests for PyMOO trainer and GymSingleObjectiveOutput."""

import os
import shutil
import unittest
from pymoo.algorithms.soo.nonconvex.ga import GA
from neuroevolution.trainer import (
    GymOptimizationProblem,
    PyTorchGeneticTrainer,
    GymSingleObjectiveOutput,
    make_env,
)


class TestTrainer(unittest.TestCase):
    def tearDown(self):
        test_dir = 'scratch_test_checkpoints'
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

    def test_make_env_minigrid(self):
        env = make_env('MiniGrid-FourRooms-v0')
        self.assertEqual(len(env.observation_space.shape), 1)
        self.assertEqual(env.observation_space.shape[0], 2835)
        obs, _ = env.reset(seed=42)
        self.assertEqual(obs.shape, (2835,))
        env.close()

    def test_gym_single_objective_output_columns(self):
        output = GymSingleObjectiveOutput()
        self.assertIsNotNone(output.reward_max)
        self.assertIsNotNone(output.reward_avg)
        self.assertIsNotNone(output.reward_best)

        # Create problem and algorithm
        problem = GymOptimizationProblem('CartPole-v1', [4, 3], max_steps_per_eval=50)
        algorithm = GA(pop_size=6, output=output)
        algorithm.setup(problem, termination=('n_gen', 2))

        # Run 1 step and trigger output update
        algorithm.next()
        output(algorithm)

        # Check that reward columns have values
        self.assertIsNotNone(output.reward_max.value)
        self.assertIsNotNone(output.reward_avg.value)
        self.assertIsNotNone(output.reward_best.value)

        max_val = float(output.reward_max.value)
        avg_val = float(output.reward_avg.value)
        best_val = float(output.reward_best.value)

        self.assertGreaterEqual(max_val, avg_val)
        self.assertGreaterEqual(best_val, max_val)

    def test_trainer_train_smoke(self):
        trainer = PyTorchGeneticTrainer(
            env_name='CartPole-v1',
            network_architecture=[4],
            model_prefix='test_cartpole',
            checkpoint_dir='scratch_test_checkpoints',
        )
        best_net, best_fitness, n_gens = trainer.train(
            n_generations=2,
            population_size=6,
        )
        self.assertIsNotNone(best_net)
        self.assertIsInstance(best_fitness, float)
        self.assertEqual(n_gens, 2)
        self.assertEqual(len(trainer.fitness_history), 2)

    def test_trainer_fourrooms_smoke(self):
        trainer = PyTorchGeneticTrainer(
            env_name='MiniGrid-FourRooms-v0',
            network_architecture=[8],
            model_prefix='test_fourrooms',
            checkpoint_dir='scratch_test_checkpoints',
        )
        best_net, best_fitness, n_gens = trainer.train(
            n_generations=2,
            population_size=4,
        )
        self.assertIsNotNone(best_net)
        self.assertIsInstance(best_fitness, float)
    def test_make_env_carracing(self):
        # Default sensor mode (16 features)
        env = make_env('CarRacing-v0')
        self.assertEqual(env.observation_space.shape, (16,))
        obs, _ = env.reset(seed=42)
        self.assertEqual(obs.shape, (16,))
        self.assertTrue(obs.min() >= -1.0 and obs.max() <= 1.0)
        env.close()

        # Legacy downsampled mode (256 features)
        env_down = make_env('CarRacing-v0', obs_mode='downsample')
        self.assertEqual(env_down.observation_space.shape, (256,))
        obs_down, _ = env_down.reset(seed=42)
        self.assertEqual(obs_down.shape, (256,))
        self.assertTrue(obs_down.min() >= 0.0 and obs_down.max() <= 1.0)
        env_down.close()

    def test_problem_carracing(self):
        problem = GymOptimizationProblem('CarRacing-v0', [16, 12], max_steps_per_eval=20)
        self.assertEqual(problem.env_type, 'CarRacing')
        self.assertEqual(problem.input_size, 16)
        self.assertEqual(problem.output_size, 3)

    def test_trainer_carracing_smoke(self):
        trainer = PyTorchGeneticTrainer(
            env_name='CarRacing-v0',
            network_architecture=[8],
            model_prefix='test_carracing',
            checkpoint_dir='scratch_test_checkpoints',
            n_episodes_per_eval=1,
            max_steps=10,
        )
        self.assertEqual(trainer.env_type, 'CarRacing')
        self.assertEqual(trainer.problem.output_size, 3)
        best_net, best_fitness, n_gens = trainer.train(
            n_generations=2,
            population_size=4,
        )
        self.assertIsNotNone(best_net)
        self.assertIsInstance(best_fitness, float)
        self.assertEqual(n_gens, 2)
        self.assertEqual(len(trainer.fitness_history), 2)

    def test_trainer_record_gif(self):
        trainer = PyTorchGeneticTrainer(
            env_name='CartPole-v1',
            network_architecture=[4],
            model_prefix='test_cartpole_gif',
            checkpoint_dir='scratch_test_checkpoints',
        )
        trainer.train(n_generations=1, population_size=4)
        gif_output = os.path.join('scratch_test_checkpoints', 'test_cartpole.gif')
        result_path = trainer.record_gif(output_path=gif_output, max_steps=5, fps=10)
        self.assertEqual(result_path, gif_output)
        self.assertTrue(os.path.exists(gif_output))
        self.assertGreater(os.path.getsize(gif_output), 0)

    def test_trainer_live_learning_curve(self):
        trainer = PyTorchGeneticTrainer(
            env_name='CartPole-v1',
            network_architecture=[4],
            model_prefix='test_live_plot',
            checkpoint_dir='scratch_test_checkpoints',
        )
        trainer.train(n_generations=3, population_size=4, live_plot=True)
        plot_path = os.path.join('scratch_test_checkpoints', 'test_live_plot_learning_curve.png')
        self.assertTrue(os.path.exists(plot_path))
        self.assertGreater(os.path.getsize(plot_path), 0)
        self.assertEqual(len(trainer.history_generations), 3)
        self.assertEqual(len(trainer.history_mean), 3)
        self.assertEqual(len(trainer.history_best), 3)
        self.assertEqual(len(trainer.history_episodes), 3)
        self.assertEqual(trainer.history_generations, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
