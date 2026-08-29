"""Tests for CLI arguments."""

import unittest
from neuroevolution.cli import parse_args


class TestCli(unittest.TestCase):
    def test_default_args(self):
        args = parse_args([])
        self.assertEqual(args.env, 'CartPole')
        self.assertEqual(args.generations, 100)
        self.assertEqual(args.population, 50)
        self.assertIsNone(args.hidden_layers)

    def test_custom_hidden_layers(self):
        args = parse_args(['--hidden-layers', '64', '32', '16', '--env', 'LunarLander'])
        self.assertEqual(args.hidden_layers, [64, 32, 16])
        self.assertEqual(args.env, 'LunarLander')

    def test_mountaincar_args(self):
        args = parse_args(['--env', 'MountainCar'])
        self.assertEqual(args.env, 'MountainCar')

    def test_maze_args(self):
        args = parse_args(['--env', 'Maze'])
        self.assertEqual(args.env, 'Maze')

    def test_fourrooms_args(self):
        args = parse_args(['--env', 'FourRooms'])
        self.assertEqual(args.env, 'FourRooms')

    def test_carracing_args(self):
        args = parse_args(['--env', 'CarRacing'])
        self.assertEqual(args.env, 'CarRacing')

    def test_carracing_v0_args(self):
        args = parse_args(['--env', 'CarRacing-v0'])
        self.assertEqual(args.env, 'CarRacing-v0')

    def test_record_gif_args(self):
        args = parse_args(['--record-gif', '--env', 'CartPole', '--gif-path', 'assets/custom_cartpole.gif'])
        self.assertTrue(args.record_gif)
        self.assertEqual(args.gif_path, 'assets/custom_cartpole.gif')
        self.assertEqual(args.env, 'CartPole')


if __name__ == "__main__":
    unittest.main()
