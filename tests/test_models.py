"""Tests for neural network models."""

import unittest
import numpy as np
import torch
from neuroevolution.models import CartPoleNet


class TestCartPoleNet(unittest.TestCase):
    def setUp(self):
        self.net = CartPoleNet(input_size=4, hidden_sizes=[16, 8], output_size=1)

    def test_total_parameters(self):
        # input(4) -> h1(16): 4*16 + 16 = 80
        # h1(16) -> h2(8): 16*8 + 8 = 136
        # h2(8) -> out(1): 8*1 + 1 = 9
        # Total: 80 + 136 + 9 = 225
        total_params = self.net.get_total_params()
        self.assertEqual(total_params, 225)

    def test_get_and_set_weights(self):
        weights = self.net.get_weights_as_vector()
        self.assertEqual(len(weights), 225)

        new_weights = np.ones(225, dtype=np.float32)
        self.net.set_weights_from_vector(new_weights)
        updated_weights = self.net.get_weights_as_vector().detach().numpy()
        np.testing.assert_allclose(updated_weights, new_weights)

    def test_forward_pass(self):
        x = torch.zeros((1, 4))
        out = self.net(x)
        self.assertEqual(out.shape, (1, 1))

    def test_get_action_cartpole(self):
        state = np.array([0.0, 0.1, -0.1, 0.0])
        action = self.net.get_action(state, env_type='CartPole')
        self.assertIn(action, [0, 1])

    def test_get_action_lunarlander(self):
        lander_net = CartPoleNet(input_size=8, hidden_sizes=[16, 8], output_size=4)
        state = np.zeros(8)
        action = lander_net.get_action(state, env_type='LunarLander')
        self.assertIn(action, [0, 1, 2, 3])

    def test_get_action_mountaincar(self):
        car_net = CartPoleNet(input_size=2, hidden_sizes=[16, 8], output_size=3)
        state = np.array([-0.5, 0.0])
        action = car_net.get_action(state, env_type='MountainCar')
        self.assertIn(action, [0, 1, 2])

    def test_get_action_maze(self):
        maze_net = CartPoleNet(input_size=9, hidden_sizes=[16, 12], output_size=3)
        state = np.ones(9)
        action = maze_net.get_action(state, env_type='Maze')
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.shape, (3,))
        self.assertTrue(np.all(action >= 0.0) and np.all(action <= 1.0))

    def test_get_action_carracing(self):
        racing_net = CartPoleNet(input_size=256, hidden_sizes=[16, 12], output_size=3)
        state = np.zeros(256, dtype=np.float32)
        action = racing_net.get_action(state, env_type='CarRacing')
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.shape, (3,))
        steering, gas, brake = action
        self.assertTrue(-1.0 <= steering <= 1.0)
        self.assertTrue(0.0 <= gas <= 1.0)
        self.assertTrue(0.0 <= brake <= 1.0)

    def test_multidim_forward(self):
        racing_net = CartPoleNet(input_size=16 * 16, hidden_sizes=[16], output_size=3)
        img = torch.zeros((1, 16, 16))
        out = racing_net(img)
        self.assertEqual(out.shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
