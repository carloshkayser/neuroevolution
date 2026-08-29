"""Tests for agent wrapper."""

import unittest
import numpy as np
from neuroevolution.models import CartPoleNet
from neuroevolution.agent import NeuralNetworkAgent


class TestAgent(unittest.TestCase):
    def test_agent_prediction(self):
        net = CartPoleNet(input_size=4, hidden_sizes=[8], output_size=1)
        agent = NeuralNetworkAgent(net, env_type='CartPole')

        obs = np.array([0.1, -0.2, 0.05, 0.1])
        action = agent.predict(obs)
        self.assertIn(action, [0, 1])

    def test_agent_callable_fallback(self):
        agent = NeuralNetworkAgent(lambda obs: 1)
        obs = np.array([0.0, 0.0])
        action = agent.predict(obs)
        self.assertEqual(action, 1)

    def test_agent_prediction_fourrooms(self):
        net = CartPoleNet(input_size=2835, hidden_sizes=[16], output_size=7)
        agent = NeuralNetworkAgent(net, env_type='FourRooms')
        obs = np.zeros(2835, dtype=np.float32)
        action = agent.predict(obs)
        self.assertIn(action, list(range(7)))


if __name__ == "__main__":
    unittest.main()
