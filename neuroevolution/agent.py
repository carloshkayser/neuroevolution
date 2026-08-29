"""Agent wrapper and evaluation utilities."""

import os
import numpy as np


def load_model(model_path: str) -> np.ndarray | None:
    """Load a saved model or weights array from numpy file."""
    if os.path.exists(model_path):
        return np.load(model_path)
    else:
        print(f"Model file {model_path} not found!")
        return None


class NeuralNetworkAgent:
    """Agent wrapper class for policy evaluation."""
    
    def __init__(self, network, weights=None, env_type: str = 'CartPole'):
        self.network = network
        self.weights = weights
        self.env_type = env_type
    
    def predict(self, obs) -> int:
        """Predict action using the neural network."""
        # If the network exposes get_action (PyTorch implementation)
        if hasattr(self.network, 'get_action'):
            return self.network.get_action(obs, self.env_type)

        # Legacy API: network.feedforward(obs, weights)
        if hasattr(self.network, 'feedforward'):
            return self.network.feedforward(obs, self.weights)

        # If network is a callable
        if callable(self.network):
            return self.network(obs)

        raise ValueError('Unsupported network type for prediction')


def evaluate(agent, env, n_episodes: int = 20, render: bool = False, env_type: str = 'CartPole', max_steps: int = 1000) -> tuple[float, float]:
    """Evaluate the agent performance over multiple episodes."""
    rewards = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done, truncated = False, False
        total_reward = 0
        steps = 0
        
        while not (done or truncated) and steps < max_steps:
            if render:
                env.render()

            # Agent may be a NeuralNetworkAgent (with predict) or a PyTorch network
            if hasattr(agent, 'predict'):
                action = agent.predict(obs)
            elif hasattr(agent, 'get_action'):
                action = agent.get_action(obs, env_type)
            else:
                # Fallback: assume callable
                action = agent(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        
        rewards.append(total_reward)
        print(f"Episode {ep+1}: reward = {total_reward}")
    
    avg_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    
    print(f"\nAverage reward over {n_episodes} episodes: {avg_reward:.2f}")
    print(f"Standard deviation: {std_reward:.2f}")
    
    return avg_reward, std_reward

