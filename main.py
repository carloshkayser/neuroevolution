from ann import PyTorchGeneticTrainer, CartPoleNet

import gymnasium as gym
import numpy as np
import argparse
import os

# Function to load a saved model
def load_model(model_path):
    """Load a saved model from numpy file"""
    if os.path.exists(model_path):
        return np.load(model_path)
    else:
        print(f"Model file {model_path} not found!")
        return None

# Agent wrapper class for evaluation
class NeuralNetworkAgent:
    def __init__(self, network, weights, env_type='CartPole'):
        # network can be a PyTorch CartPoleNet (with get_action),
        # a legacy network with feedforward(network, weights), or a callable.
        self.network = network
        self.weights = weights
        self.env_type = env_type
def load_model(model_path):
    """Load a saved model from numpy file"""
    if os.path.exists(model_path):
        return np.load(model_path)
    else:
        print(f"Model file {model_path} not found!")
        return None

# Agent wrapper class for evaluation
class NeuralNetworkAgent:
    def __init__(self, network, weights):
        # network can be a PyTorch CartPoleNet (with get_action),
        # a legacy network with feedforward(network, weights), or a callable.
        self.network = network
        self.weights = weights
        self.env_type = env_type
    
    def predict(self, obs):
        """Predict action using the neural network"""
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

def evaluate(agent, env, n_episodes=20, render=False, env_type='CartPole'):
    """Evaluate the agent performance over multiple episodes"""
    rewards = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done, truncated = False, False
        total_reward = 0
        
        while not (done or truncated):
            if render:
                env.render()

            # Agent may be a NeuralNetworkAgent (with predict) or a PyTorch network
            if hasattr(agent, 'predict'):
                action = agent.predict(obs)
            elif hasattr(agent, 'get_action'):
                # Default to CartPole env type for get_action
                action = agent.get_action(obs, env_type)
            else:
                # Fallback: assume callable
                action = agent(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
        
        rewards.append(total_reward)
        print(f"Episode {ep+1}: reward = {total_reward}")
    
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    print(f"\nAverage reward over {n_episodes} episodes: {avg_reward:.2f}")
    print(f"Standard deviation: {std_reward:.2f}")
    
    return avg_reward, std_reward

def main():
    parser = argparse.ArgumentParser(description='Neural Network with Genetic Algorithm for Gym Environments')
    parser.add_argument('--train', action='store_true', help='Train the neural network')
    parser.add_argument('--test', action='store_true', help='Test/evaluate the trained agent')
    # parser.add_argument('--env', choices=['CartPole', 'MountainCar'], default='CartPole', 
    #                    help='Environment to use: CartPole (CartPole-v1) or MountainCar (MountainCar-v0)')
    parser.add_argument('--env', choices=['CartPole', 'LunarLander'], default='CartPole', 
                       help='Environment to use: CartPole (CartPole-v1) or LunarLander (LunarLander-v2)')
    parser.add_argument('--generations', type=int, default=100, 
                       help='Maximum number of generations for training (default: 100)')
    
    # Genetic Algorithm parameters
    parser.add_argument('--population', type=int, default=50, 
                       help='Population size (number of individuals) (default: 50)')
    parser.add_argument('--mutation-prob', type=float, default=0.01, 
                       help='Mutation probability (default: 0.01)')
    parser.add_argument('--crossover-prob', type=float, default=0.01, 
                       help='Crossover probability (default: 0.01)')
    parser.add_argument('--generation-interval', type=float, default=0.50, 
                       help='Generation replacement ratio (default: 0.50)')
    parser.add_argument('--render', action='store_true', 
                       help='Render the environment during testing')
    
    args = parser.parse_args()
    
    # If no arguments provided, default to training
    if not args.train and not args.test:
        args.train = True
    
    # Environment configuration
    if args.env == 'CartPole':
        env_name = 'CartPole-v1'
        model_prefix = 'cartpole'
        max_steps = 500
        success_threshold = 475  # Consider success if >= 475 steps for CartPole
    elif args.env == 'LunarLander':
        env_name = 'LunarLander-v3'
        model_prefix = 'lunarlander'
        max_steps = 1000
        success_threshold = 200  # LunarLander success is typically 200+ reward
    
    print(f"Using environment: {env_name}")
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)

    # Set seed for reproducibility
    SEED = 42
    np.random.seed(SEED)

    env = gym.make(env_name)
    env.reset(seed=SEED)

    number_of_inputs = env.observation_space.shape[0]
    
    # Configure network based on environment
    if args.env == 'CartPole':
        number_of_actions = env.action_space.n  # Discrete actions for CartPole
        size_of_network = [number_of_inputs, 4, 3, 1]  # Output 1 node for binary decision

    elif args.env == 'LunarLander':
        number_of_actions = env.action_space.n  # 4 discrete actions for LunarLander
        size_of_network = [number_of_inputs, 16, 12, number_of_actions]  # Output 4 nodes for 4 actions

    if args.train:
        print("=== TRAINING MODE ===")
        print(f"Maximum generations: {args.generations}")
        print(f"Population size: {args.population}")
        print(f"Mutation rate: {args.mutation_prob}")
        print(f"Crossover rate: {args.crossover_prob}")
        print(f"Generation interval: {args.generation_interval}")
        
        # Map size_of_network ([input, hidden..., output]) to hidden layers
        hidden_layers = size_of_network[1:-1]

        # Create PyTorch-based trainer and run training
        trainer = PyTorchGeneticTrainer(env_name, hidden_layers, model_prefix=model_prefix)

        best_network, best_fitness, n_generations = trainer.train(
            n_generations=args.generations,
            population_size=args.population,
            crossover_prob=args.crossover_prob,
            mutation_prob=args.mutation_prob
        )

        print(f"Training completed after {n_generations} generations. Best fitness: {best_fitness:.2f}")

        # Use evaluate function to test the trained agent
        print("\n=== EVALUATING TRAINED AGENT ===")
        # trainer.best_network is a CartPoleNet with get_action()
        agent = trainer.best_network
        
        # Create evaluation environment (no rendering)
        eval_env = gym.make(env_name)
        print("Testing trained agent without rendering:")
        avg_reward, std_reward = evaluate(agent, eval_env, n_episodes=10, render=False, env_type=args.env)
        
        # Determine success based on average performance
        if args.env == 'CartPole':
            success = avg_reward >= success_threshold
        elif args.env == 'LunarLander':
            success = avg_reward >= success_threshold
            
        if success:
            print(f'\nSUCCESS! Average reward {avg_reward:.2f} meets threshold {success_threshold}')
        else:
            print(f'\nNot quite there yet. Average reward {avg_reward:.2f}, threshold is {success_threshold}')

        eval_env.close()

    if args.test:
        print("=== TESTING MODE ===")

        eval_env = gym.make(env_name, render_mode='human' if args.render else None)

        # Load the saved PyTorch model using the trainer and evaluate
        hidden_layers = size_of_network[1:-1]
        trainer = PyTorchGeneticTrainer(env_name, hidden_layers, model_prefix=model_prefix)

        # Load PyTorch checkpoint (saved by PyTorchGeneticTrainer.save_model)
        model_path = f'checkpoints/{model_prefix}_best.pth'
        if os.path.exists(model_path):
            trainer.load_model(model_path)
            agent = trainer.best_network

            # First evaluate without rendering to get statistics
            print(f"Evaluating trained {args.env} agent (10 episodes)...")
            avg_reward, std_reward = evaluate(agent, eval_env, n_episodes=10, env_type=args.env)

        else:
            # Fall back to numpy weights file if PyTorch checkpoint not present
            weights_file = f'checkpoints/{model_prefix}_weights.npy'
            if os.path.exists(weights_file):
                saved_weights = load_model(weights_file)

                # Create a small CartPoleNet instance and set weights
                net = CartPoleNet(number_of_inputs, hidden_layers, size_of_network[-1])
                net.set_weights_from_vector(saved_weights)
                agent = net

                print(f"Evaluating trained {args.env} agent (10 episodes)...")
                avg_reward, std_reward = evaluate(agent, eval_env, n_episodes=10, env_type=args.env)

            else:
                print(f"No saved model found at {model_path} or {weights_file}! Please train first with --train --env {args.env}")
                return

        eval_env.close()


if __name__ == "__main__":
    main()
