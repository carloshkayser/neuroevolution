from ann import NeuralNetwork
from ga import GeneticAlgorithm

import gymnasium as gym
import numpy as np
import os
import argparse

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
    def __init__(self, network, weights):
        self.network = network
        self.weights = weights
    
    def predict(self, obs):
        """Predict action using the neural network"""
        return self.network.feedforward(obs, self.weights)

def evaluate(agent, env, n_episodes=20, render=False):
    """Evaluate the agent performance over multiple episodes"""
    rewards = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done, truncated = False, False
        total_reward = 0
        
        while not (done or truncated):
            if render:
                env.render()

            action = agent.predict(obs)  # use trained model
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
    parser.add_argument('--env', choices=['CartPole', 'MountainCar'], default='CartPole', 
                       help='Environment to use: CartPole (CartPole-v1) or MountainCar (MountainCar-v0)')
    parser.add_argument('--generations', type=int, default=100, 
                       help='Maximum number of generations for training (default: 100)')
    
    # Genetic Algorithm parameters
    parser.add_argument('--population', type=int, default=50, 
                       help='Population size (number of individuals) (default: 50)')
    parser.add_argument('--mutation-rate', type=float, default=0.01, 
                       help='Mutation chance/rate (default: 0.01)')
    parser.add_argument('--crossover-rate', type=float, default=0.01, 
                       help='Crossover chance/rate (default: 0.01)')
    parser.add_argument('--generation-interval', type=float, default=0.50, 
                       help='Generation replacement ratio (default: 0.50)')
    
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
    elif args.env == 'MountainCar':
        env_name = 'MountainCar-v0'
        model_prefix = 'mountaincar'
        max_steps = 200
        success_threshold = -110  # MountainCar has negative rewards, success is reaching the goal
    
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

    elif args.env == 'MountainCar':
        number_of_actions = env.action_space.n  # Discrete actions for MountainCar
        size_of_network = [number_of_inputs, 8, 6, number_of_actions]  # Output 3 nodes for 3 actions

    if args.train:
        print("=== TRAINING MODE ===")
        print(f"Maximum generations: {args.generations}")
        print(f"Population size: {args.population}")
        print(f"Mutation rate: {args.mutation_rate}")
        print(f"Crossover rate: {args.crossover_rate}")
        print(f"Generation interval: {args.generation_interval}")
        
        genetic = GeneticAlgorithm(size_of_network, args.population, args.mutation_rate, 
                                 args.crossover_rate, generation_interval=args.generation_interval)
        network = NeuralNetwork(size_of_network, env = env, genetics = genetic, model_prefix = model_prefix)

        the_best, _, n_episodes = network.train(max_generations=args.generations)

        print(f"Training completed after {args.generations} generations and {n_episodes} episodes.")
        
        # Use evaluate function to test the trained agent
        print("\n=== EVALUATING TRAINED AGENT ===")
        agent = NeuralNetworkAgent(network, the_best.weights)
        
        # Create evaluation environment (no rendering)
        eval_env = gym.make(env_name)
        print("Testing trained agent without rendering:")
        avg_reward, std_reward = evaluate(agent, eval_env, n_episodes=10, render=False)
        
        # Determine success based on average performance
        if args.env == 'CartPole':
            success = avg_reward >= success_threshold
        elif args.env == 'MountainCar':
            success = avg_reward >= success_threshold
            
        if success:
            print(f'\nSUCCESS! Average reward {avg_reward:.2f} meets threshold {success_threshold}')
        else:
            print(f'\nNot quite there yet. Average reward {avg_reward:.2f}, threshold is {success_threshold}')

        eval_env.close()

    if args.test:
        print("=== TESTING MODE ===")
        
        # Load the best saved model
        model_file = f'checkpoints/{model_prefix}_best.npy'
        saved_weights = load_model(model_file)
        if saved_weights is None:
            print(f"No saved model found at {model_file}! Please train first with --train --env {args.env}")
            return
        
        # Create network for evaluation (needs env parameter)
        network = NeuralNetwork(size_of_network, env=env, model_prefix=model_prefix)
        agent = NeuralNetworkAgent(network, saved_weights)
        
        # First evaluate without rendering to get statistics
        print(f"Evaluating trained {args.env} agent (10 episodes, no rendering)...")
        eval_env = gym.make(env_name)
        avg_reward, std_reward = evaluate(agent, eval_env, n_episodes=10, render=False)
        eval_env.close()
        
        # Then run with rendering for visual confirmation
        print(f"\nRunning 5 episodes with visual rendering:")
        render_env = gym.make(env_name, render_mode='human')
        evaluate(agent, render_env, n_episodes=5, render=True)
        render_env.close()

    env.close()

if __name__ == "__main__":
    main()
