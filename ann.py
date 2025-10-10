"""PyTorch Neural Network with PyMOO Genetic Algorithm"""

from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


class CartPoleNet(nn.Module):
    """PyTorch Neural Network for CartPole and MountainCar environments"""
    
    def __init__(self, input_size, hidden_sizes, output_size):
        super(CartPoleNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Store architecture info
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
    def forward(self, x):
        return self.network(x)
    
    def get_action(self, state, env_type='CartPole'):
        """Get action based on network output and environment type"""
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).unsqueeze(0)
            
            output = self.forward(state)
            
            if env_type == 'CartPole':
                # Binary decision: left (0) or right (1)
                action = 1 if output.item() > 0.0 else 0
            elif env_type == 'MountainCar':
                # Three actions: left (0), no action (1), right (2)
                if self.output_size == 3:
                    action = torch.argmax(output, dim=1).item()
                else:
                    # Single output mapped to 3 actions
                    output_val = output.item()
                    if output_val < -0.33:
                        action = 0  # Push left
                    elif output_val > 0.33:
                        action = 2  # Push right
                    else:
                        action = 1  # No push
            elif env_type == 'LunarLander':
                # Four actions: do nothing (0), fire left (1), fire main (2), fire right (3)
                action = torch.argmax(output, dim=1).item()
            else:
                # Default behavior
                if self.output_size > 1:
                    action = torch.argmax(output, dim=1).item()
                else:
                    action = 1 if output.item() > 0.0 else 0
                    
            return action
    
    def get_weights_as_vector(self):
        """Get all network parameters as a single vector"""
        return torch.cat([param.data.view(-1) for param in self.parameters()])
    
    def set_weights_from_vector(self, weights_vector):
        """Set network parameters from a vector"""
        if isinstance(weights_vector, np.ndarray):
            weights_vector = torch.FloatTensor(weights_vector)
            
        idx = 0
        for param in self.parameters():
            param_size = param.numel()
            param.data = weights_vector[idx:idx + param_size].view(param.shape)
            idx += param_size
    
    def get_total_params(self):
        """Get total number of parameters in the network"""
        return sum(p.numel() for p in self.parameters())


class GymOptimizationProblem(Problem):
    """PyMOO optimization problem for Gym environments"""
    
    def __init__(self, env_name, network_architecture, max_steps_per_eval=500, n_episodes_per_eval=1):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.network_architecture = network_architecture
        self.max_steps_per_eval = max_steps_per_eval
        self.n_episodes_per_eval = n_episodes_per_eval
        
        # Create a sample network to get parameter count
        input_size = self.env.observation_space.shape[0]
        if 'CartPole' in env_name:
            output_size = 1
            self.env_type = 'CartPole'
        elif 'MountainCar' in env_name:
            output_size = 3
            self.env_type = 'MountainCar'
        elif 'LunarLander' in env_name:
            output_size = 4
            self.env_type = 'LunarLander'
        else:
            output_size = self.env.action_space.n
            self.env_type = 'Unknown'
            
        sample_net = CartPoleNet(input_size, network_architecture, output_size)
        n_params = sample_net.get_total_params()
        
        # Problem bounds (weights typically in [-2, 2] range)
        super().__init__(n_var=n_params, n_obj=1, xl=-2.0, xu=2.0)
        
        # Store network info
        self.input_size = input_size
        self.output_size = output_size
        
    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate fitness for each individual in the population"""
        fitness_scores = []
        
        for weights in X:
            # Create network and set weights
            network = CartPoleNet(self.input_size, self.network_architecture, self.output_size)
            network.set_weights_from_vector(weights)
            
            # Evaluate network
            total_reward = 0
            for episode in range(self.n_episodes_per_eval):
                obs, _ = self.env.reset(seed=42 + episode)
                episode_reward = 0
                steps = 0
                
                while steps < self.max_steps_per_eval:
                    action = network.get_action(obs, self.env_type)
                    obs, reward, done, truncated, _ = self.env.step(action)
                    episode_reward += reward
                    steps += 1
                    
                    if done or truncated:
                        break
                
                total_reward += episode_reward
            
            # Average reward across episodes
            avg_reward = total_reward / self.n_episodes_per_eval
            fitness_scores.append(-avg_reward)  # Negative because PyMOO minimizes
        
        out["F"] = np.array(fitness_scores).reshape(-1, 1)


class PyTorchGeneticTrainer:
    """Main trainer class using PyTorch and PyMOO"""
    
    def __init__(self, env_name, network_architecture, model_prefix="pytorch_model"):
        self.env_name = env_name
        self.network_architecture = network_architecture
        self.model_prefix = model_prefix
        self.env = gym.make(env_name)
        
        # Environment-specific settings
        if 'CartPole' in env_name:
            self.env_type = 'CartPole'
            self.success_threshold = 475
        elif 'MountainCar' in env_name:
            self.env_type = 'MountainCar'
            self.success_threshold = -110
        elif 'LunarLander' in env_name:
            self.env_type = 'LunarLander'
            self.success_threshold = 200
        else:
            self.env_type = 'Unknown'
            self.success_threshold = 0
            
        # Create the optimization problem
        if 'CartPole' in env_name:
            max_steps = 500
        elif 'LunarLander' in env_name:
            max_steps = 1000
        else:
            max_steps = 500
        self.problem = GymOptimizationProblem(env_name, network_architecture, max_steps_per_eval=max_steps)
        
        # Store best network
        self.best_network = None
        self.best_fitness = float('-inf')
        self.fitness_history = []
        
    def train(self, n_generations=100, population_size=50, crossover_prob=0.9, mutation_prob=0.1):
        """Train using PyMOO genetic algorithm"""
        
        print(f"Training {self.env_type} agent with PyTorch + PyMOO")
        print(f"Generations: {n_generations}, Population: {population_size}")
        print(f"Network architecture: {self.problem.input_size} -> {self.network_architecture} -> {self.problem.output_size}")
        
        # Configure genetic algorithm
        algorithm = GA(
            pop_size=population_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=crossover_prob, eta=15),
            mutation=PM(prob=mutation_prob, eta=20),
            eliminate_duplicates=True
        )
        
        # Run optimization
        result = minimize(
            self.problem,
            algorithm,
            ('n_gen', n_generations),
            verbose=True,
            save_history=True
        )
        
        # Extract best solution
        best_weights = result.X
        best_fitness = -result.F.item()  # Convert back from minimization
        
        # Create and save best network
        self.best_network = CartPoleNet(
            self.problem.input_size, 
            self.network_architecture, 
            self.problem.output_size
        )
        self.best_network.set_weights_from_vector(best_weights)
        self.best_fitness = best_fitness
        
        # Extract fitness history
        self.fitness_history = [-gen.opt.get("F").item() for gen in result.history]
        
        # Save models
        self.save_model()
        
        # Generate learning curve
        self.plot_learning_curve()
        
        print(f"\nTraining completed!")
        print(f"Best fitness: {best_fitness:.2f}")
        print(f"Success threshold: {self.success_threshold}")
        
        return self.best_network, best_fitness, len(self.fitness_history)
    
    def save_model(self):
        """Save the best model"""
        if self.best_network is not None:
            # Save PyTorch model
            torch.save({
                'network_state_dict': self.best_network.state_dict(),
                'network_architecture': self.network_architecture,
                'input_size': self.problem.input_size,
                'output_size': self.problem.output_size,
                'env_type': self.env_type,
                'fitness': self.best_fitness
            }, f'checkpoints/{self.model_prefix}_best.pth')
            
            # Also save weights as numpy array for compatibility
            weights_vector = self.best_network.get_weights_as_vector().numpy()
            np.save(f'checkpoints/{self.model_prefix}_weights.npy', weights_vector)
            
            print(f"Model saved:")
            print(f"- PyTorch model: checkpoints/{self.model_prefix}_best.pth")
            print(f"- Weights vector: checkpoints/{self.model_prefix}_weights.npy")
    
    def load_model(self, model_path):
        """Load a saved PyTorch model"""
        checkpoint = torch.load(model_path)
        
        self.best_network = CartPoleNet(
            checkpoint['input_size'],
            checkpoint['network_architecture'],
            checkpoint['output_size']
        )
        self.best_network.load_state_dict(checkpoint['network_state_dict'])
        self.env_type = checkpoint['env_type']
        self.best_fitness = checkpoint['fitness']
        
        return self.best_network
    
    def plot_learning_curve(self):
        """Generate and save learning curve plot"""
        if not self.fitness_history:
            return
            
        plt.figure(figsize=(12, 8))
        plt.plot(self.fitness_history, linewidth=2, color='blue', marker='o', markersize=4)
        plt.title(f'PyTorch + PyMOO Learning Curve - {self.env_type}', fontsize=14, fontweight='bold')
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Best Fitness Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add algorithm and environment info
        info_text = f'Algorithm: PyMOO Genetic Algorithm\nFramework: PyTorch\nEnvironment: {self.env_name}\nGenerations: {len(self.fitness_history)}\nBest Fitness: {self.best_fitness:.2f}'
        plt.text(0.02, 0.98, info_text, 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Save the plot
        plot_filename = f"checkpoints/{self.model_prefix}_learning_curve.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"- Learning curve: {plot_filename}")
        
        return plot_filename
    
    def evaluate(self, n_episodes=20, render=False):
        """Evaluate the trained agent"""
        if self.best_network is None:
            print("No trained model available!")
            return None, None
            
        rewards = []
        
        # Create evaluation environment
        eval_env = gym.make(self.env_name, render_mode='human' if render else None)
        
        for episode in range(n_episodes):
            obs, _ = eval_env.reset(seed=42 + episode)
            total_reward = 0
            steps = 0
            
            while True:
                action = self.best_network.get_action(obs, self.env_type)
                obs, reward, done, truncated, _ = eval_env.step(action)
                total_reward += reward
                steps += 1
                
                if done or truncated:
                    break
            
            rewards.append(total_reward)
            if render:
                print(f"Episode {episode + 1}: {steps} steps, Reward: {total_reward}")
        
        eval_env.close()
        
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        if not render:
            print(f"Evaluation over {n_episodes} episodes:")
            print(f"Average reward: {avg_reward:.2f}")
            print(f"Standard deviation: {std_reward:.2f}")
            
            # Check success
            if self.env_type == 'CartPole':
                success = avg_reward >= self.success_threshold
            elif self.env_type == 'MountainCar':
                success = avg_reward >= self.success_threshold
            else:
                success = False
                
            if success:
                print(f"SUCCESS! Average reward {avg_reward:.2f} meets threshold {self.success_threshold}")
            else:
                print(f"Not quite there yet. Average reward {avg_reward:.2f}, threshold is {self.success_threshold}")
        
        return avg_reward, std_reward
