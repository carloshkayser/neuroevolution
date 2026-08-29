"""Genetic algorithm trainer using PyMOO and PyTorch."""

import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.util.display.column import Column

try:
    import gymnasium_hardmaze  # Register HardMaze-v0
except ImportError:
    pass

try:
    import minigrid  # Register MiniGrid environments
except ImportError:
    pass

from .models import CartPoleNet


class CarRacingObsWrapper(gym.ObservationWrapper):
    """Observation wrapper for CarRacing to grayscale and downsample images."""

    def __init__(self, env, size: tuple[int, int] = (16, 16)):
        super().__init__(env)
        self.size = size
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(size[0] * size[1],), dtype=np.float32
        )

    def observation(self, obs):
        # Convert RGB to grayscale using standard luminance weights
        gray = np.dot(obs[..., :3], [0.2989, 0.5870, 0.1140]) / 255.0
        step_h = obs.shape[0] // self.size[0]
        step_w = obs.shape[1] // self.size[1]
        downsampled = gray[::step_h, ::step_w][: self.size[0], : self.size[1]]
        return downsampled.flatten().astype(np.float32)


def make_env(env_name: str, render_mode: str | None = None) -> gym.Env:
    """Create and wrap environment if needed (e.g. FlatObsWrapper for MiniGrid, CarRacingObsWrapper for CarRacing)."""
    try:
        import minigrid
    except ImportError:
        pass

    try:
        import gymnasium_hardmaze
    except ImportError:
        pass

    target_env = env_name
    if env_name in ('CarRacing', 'CarRacing-v0', 'CarRacing-v2'):
        try:
            env = gym.make(env_name, render_mode=render_mode) if render_mode is not None else gym.make(env_name)
        except Exception:
            target_env = 'CarRacing-v3'
            env = gym.make(target_env, render_mode=render_mode) if render_mode is not None else gym.make(target_env)
    else:
        if render_mode is not None:
            env = gym.make(target_env, render_mode=render_mode)
        else:
            env = gym.make(target_env)

    if isinstance(env.observation_space, gym.spaces.Dict) or 'MiniGrid' in target_env:
        from minigrid.wrappers import FlatObsWrapper
        env = FlatObsWrapper(env)
    elif 'CarRacing' in target_env:
        env = CarRacingObsWrapper(env)

    return env


class GymSingleObjectiveOutput(SingleObjectiveOutput):
    """PyMOO display output showing RL/Gym rewards per generation."""

    def __init__(self):
        super().__init__()
        self.reward_max = Column(name="reward_max", width=12)
        self.reward_avg = Column(name="reward_avg", width=12)
        self.reward_best = Column(name="reward_best", width=12)

    def initialize(self, algorithm):
        super().initialize(algorithm)
        self.columns += [self.reward_max, self.reward_avg, self.reward_best]

    def update(self, algorithm):
        super().update(algorithm)

        f, feas = algorithm.pop.get("f", "feas")
        if feas.sum() > 0:
            rewards = -f[feas]
            self.reward_max.set(f"{float(np.max(rewards)):.2f}")
            self.reward_avg.set(f"{float(np.mean(rewards)):.2f}")
        else:
            self.reward_max.set(None)
            self.reward_avg.set(None)

        if algorithm.opt is not None and len(algorithm.opt) > 0:
            opt = algorithm.opt[0]
            if opt.feas and opt.f is not None:
                self.reward_best.set(f"{-float(np.asarray(opt.f).item()):.2f}")
            else:
                self.reward_best.set(None)
        else:
            self.reward_best.set(None)


class GymOptimizationProblem(Problem):
    """PyMOO optimization problem for Gym environments."""
    
    def __init__(
        self,
        env_name: str,
        network_architecture: list[int] | tuple[int, ...],
        max_steps_per_eval: int = 500,
        n_episodes_per_eval: int = 1,
    ):
        self.env_name = env_name
        self.env = make_env(env_name)
        self.network_architecture = network_architecture
        self.max_steps_per_eval = max_steps_per_eval
        self.n_episodes_per_eval = n_episodes_per_eval
        
        # Create a sample network to get parameter count
        input_size = int(np.prod(self.env.observation_space.shape))
        if 'CartPole' in env_name:
            output_size = 1
            self.env_type = 'CartPole'
        elif 'MountainCar' in env_name:
            output_size = 1
            self.env_type = 'MountainCar'
        elif 'LunarLander' in env_name:
            output_size = 4
            self.env_type = 'LunarLander'
        elif 'HardMaze' in env_name or 'Maze' in env_name:
            output_size = 3
            self.env_type = 'Maze'
        elif 'CarRacing' in env_name:
            output_size = 3
            self.env_type = 'CarRacing'
        elif 'FourRooms' in env_name or 'MiniGrid' in env_name:
            output_size = getattr(self.env.action_space, 'n', 7)
            self.env_type = 'FourRooms'
        else:
            if hasattr(self.env.action_space, 'n'):
                output_size = self.env.action_space.n
            elif hasattr(self.env.action_space, 'shape'):
                output_size = self.env.action_space.shape[0]
            else:
                output_size = 1
            self.env_type = 'Unknown'
            
        sample_net = CartPoleNet(input_size, network_architecture, output_size)
        n_params = sample_net.get_total_params()
        
        # Problem bounds (weights typically in [-2, 2] range)
        super().__init__(n_var=n_params, n_obj=1, xl=-2.0, xu=2.0)
        
        # Store network info
        self.input_size = input_size
        self.output_size = output_size
        
    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate fitness for each individual in the population."""
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
    """Main trainer class using PyTorch and PyMOO."""
    
    def __init__(
        self,
        env_name: str,
        network_architecture: list[int] | tuple[int, ...],
        model_prefix: str = "pytorch_model",
        checkpoint_dir: str = "checkpoints",
        n_episodes_per_eval: int = 3,
        max_steps: int | None = None,
    ):
        self.env_name = env_name
        self.network_architecture = network_architecture
        self.model_prefix = model_prefix
        self.checkpoint_dir = checkpoint_dir
        self.env = make_env(env_name)
        
        # Environment-specific settings
        if 'CartPole' in env_name:
            self.env_type = 'CartPole'
            self.success_threshold = 475
            default_max_steps = 500
        elif 'MountainCar' in env_name:
            self.env_type = 'MountainCar'
            self.success_threshold = -110
            default_max_steps = 200
        elif 'LunarLander' in env_name:
            self.env_type = 'LunarLander'
            self.success_threshold = 200
            default_max_steps = 1000
        elif 'HardMaze' in env_name or 'Maze' in env_name:
            self.env_type = 'Maze'
            self.success_threshold = 1.0
            default_max_steps = 500
        elif 'CarRacing' in env_name:
            self.env_type = 'CarRacing'
            self.success_threshold = 900
            default_max_steps = 1000
        elif 'FourRooms' in env_name or 'MiniGrid' in env_name:
            self.env_type = 'FourRooms'
            self.success_threshold = 0.5
            default_max_steps = 100
        else:
            self.env_type = 'Unknown'
            self.success_threshold = 0
            default_max_steps = 500
            
        steps = max_steps if max_steps is not None else default_max_steps
        self.problem = GymOptimizationProblem(
            env_name,
            network_architecture,
            max_steps_per_eval=steps,
            n_episodes_per_eval=n_episodes_per_eval,
        )
        
        # Store best network
        self.best_network = None
        self.best_fitness = float('-inf')
        self.fitness_history = []
        
    def train(
        self,
        n_generations: int = 100,
        population_size: int = 50,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        output: SingleObjectiveOutput | None = None,
    ):
        """Train using PyMOO genetic algorithm."""
        print(f"Training {self.env_type} agent with PyTorch + PyMOO")
        print(f"Generations: {n_generations}, Population: {population_size}")
        print(f"Network architecture: {self.problem.input_size} -> {self.network_architecture} -> {self.problem.output_size}")
        
        if output is None:
            output = GymSingleObjectiveOutput()

        # Configure genetic algorithm
        algorithm = GA(
            pop_size=population_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=crossover_prob, eta=15),
            mutation=PM(prob=mutation_prob, eta=20),
            eliminate_duplicates=True,
            output=output,
        )
        
        # Run optimization
        result = minimize(
            self.problem,
            algorithm,
            ('n_gen', n_generations),
            verbose=True,
            save_history=True,
        )
        
        # Extract best solution
        best_weights = result.X
        best_fitness = -result.F.item()  # Convert back from minimization
        
        # Create and save best network
        self.best_network = CartPoleNet(
            self.problem.input_size, 
            self.network_architecture, 
            self.problem.output_size,
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
    
    def save_model(self) -> None:
        """Save the best model and weights."""
        if self.best_network is not None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
            # Save PyTorch model
            pth_path = os.path.join(self.checkpoint_dir, f"{self.model_prefix}_best.pth")
            torch.save({
                'network_state_dict': self.best_network.state_dict(),
                'network_architecture': self.network_architecture,
                'input_size': self.problem.input_size,
                'output_size': self.problem.output_size,
                'env_type': self.env_type,
                'fitness': self.best_fitness,
            }, pth_path)
            
            # Also save weights as numpy array for compatibility
            npy_path = os.path.join(self.checkpoint_dir, f"{self.model_prefix}_weights.npy")
            weights_vector = self.best_network.get_weights_as_vector().numpy()
            np.save(npy_path, weights_vector)
            
            print(f"Model saved:")
            print(f"- PyTorch model: {pth_path}")
            print(f"- Weights vector: {npy_path}")
    
    def load_model(self, model_path: str) -> CartPoleNet:
        """Load a saved PyTorch model."""
        checkpoint = torch.load(model_path, weights_only=False)
        
        self.best_network = CartPoleNet(
            checkpoint['input_size'],
            checkpoint['network_architecture'],
            checkpoint['output_size'],
        )
        self.best_network.load_state_dict(checkpoint['network_state_dict'])
        self.env_type = checkpoint['env_type']
        self.best_fitness = checkpoint['fitness']
        
        return self.best_network
    
    def plot_learning_curve(self) -> str | None:
        """Generate and save learning curve plot."""
        if not self.fitness_history:
            return None
            
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        plt.figure(figsize=(12, 8))
        plt.plot(self.fitness_history, linewidth=2, color='blue', marker='o', markersize=4)
        plt.title(f'PyTorch + PyMOO Learning Curve - {self.env_type}', fontsize=14, fontweight='bold')
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Best Fitness Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add algorithm and environment info
        info_text = (
            f'Algorithm: PyMOO Genetic Algorithm\n'
            f'Framework: PyTorch\n'
            f'Environment: {self.env_name}\n'
            f'Generations: {len(self.fitness_history)}\n'
            f'Best Fitness: {self.best_fitness:.2f}'
        )
        plt.text(
            0.02, 0.98, info_text, 
            transform=plt.gca().transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
        )
        
        # Save the plot
        plot_filename = os.path.join(self.checkpoint_dir, f"{self.model_prefix}_learning_curve.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"- Learning curve: {plot_filename}")
        return plot_filename
    
    def evaluate(self, n_episodes: int = 20, render: bool = False, max_steps: int | None = None) -> tuple[float, float] | tuple[None, None]:
        """Evaluate the trained agent."""
        if self.best_network is None:
            print("No trained model available!")
            return None, None
            
        if max_steps is None:
            max_steps = getattr(self.problem, 'max_steps_per_eval', 1000)
            
        rewards = []
        eval_env = make_env(self.env_name, render_mode='human' if render else None)
        
        for episode in range(n_episodes):
            obs, _ = eval_env.reset(seed=42 + episode)
            total_reward = 0
            steps = 0
            
            while steps < max_steps:
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
        
        avg_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        
        if not render:
            print(f"Evaluation over {n_episodes} episodes:")
            print(f"Average reward: {avg_reward:.2f}")
            print(f"Standard deviation: {std_reward:.2f}")
            
            if self.env_type == 'CartPole':
                success = avg_reward >= self.success_threshold
            elif self.env_type == 'MountainCar':
                success = avg_reward >= self.success_threshold
            elif self.env_type == 'LunarLander':
                success = avg_reward >= self.success_threshold
            elif self.env_type == 'Maze':
                success = avg_reward >= self.success_threshold
            elif self.env_type == 'CarRacing':
                success = avg_reward >= self.success_threshold
            elif self.env_type in ('FourRooms', 'MiniGrid'):
                success = avg_reward >= self.success_threshold
            else:
                success = False
                
            if success:
                print(f"SUCCESS! Average reward {avg_reward:.2f} meets threshold {self.success_threshold}")
            else:
                print(f"Not quite there yet. Average reward {avg_reward:.2f}, threshold is {self.success_threshold}")
        
        return avg_reward, std_reward

    def record_gif(
        self,
        output_path: str = "assets/agent.gif",
        max_steps: int | None = None,
        fps: int = 30,
        seed: int = 42,
    ) -> str | None:
        """Record an episode of the trained agent and save as an animated GIF."""
        if self.best_network is None:
            print("No trained model available to record!")
            return None

        from PIL import Image

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        if max_steps is None:
            max_steps = getattr(self.problem, 'max_steps_per_eval', 1000)

        eval_env = make_env(self.env_name, render_mode='rgb_array')
        obs, _ = eval_env.reset(seed=seed)
        frames = []

        initial_frame = eval_env.render()
        if initial_frame is not None:
            frames.append(Image.fromarray(initial_frame))

        total_reward = 0
        steps = 0
        done, truncated = False, False

        while not (done or truncated) and steps < max_steps:
            action = self.best_network.get_action(obs, self.env_type)
            obs, reward, done, truncated, _ = eval_env.step(action)
            total_reward += reward
            steps += 1
            frame = eval_env.render()
            if frame is not None:
                frames.append(Image.fromarray(frame))

        eval_env.close()

        if frames:
            duration = int(1000 / fps)
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,
                optimize=True,
            )
            print(f"Recorded GIF ({len(frames)} frames, reward={total_reward:.2f}) -> {output_path}")
            return output_path
        return None

