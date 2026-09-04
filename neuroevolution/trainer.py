"""Genetic algorithm trainer using PyMOO and PyTorch."""

import os
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.util.display.column import Column


class WarmStartSampling(Sampling):
    """Initial population sampling seeded from existing weights plus perturbations."""
    def __init__(self, seed_weights: np.ndarray):
        super().__init__()
        self.seed_weights = seed_weights

    def _do(self, problem, n_samples, **kwargs):
        X = np.empty((n_samples, problem.n_var))
        X[0] = self.seed_weights.copy()
        n_close = min(n_samples // 3, 15)
        n_med = min(n_samples // 3, 15)
        for i in range(1, n_close):
            noise = np.random.randn(problem.n_var) * 0.03
            X[i] = np.clip(self.seed_weights + noise, problem.xl, problem.xu)
        for i in range(n_close, n_close + n_med):
            noise = np.random.randn(problem.n_var) * 0.08
            X[i] = np.clip(self.seed_weights + noise, problem.xl, problem.xu)
        for i in range(n_close + n_med, n_samples):
            X[i] = np.random.uniform(problem.xl, problem.xu, problem.n_var)
        return X

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
    """Observation wrapper for CarRacing with ray sensors or legacy downsample."""

    def __init__(self, env, mode: str = "sensors", size: tuple[int, int] = (16, 16)):
        super().__init__(env)
        self.mode = mode
        self.size = size
        if mode == "sensors":
            # 11 rays + 3 lateral offsets (car, mid, far) + speed + angular velocity = 16 features
            self.observation_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(16,), dtype=np.float32
            )
            self.angles = np.radians([-75, -55, -35, -20, -10, 0, 10, 20, 35, 55, 75])
            self.max_dist = 60.0
        else:
            self.observation_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(size[0] * size[1],), dtype=np.float32
            )

    def observation(self, obs):
        if self.mode != "sensors":
            # Legacy downsampled grayscale
            gray = np.dot(obs[..., :3], [0.2989, 0.5870, 0.1140]) / 255.0
            step_h = obs.shape[0] // self.size[0]
            step_w = obs.shape[1] // self.size[1]
            downsampled = gray[::step_h, ::step_w][: self.size[0], : self.size[1]]
            return downsampled.flatten().astype(np.float32)

        x0, y0 = 48.0, 66.0
        # Road is grey/curb, grass is green: G - R > 15
        is_grass = (obs[:84, :, 1].astype(np.int16) - obs[:84, :, 0].astype(np.int16)) > 15

        # Ray casts
        dists = []
        for angle in self.angles:
            dx = np.sin(angle)
            dy = -np.cos(angle)
            d_val = self.max_dist
            for d in range(2, int(self.max_dist)):
                xi = int(round(x0 + d * dx))
                yi = int(round(y0 + d * dy))
                if xi < 0 or xi >= 96 or yi < 0 or yi >= 84 or is_grass[yi, xi]:
                    d_val = d
                    break
            # Normalize to [-1, 1]
            dists.append((d_val / self.max_dist) * 2.0 - 1.0)

        # Lateral offsets at row 66 (car), 46 (mid), 26 (far)
        offsets = []
        for row in (66, 46, 26):
            road_cols = np.where(~is_grass[row, :])[0]
            if len(road_cols) > 0:
                center_x = (road_cols[0] + road_cols[-1]) / 2.0
                off = (center_x - 48.0) / 30.0
            else:
                off = 0.0
            offsets.append(float(np.clip(off, -1.0, 1.0)))

        # Vehicle dynamics
        car = getattr(self.unwrapped, 'car', None)
        if car is not None:
            v = car.hull.linearVelocity
            speed = (np.sqrt(v.x**2 + v.y**2) / 40.0) * 2.0 - 1.0
            ang_vel = np.clip(car.hull.angularVelocity / 5.0, -1.0, 1.0)
        else:
            speed = -1.0
            ang_vel = 0.0

        return np.array(dists + offsets + [speed, ang_vel], dtype=np.float32)


def make_env(env_name: str, render_mode: str | None = None, obs_mode: str = "sensors") -> gym.Env:
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
        env = CarRacingObsWrapper(env, mode=obs_mode)

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


class GenerationPlotCallback(Callback):
    """Callback to record population statistics and update the learning curve plot at each generation."""

    def __init__(self, trainer: "PyTorchGeneticTrainer", live_plot: bool = True):
        super().__init__()
        self.trainer = trainer
        self.live_plot = live_plot

    def notify(self, algorithm):
        f = algorithm.pop.get("F")
        if f is not None and len(f) > 0:
            rewards = -f.flatten()
            gen_best = float(np.max(rewards))
            gen_mean = float(np.mean(rewards))
            gen_std = float(np.std(rewards))
            gen_min = float(np.min(rewards))
        else:
            gen_best = gen_mean = gen_std = gen_min = 0.0

        if algorithm.opt is not None and len(algorithm.opt) > 0:
            opt = algorithm.opt[0]
            if opt.f is not None:
                best_so_far = -float(np.asarray(opt.f).item())
            else:
                best_so_far = gen_best
        else:
            best_so_far = gen_best

        n_eval = int(algorithm.evaluator.n_eval)
        n_episodes = n_eval * self.trainer.problem.n_episodes_per_eval

        self.trainer.history_generations.append(algorithm.n_gen)
        self.trainer.fitness_history.append(best_so_far)
        self.trainer.history_best.append(best_so_far)
        self.trainer.history_gen_best.append(gen_best)
        self.trainer.history_mean.append(gen_mean)
        self.trainer.history_std.append(gen_std)
        self.trainer.history_min.append(gen_min)
        self.trainer.history_evals.append(n_eval)
        self.trainer.history_episodes.append(n_episodes)

        if self.live_plot:
            self.trainer.plot_learning_curve(verbose=False)


def _evaluate_individual_worker(args):
    """Worker function to evaluate a single candidate individual in parallel."""
    (
        weights,
        env_name,
        network_architecture,
        input_size,
        output_size,
        env_type,
        max_steps,
        n_episodes,
    ) = args

    env = make_env(env_name)
    network = CartPoleNet(input_size, network_architecture, output_size)
    network.set_weights_from_vector(weights)

    total_reward = 0.0
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=42 + episode)
        episode_reward = 0.0
        steps = 0
        tiles_visited = 0
        steps_without_tile = 0

        while steps < max_steps:
            action = network.get_action(obs, env_type)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1

            if env_type == 'CarRacing':
                cur_tiles = getattr(env.unwrapped, 'tile_visited_count', 0)
                if cur_tiles > tiles_visited:
                    tiles_visited = cur_tiles
                    steps_without_tile = 0
                else:
                    steps_without_tile += 1

                # Early stopping if stuck or off-track
                if steps > 50 and steps_without_tile > 40:
                    episode_reward -= 30.0
                    break

            if done or truncated:
                break

        total_reward += episode_reward

    env.close()
    avg_reward = total_reward / n_episodes
    return -avg_reward


class GymOptimizationProblem(Problem):
    """PyMOO optimization problem for Gym environments with parallel evaluation."""
    
    def __init__(
        self,
        env_name: str,
        network_architecture: list[int] | tuple[int, ...],
        max_steps_per_eval: int = 500,
        n_episodes_per_eval: int = 1,
        n_workers: int | None = None,
    ):
        self.env_name = env_name
        self.env = make_env(env_name)
        self.network_architecture = network_architecture
        self.max_steps_per_eval = max_steps_per_eval
        self.n_episodes_per_eval = n_episodes_per_eval
        
        if n_workers is None:
            n_workers = min(os.cpu_count() or 4, 8)
        self.n_workers = max(1, n_workers)
        self.pool = None
        
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

    def _get_pool(self):
        if self.n_workers > 1 and self.pool is None:
            import multiprocessing as mp
            self.pool = mp.Pool(processes=self.n_workers)
        return self.pool

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["pool"] = None
        state["env"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.pool = None
        if getattr(self, "env", None) is None:
            self.env = make_env(self.env_name)

    def __deepcopy__(self, memo):
        import copy
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ("pool", "env"):
                setattr(result, k, None)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate fitness for each individual in the population."""
        tasks = [
            (
                weights,
                self.env_name,
                self.network_architecture,
                self.input_size,
                self.output_size,
                self.env_type,
                self.max_steps_per_eval,
                self.n_episodes_per_eval,
            )
            for weights in X
        ]

        if self.n_workers > 1:
            pool = self._get_pool()
            fitness_scores = pool.map(_evaluate_individual_worker, tasks)
        else:
            fitness_scores = [_evaluate_individual_worker(t) for t in tasks]

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
        n_workers: int | None = None,
        job: Any | None = None,
    ):
        self.env_name = env_name
        self.network_architecture = network_architecture
        self.model_prefix = model_prefix
        self.checkpoint_dir = checkpoint_dir
        self.env = make_env(env_name)
        self.n_workers = n_workers
        self.job = job
        
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
            n_workers=n_workers,
        )
        
        # Store best network
        self.best_network = None
        self.best_fitness = float('-inf')
        self.current_pop_size = 50
        self.fitness_history = []
        self.history_generations = []
        self.history_best = []
        self.history_gen_best = []
        self.history_mean = []
        self.history_std = []
        self.history_min = []
        self.history_evals = []
        self.history_episodes = []
        
    def train(
        self,
        n_generations: int = 100,
        population_size: int = 50,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        output: SingleObjectiveOutput | None = None,
        callback: Callback | None = None,
        live_plot: bool = True,
        show_plot: bool = False,
        resume: bool = False,
        initial_weights: np.ndarray | None = None,
    ):
        """Train using PyMOO genetic algorithm."""
        print(f"Training {self.env_type} agent with PyTorch + PyMOO")
        print(f"Generations: {n_generations}, Population: {population_size}")
        print(f"Network architecture: {self.problem.input_size} -> {self.network_architecture} -> {self.problem.output_size}")
        
        self.current_pop_size = population_size
        self.fitness_history = []
        self.history_generations = []
        self.history_best = []
        self.history_gen_best = []
        self.history_mean = []
        self.history_std = []
        self.history_min = []
        self.history_evals = []
        self.history_episodes = []

        if output is None:
            output = GymSingleObjectiveOutput()

        plot_cb = GenerationPlotCallback(self, live_plot=live_plot)
        if callback is not None:
            class CompositeCallback(Callback):
                def __init__(self, *cbs):
                    super().__init__()
                    self.cbs = cbs
                def notify(self, alg):
                    for c in self.cbs:
                        c.notify(alg)
            active_callback = CompositeCallback(plot_cb, callback)
        else:
            active_callback = plot_cb

        # Check for warm-start / resume
        sampling = FloatRandomSampling()
        if resume and initial_weights is None:
            weights_file = os.path.join(self.checkpoint_dir, f"{self.model_prefix}_weights.npy")
            model_file = os.path.join(self.checkpoint_dir, f"{self.model_prefix}_best.pth")
            if os.path.exists(weights_file):
                initial_weights = np.load(weights_file)
                print(f"Resuming training from {weights_file}")
            elif os.path.exists(model_file):
                net = self.load_model(model_file)
                initial_weights = net.get_weights_as_vector().detach().cpu().numpy()
                print(f"Resuming training from {model_file}")

        if initial_weights is not None:
            sampling = WarmStartSampling(initial_weights)
            print(f"Seeded initial population with provided weights ({len(initial_weights)} parameters)")

        # Configure genetic algorithm
        algorithm = GA(
            pop_size=population_size,
            sampling=sampling,
            crossover=SBX(prob=crossover_prob, eta=15),
            mutation=PM(prob=mutation_prob, eta=20),
            eliminate_duplicates=True,
            output=output,
        )
        
        # Run optimization
        try:
            result = minimize(
                self.problem,
                algorithm,
                ('n_gen', n_generations),
                callback=active_callback,
                verbose=True,
                save_history=True,
            )
        finally:
            self.problem.close()
        
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
        
        # Fallback if callback did not populate fitness_history
        if not self.fitness_history and result.history:
            self.fitness_history = [-gen.opt.get("F").item() for gen in result.history]
            self.history_generations = list(range(1, len(self.fitness_history) + 1))
            self.history_best = list(self.fitness_history)
        
        # Save models
        self.save_model()
        
        # Generate final learning curve plot
        self.plot_learning_curve(verbose=True, show=show_plot)
        
        # Save metrics to attached job if present
        if self.job is not None and hasattr(self.job, 'save_metrics'):
            self.job.save_metrics({
                "best_fitness": float(best_fitness),
                "success_threshold": self.success_threshold,
                "success": bool(best_fitness >= self.success_threshold),
                "generations_completed": len(self.fitness_history),
                "history_best": [float(x) for x in self.history_best],
                "history_mean": [float(x) for x in self.history_mean] if self.history_mean else [],
                "total_episodes": int(self.history_episodes[-1]) if self.history_episodes else 0,
            })

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
    
    def plot_learning_curve(self, verbose: bool = False, show: bool = False) -> str | None:
        """Generate and save learning curve plot with generations and cumulative episodes."""
        if not self.history_generations and not self.fitness_history:
            return None

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        gens = np.array(
            self.history_generations
            if self.history_generations
            else list(range(1, len(self.fitness_history) + 1))
        )
        bests = np.array(self.history_best if self.history_best else self.fitness_history)
        means = np.array(self.history_mean) if self.history_mean else None
        stds = np.array(self.history_std) if self.history_std else None

        fig, ax1 = plt.subplots(figsize=(11, 6.5))

        # Best fitness curve
        ax1.plot(
            gens,
            bests,
            linewidth=2.5,
            color='#1f77b4',
            marker='o' if len(gens) <= 40 else None,
            markersize=4,
            label='Best Fitness (All-Time)',
        )

        # Population mean curve & std fill
        if means is not None and len(means) == len(gens):
            ax1.plot(
                gens,
                means,
                linewidth=2.0,
                color='#ff7f0e',
                linestyle='--',
                marker='s' if len(gens) <= 40 else None,
                markersize=3,
                label='Population Mean',
            )
            if stds is not None and len(stds) == len(gens):
                ax1.fill_between(
                    gens,
                    means - stds,
                    means + stds,
                    color='#ff7f0e',
                    alpha=0.2,
                    label='Pop Mean ± 1 Std',
                )

        # Target threshold
        if hasattr(self, 'success_threshold') and self.success_threshold is not None:
            ax1.axhline(
                self.success_threshold,
                color='forestgreen',
                linestyle=':',
                linewidth=1.8,
                label=f'Success Threshold ({self.success_threshold})',
            )

        ax1.set_xlabel('Generation', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Reward / Fitness Score', fontsize=12, fontweight='bold')

        if len(gens) == 1:
            ax1.set_xlim(0.5, 1.5)

        current_gen = int(gens[-1])
        ax1.set_title(
            f'PyTorch + PyMOO Learning Curve - {self.env_type} (Gen {current_gen})',
            fontsize=14,
            fontweight='bold',
            pad=15,
        )
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower right', framealpha=0.9)

        # Secondary X-axis for Cumulative Episodes
        episodes_per_gen = self.problem.n_episodes_per_eval * getattr(self, 'current_pop_size', 50)
        if episodes_per_gen > 0:
            secax = ax1.secondary_xaxis(
                'top',
                functions=(lambda g: g * episodes_per_gen, lambda e: e / episodes_per_gen),
            )
            secax.set_xlabel(
                f'Cumulative Episodes (~{episodes_per_gen} per gen)',
                fontsize=11,
                fontweight='bold',
                labelpad=8,
            )

        # Add information box
        curr_best = float(bests[-1])
        curr_mean = float(means[-1]) if means is not None and len(means) > 0 else curr_best
        curr_episodes = (
            int(self.history_episodes[-1])
            if self.history_episodes
            else int(current_gen * episodes_per_gen)
        )
        info_text = (
            f'Environment: {self.env_name}\n'
            f'Algorithm: PyMOO GA\n'
            f'Current Gen: {current_gen}\n'
            f'Best Fitness: {curr_best:.2f}\n'
            f'Pop Mean: {curr_mean:.2f}\n'
            f'Total Episodes: {curr_episodes:,}'
        )
        ax1.text(
            0.02,
            0.98,
            info_text,
            transform=ax1.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.85),
            fontsize=9.5,
        )

        plt.tight_layout()
        plot_filename = os.path.join(self.checkpoint_dir, f"{self.model_prefix}_learning_curve.png")
        plt.savefig(plot_filename, dpi=200, bbox_inches='tight')

        if show:
            try:
                plt.show()
            except Exception as e:
                print(f"Could not display plot window: {e}")

        plt.close(fig)

        if verbose:
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

