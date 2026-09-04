"""Command-line interface for neuroevolution."""

import argparse
import os
import shutil
import gymnasium as gym
import numpy as np

try:
    import gymnasium_hardmaze
except ImportError:
    pass

try:
    import minigrid
except ImportError:
    pass

from typing import Any
from contextlib import nullcontext
from .models import CartPoleNet
from .trainer import PyTorchGeneticTrainer, make_env
from .agent import NeuralNetworkAgent, evaluate, load_model
from .tracker import TrainingJob, create_job, list_jobs


def print_jobs_table(jobs: list[dict[str, Any]]) -> None:
    """Print a formatted terminal table summarizing training jobs."""
    if not jobs:
        print("No training jobs found.")
        return

    headers = ["Job ID", "Env", "Created", "Gens", "Pop", "Mut", "Cross", "Fitness", "Eval", "Artifacts"]
    rows = []
    for j in jobs:
        created = str(j.get("created_at", "-"))
        if "T" in created:
            created = created.replace("T", " ")[:19]
        best_fit = j.get("best_fitness", "-")
        best_fit_str = f"{best_fit:.2f}" if isinstance(best_fit, (int, float)) else str(best_fit)

        avg_eval = j.get("avg_eval_reward", "-")
        avg_eval_str = f"{avg_eval:.2f}" if isinstance(avg_eval, (int, float)) else str(avg_eval)

        artifacts = []
        if j.get("has_model"):
            artifacts.append(".pth")
        if j.get("has_plot"):
            artifacts.append(".png")
        art_str = ",".join(artifacts) if artifacts else "-"

        rows.append([
            str(j.get("job_id", "-")),
            str(j.get("env_name", "-")),
            created,
            str(j.get("generations", "-")),
            str(j.get("population", "-")),
            str(j.get("mutation_prob", "-")),
            str(j.get("crossover_prob", "-")),
            best_fit_str,
            avg_eval_str,
            art_str,
        ])

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))

    print("\n" + header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)))
    print(f"\nTotal jobs: {len(jobs)}\n")



def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Neural Network with Genetic Algorithm for Gym Environments',
        prog='neuroevolution',
    )
    parser.add_argument('--train', action='store_true', help='Train the neural network')
    parser.add_argument('--test', action='store_true', help='Test/evaluate the trained agent')
    parser.add_argument(
        '--env',
        choices=['CartPole', 'LunarLander', 'MountainCar', 'Maze', 'HardMaze', 'FourRooms', 'MiniGrid-FourRooms-v0', 'CarRacing', 'CarRacing-v0', 'CarRacing-v3'],
        default='CartPole',
        help='Environment to use: CartPole (CartPole-v1), LunarLander (LunarLander-v3), MountainCar (MountainCar-v0), Maze (HardMaze-v0), FourRooms (MiniGrid-FourRooms-v0), or CarRacing (CarRacing-v0/v3)',
    )
    parser.add_argument(
        '--hidden-layers',
        type=int,
        nargs='+',
        default=None,
        help='Hidden layer sizes, e.g. --hidden-layers 64 32',
    )
    parser.add_argument(
        '--generations',
        type=int,
        default=100,
        help='Maximum number of generations for training (default: 100)',
    )
    parser.add_argument(
        '--population',
        type=int,
        default=50,
        help='Population size (number of individuals) (default: 50)',
    )
    parser.add_argument(
        '--mutation-prob',
        type=float,
        default=0.1,
        help='Mutation probability (default: 0.1)',
    )
    parser.add_argument(
        '--crossover-prob',
        type=float,
        default=0.9,
        help='Crossover probability (default: 0.9)',
    )
    parser.add_argument(
        '--generation-interval',
        type=float,
        default=0.50,
        help='Generation replacement ratio (default: 0.50)',
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render the environment during testing',
    )
    parser.add_argument(
        '--record-gif',
        action='store_true',
        help='Record an animated GIF of the trained agent',
    )
    parser.add_argument(
        '--gif-path',
        type=str,
        default=None,
        help='Path to save the recorded GIF (default: assets/<env>.gif)',
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory to save/load checkpoints (default: checkpoints)',
    )
    parser.add_argument(
        '--episodes-per-eval',
        type=int,
        default=None,
        help='Number of episodes per evaluation during training (default: 1 for deterministic Maze, 3 for others)',
    )
    parser.add_argument(
        '--show-plot',
        action='store_true',
        help='Display the plot window when training finishes',
    )
    parser.add_argument(
        '--no-live-plot',
        action='store_true',
        help='Disable real-time plot saving at each generation (only save at the end)',
    )
    parser.add_argument(
        '--n-workers',
        type=int,
        default=None,
        help='Number of parallel workers for evaluation (default: auto, min(CPU count, 8))',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from existing checkpoint weights if available',
    )
    parser.add_argument(
        '--list-jobs',
        action='store_true',
        help='List all past training jobs and their results',
    )
    parser.add_argument(
        '--jobs-dir',
        type=str,
        default='jobs',
        help='Base directory where training experiment jobs are stored (default: jobs)',
    )
    parser.add_argument(
        '--no-job',
        action='store_true',
        help='Disable experiment job directory creation and save checkpoints directly to --checkpoint-dir',
    )
    
    return parser.parse_args(args)


def main(args=None):
    parsed_args = parse_args(args)

    if parsed_args.list_jobs:
        jobs = list_jobs(base_dir=parsed_args.jobs_dir)
        print_jobs_table(jobs)
        return

    # If neither train, test, nor record_gif specified, default to train
    if not parsed_args.train and not parsed_args.test and not parsed_args.record_gif:
        parsed_args.train = True

    # Environment configuration
    if parsed_args.env == 'CartPole':
        env_name = 'CartPole-v1'
        model_prefix = 'cartpole'
        default_hidden = [4, 3]
        output_size = 1
        success_threshold = 475
    elif parsed_args.env == 'LunarLander':
        env_name = 'LunarLander-v3'
        model_prefix = 'lunarlander'
        default_hidden = [16, 12]
        output_size = 4
        success_threshold = 200
    elif parsed_args.env == 'MountainCar':
        env_name = 'MountainCar-v0'
        model_prefix = 'mountaincar'
        default_hidden = [16, 8]
        output_size = 1
        success_threshold = -110
    elif parsed_args.env in ('Maze', 'HardMaze'):
        env_name = 'HardMaze-v0'
        model_prefix = 'maze'
        default_hidden = [16, 12]
        output_size = 3
        success_threshold = 1.0
    elif parsed_args.env in ('CarRacing', 'CarRacing-v0', 'CarRacing-v3'):
        env_name = 'CarRacing-v3'
        model_prefix = 'carracing'
        default_hidden = [16, 12]
        output_size = 3
        success_threshold = 900
    elif parsed_args.env in ('FourRooms', 'MiniGrid-FourRooms-v0'):
        env_name = 'MiniGrid-FourRooms-v0'
        model_prefix = 'fourrooms'
        default_hidden = [32, 16]
        output_size = 7
        success_threshold = 0.5
    else:
        raise ValueError(f"Unsupported environment: {parsed_args.env}")

    hidden_layers = parsed_args.hidden_layers if parsed_args.hidden_layers is not None else default_hidden

    print(f"Using environment: {env_name}")
    print(f"Hidden layer architecture: {hidden_layers}")

    os.makedirs(parsed_args.checkpoint_dir, exist_ok=True)

    # Set seed for reproducibility
    seed = 42
    np.random.seed(seed)

    env = make_env(env_name)
    env.reset(seed=seed)
    number_of_inputs = int(np.prod(env.observation_space.shape))
    env.close()

    is_single_episode_env = parsed_args.env in ('Maze', 'HardMaze', 'CarRacing', 'CarRacing-v0', 'CarRacing-v3')
    n_episodes_per_eval = parsed_args.episodes_per_eval if parsed_args.episodes_per_eval is not None else (1 if is_single_episode_env else 3)

    job = None
    if parsed_args.train and not parsed_args.no_job:
        job_config = {
            "env_name": env_name,
            "env_arg": parsed_args.env,
            "generations": parsed_args.generations,
            "population": parsed_args.population,
            "mutation_prob": parsed_args.mutation_prob,
            "crossover_prob": parsed_args.crossover_prob,
            "generation_interval": parsed_args.generation_interval,
            "hidden_layers": hidden_layers,
            "n_workers": parsed_args.n_workers,
            "n_episodes_per_eval": n_episodes_per_eval,
            "model_prefix": model_prefix,
        }
        job = create_job(env_name=model_prefix, config=job_config, base_dir=parsed_args.jobs_dir)
        train_checkpoint_dir = job.job_dir
    else:
        train_checkpoint_dir = parsed_args.checkpoint_dir

    tee_context = job.capture_output() if job is not None else nullcontext()
    with tee_context:
        if job is not None:
            print(f"\n[Job Tracker] Initialized job: {job.job_id}")
            print(f"[Job Tracker] Job directory: {job.job_dir}")
            print(f"[Job Tracker] Output log: {os.path.join(job.job_dir, 'train.log')}")

        if parsed_args.train:
            print("\n=== TRAINING MODE ===")
            print(f"Maximum generations: {parsed_args.generations}")
            print(f"Population size: {parsed_args.population}")
            print(f"Mutation rate: {parsed_args.mutation_prob}")
            print(f"Crossover rate: {parsed_args.crossover_prob}")
            print(f"Generation interval: {parsed_args.generation_interval}")
            print(f"Episodes per evaluation: {n_episodes_per_eval}")
            if parsed_args.n_workers is not None:
                print(f"Parallel workers: {parsed_args.n_workers}")

            initial_weights = None
            if parsed_args.resume:
                for candidate_dir in [train_checkpoint_dir, parsed_args.checkpoint_dir]:
                    weights_file = os.path.join(candidate_dir, f"{model_prefix}_weights.npy")
                    model_file = os.path.join(candidate_dir, f"{model_prefix}_best.pth")
                    if os.path.exists(weights_file):
                        initial_weights = np.load(weights_file)
                        print(f"Resuming training from weights: {weights_file}")
                        break
                    elif os.path.exists(model_file):
                        tmp_trainer = PyTorchGeneticTrainer(
                            env_name=env_name,
                            network_architecture=hidden_layers,
                            model_prefix=model_prefix,
                            checkpoint_dir=candidate_dir,
                            n_workers=1,
                        )
                        net = tmp_trainer.load_model(model_file)
                        initial_weights = net.get_weights_as_vector().detach().cpu().numpy()
                        print(f"Resuming training from model: {model_file}")
                        break

            trainer = PyTorchGeneticTrainer(
                env_name=env_name,
                network_architecture=hidden_layers,
                model_prefix=model_prefix,
                checkpoint_dir=train_checkpoint_dir,
                n_episodes_per_eval=n_episodes_per_eval,
                n_workers=parsed_args.n_workers,
                job=job,
            )

            best_network, best_fitness, n_generations = trainer.train(
                n_generations=parsed_args.generations,
                population_size=parsed_args.population,
                crossover_prob=parsed_args.crossover_prob,
                mutation_prob=parsed_args.mutation_prob,
                live_plot=not parsed_args.no_live_plot,
                show_plot=parsed_args.show_plot,
                resume=parsed_args.resume,
                initial_weights=initial_weights,
            )

            print(f"Training completed after {n_generations} generations. Best fitness: {best_fitness:.2f}")

            # Evaluate trained agent
            print("\n=== EVALUATING TRAINED AGENT ===")
            eval_env = make_env(env_name)
            print("Testing trained agent without rendering:")
            avg_reward, std_reward = evaluate(best_network, eval_env, n_episodes=10, render=False, env_type=parsed_args.env)

            if avg_reward >= success_threshold:
                print(f'\nSUCCESS! Average reward {avg_reward:.2f} meets threshold {success_threshold}')
            else:
                print(f'\nNot quite there yet. Average reward {avg_reward:.2f}, threshold is {success_threshold}')

            eval_env.close()

            if job is not None:
                job.save_metrics({
                    "best_fitness": float(best_fitness),
                    "success_threshold": success_threshold,
                    "success": bool(best_fitness >= success_threshold),
                    "avg_eval_reward": float(avg_reward),
                    "std_eval_reward": float(std_reward),
                    "generations_completed": int(n_generations),
                })
                promoted = job.promote(
                    checkpoint_dir=parsed_args.checkpoint_dir,
                    assets_dir="assets",
                    model_prefix=model_prefix,
                )
                if promoted:
                    print(f"[Job Tracker] Promoted best artifacts to top-level: {', '.join(promoted)}")

        if parsed_args.test or parsed_args.record_gif:
            trainer = PyTorchGeneticTrainer(
                env_name=env_name,
                network_architecture=hidden_layers,
                model_prefix=model_prefix,
                checkpoint_dir=parsed_args.checkpoint_dir,
                n_workers=1,
            )

            model_path = os.path.join(parsed_args.checkpoint_dir, f'{model_prefix}_best.pth')
            weights_file = os.path.join(parsed_args.checkpoint_dir, f'{model_prefix}_weights.npy')

            if os.path.exists(model_path):
                trainer.load_model(model_path)
                agent = trainer.best_network
                print(f"Loaded PyTorch model from {model_path}")
            elif os.path.exists(weights_file):
                saved_weights = load_model(weights_file)
                net = CartPoleNet(number_of_inputs, hidden_layers, output_size)
                net.set_weights_from_vector(saved_weights)
                trainer.best_network = net
                agent = net
                print(f"Loaded weights vector from {weights_file}")
            else:
                print(f"No saved model found at {model_path} or {weights_file}! Please train first with --train --env {parsed_args.env}")
                return

            if parsed_args.test:
                print("\n=== TESTING MODE ===")
                eval_env = make_env(env_name, render_mode='human' if parsed_args.render else None)
                print(f"Evaluating trained {parsed_args.env} agent (10 episodes)...")
                avg_reward, std_reward = evaluate(agent, eval_env, n_episodes=10, render=parsed_args.render, env_type=parsed_args.env)
                eval_env.close()

            if parsed_args.record_gif:
                print("\n=== RECORDING GIF ===")
                gif_path = parsed_args.gif_path or os.path.join('assets', f'{model_prefix}.gif')
                trainer.record_gif(output_path=gif_path)
                if job is not None:
                    job_gif = os.path.join(job.job_dir, f'{model_prefix}.gif')
                    if os.path.abspath(gif_path) != os.path.abspath(job_gif) and os.path.exists(gif_path):
                        shutil.copy2(gif_path, job_gif)
                        job.save_metrics({"has_gif": True})


if __name__ == '__main__':
    main()
