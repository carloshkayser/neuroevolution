# Neuroevolution: Evolving Neural Networks with Genetic Algorithms

A modular Python package implementing Artificial Neural Networks (ANN) trained using Genetic Algorithms (GA via PyMOO) to solve OpenAI Gymnasium environments including **CartPole**, **MountainCar**, **Maze** (`HardMaze`), and **CarRacing** (`CarRacing-v3`).

## Overview

This project demonstrates the application of neuroevolution, combining PyTorch neural networks with genetic algorithms to evolve optimal control policies without traditional backpropagation.

## Supported Environments

| Environment | Gym ID | Observation Space | Action Space | Description |
| :--- | :--- | :--- | :--- | :--- |
| **CartPole** | `CartPole-v1` | `Box(4)` | `Discrete(2)` | Balance a pole on a moving cart |
| **MountainCar** | `MountainCar-v0` | `Box(2)` | `Discrete(3)` | Drive a car up a steep hill |
| **Maze** | `HardMaze-v0` | `Box(9)` | `Box(3)` | Navigate a robot through a complex maze |
| **CarRacing** | `CarRacing-v3` | `Box(16)` | `Box(3)` | Race a car around procedural tracks and complete laps |

## Demos

| **CartPole** (`CartPole-v1`) | **MountainCar** (`MountainCar-v0`) |
| :---: | :---: |
| ![CartPole Demo](assets/cartpole.gif) | ![MountainCar Demo](assets/mountaincar.gif) |
| *Pole balancing control* | *Momentum building & hill climb* |

| **Maze** (`HardMaze-v0`) | **CarRacing** (`CarRacing-v3`) |
| :---: | :---: |
| ![Maze Demo](assets/maze.gif) | ![CarRacing Demo](assets/carracing.gif) |
| *Complex maze navigation* | *Full lap track completion (Reward 900+)* |

---

## Installation

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/carloshkayser/neuroevolution.git
cd neuroevolution

# Install dependencies and package
poetry install
```

### Using pip / virtualenv

```bash
# Install in editable mode
pip install -e .
```

---

## Usage

### 1. Command Line Interface (CLI)

Once installed, you can use the `neuroevolution` command (or `python -m neuroevolution` / `python main.py`):

#### Training
```bash
# Train on CartPole (default)
neuroevolution --train --env CartPole

# Train on MountainCar
neuroevolution --train --env MountainCar --generations 100 --population 50

# Train on Maze (HardMaze navigation)
neuroevolution --train --env Maze --generations 150 --population 60

# Train on CarRacing in parallel across 8 CPU cores (fast!)
neuroevolution --train --env CarRacing --generations 35 --population 50 --n-workers 8

# Resume training from existing weights with warm-start
neuroevolution --train --env CarRacing --resume --generations 20 --n-workers 8

# Custom network architecture
neuroevolution --train --env CartPole --hidden-layers 64 32 16

# Advanced GA hyperparameters
neuroevolution --train --env CarRacing --generations 35 --population 50 --crossover-prob 0.9 --mutation-prob 0.15 --n-workers 8
```

#### Testing / Evaluation
```bash
# Evaluate trained CartPole model
neuroevolution --test --env CartPole

# Evaluate trained MountainCar model
neuroevolution --test --env MountainCar --render

# Evaluate trained Maze model
neuroevolution --test --env Maze

# Evaluate trained CarRacing model (10 episodes)
neuroevolution --test --env CarRacing
```

#### Recording Demos (GIFs)
```bash
# Record GIF for trained CartPole model
neuroevolution --record-gif --env CartPole

# Record GIF for trained MountainCar model
neuroevolution --record-gif --env MountainCar

# Record GIF for trained Maze model
neuroevolution --record-gif --env Maze

# Record GIF for trained CarRacing model
neuroevolution --record-gif --env CarRacing --gif-path assets/carracing.gif
```

---

### 2. Python API

You can import and use `neuroevolution` directly in your Python code:

```python
from neuroevolution import PyTorchGeneticTrainer, CartPoleNet

# 1. Create a trainer for MountainCar
trainer = PyTorchGeneticTrainer(
    env_name="MountainCar-v0",
    network_architecture=[16, 8],
    model_prefix="mountaincar_ga"
)

# 2. Train with Genetic Algorithm
best_network, best_fitness, generations = trainer.train(
    n_generations=50,
    population_size=40,
    crossover_prob=0.9,
    mutation_prob=0.1
)

# 3. Evaluate the trained policy
avg_reward, std_reward = trainer.evaluate(n_episodes=10, render=False)
print(f"Average Reward: {avg_reward:.2f} +/- {std_reward:.2f}")

# 4. Record an animated GIF of the agent
trainer.record_gif(output_path="assets/mountaincar_demo.gif", fps=30)
```
