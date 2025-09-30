# Neural Network with Genetic Algorithm for Gym Environments

A Python implementation of an Artificial Neural Network (ANN) trained using a Genetic Algorithm (GA) to solve OpenAI Gymnasium environments, including CartPole and MountainCar.

## Overview

This project demonstrates the application of neuroevolution, combining artificial neural networks with genetic algorithms to learn optimal control policies. The implementation uses a genetic algorithm to evolve the weights of a neural network that controls various Gym environment agents, eliminating the need for traditional backpropagation-based training.

## Features

- **Multi-Environment Support**: CartPole-v1 and MountainCar-v0 environments
- **Command-Line Interface**: Easy training and testing with argument parsing
- **Static Neural Network**: Feedforward neural network with customizable architecture
- **Genetic Algorithm**: Evolution-based optimization for neural network weights
- **Environment-Specific Networks**: Adaptive architecture for different environments
- **Professional Evaluation**: Comprehensive agent testing with statistics
- **Checkpoint System**: Automatic saving of best and latest models
- **Reproducible Results**: Seed-based deterministic training

## Architecture

### Neural Network Components
- **Activation Functions**: ReLU, Sigmoid, and Softmax implementations
- **Feedforward Propagation**: Environment-aware forward pass through network layers
- **Adaptive Architecture**: Different topologies for CartPole (binary decisions) and MountainCar (multi-action)
- **Environment Detection**: Automatic action space adaptation based on environment type

### Genetic Algorithm Components
- **Population Management**: Handling of individual agents and populations
- **Roulette Selection**: Fitness-proportionate selection for reproduction
- **Single-Point Crossover**: Genetic recombination for offspring generation
- **Mutation**: Random weight perturbation for exploration
- **Elitism**: Preservation of best-performing individuals

## Requirements

- Python 3.12+
- NumPy >= 2.3.3
- Matplotlib >= 3.10.6
- Gymnasium >= 1.2.1

## Installation

1. Clone the repository:
```bash
git clone https://github.com/carloshkayser/ann-ga-cart-pole.git
cd ann-ga-cart-pole
```

2. Install dependencies using Poetry:
```bash
poetry install
```

Or using pip:
```bash
pip install numpy matplotlib gymnasium
```

## Usage

### Command Line Interface

The project supports flexible command-line usage with the following options:

```bash
# Display help and available options
python main.py --help

# Train with CartPole environment (default)
python main.py --train --env CartPole --generations 100

# Train with MountainCar environment
python main.py --train --env MountainCar --generations 2000 --population 100 --mutation 0.02 --crossover 0.1

# Test/evaluate trained CartPole agent
python main.py --test --env CartPole

# Test/evaluate trained MountainCar agent
python main.py --test --env MountainCar

# Default behavior (training with CartPole)
python main.py
```

### Environment-Specific Features

#### CartPole-v1
- **Objective**: Balance pole on cart for 500 time steps
- **Actions**: Binary decision (left/right movement)
- **Success Criteria**: Reaching maximum steps without pole falling
- **Network Architecture**: [4, 4, 3, 1] - Single output for binary decision

#### MountainCar-v0
- **Objective**: Drive car to reach the goal at the top of the hill
- **Actions**: Three discrete actions (push left, no action, push right)
- **Success Criteria**: Reaching the goal position
- **Network Architecture**: [2, 8, 6, 3] - Three outputs for action selection

### Professional Evaluation

The evaluation system provides comprehensive statistics:
- **Multi-Episode Testing**: Runs 20 episodes for statistical significance
- **Performance Metrics**: Average reward, standard deviation
- **Visual Confirmation**: 3 episodes with rendering for verification
- **Environment-Specific Success Detection**: Proper success criteria for each environment

### Configuration

The neural network and genetic algorithm parameters can be configured in `main.py`:

```python
# Environment-specific network architectures
if args.env == 'CartPole':
    size_of_network = [number_of_inputs, 4, 3, 1]  # Binary decision
elif args.env == 'MountainCar':
    size_of_network = [number_of_inputs, 8, 6, number_of_actions]  # Multi-action

# Genetic algorithm parameters
genetic = GeneticAlgorithm(
    size_of_network,           # Network topology
    50,                        # Population size
    0.01,                      # Mutation rate
    0.01,                      # Crossover rate
    generation_interval=0.50   # Generation replacement ratio
)
```

### Key Parameters

- **Population Size**: Number of individuals in each generation
- **Mutation Rate**: Probability of weight mutation for each individual
- **Crossover Rate**: Probability of genetic recombination between parents
- **Generation Interval**: Percentage of population replaced each generation

## Project Structure

```
ann-ga-cart-pole/
├── ann.py              # Neural network implementation
├── ga.py               # Genetic algorithm implementation  
├── main.py             # Main execution script with CLI
├── pyproject.toml      # Project dependencies and metadata
├── checkpoints/        # Saved model weights
│   ├── cartpole_best.npy    # Best CartPole model
│   ├── cartpole_latest.npy  # Latest CartPole model
│   ├── mountaincar_best.npy # Best MountainCar model
│   └── mountaincar_latest.npy # Latest MountainCar model
└── README.md           # Project documentation
```

## Algorithm Flow

1. **Initialization**: Create initial population with random neural network weights
2. **Evaluation**: Test each individual in the CartPole environment
3. **Selection**: Choose best-performing individuals based on fitness scores
4. **Reproduction**: Create offspring through crossover and mutation
5. **Replacement**: Form new generation with offspring and elite individuals
6. **Iteration**: Repeat until convergence or maximum generations reached

## Performance Metrics

The system tracks several performance indicators:

- **Fitness Score**: Cumulative reward achieved in the CartPole environment
- **Episode Count**: Number of environment episodes completed
- **Generation Progress**: Population number and best score per generation
- **Convergence**: Automatic stopping when performance plateaus

## Results

### CartPole-v1
The trained neural network learns to balance the pole on the cart, achieving the CartPole-v1 success criteria of maintaining balance for 500 time steps. The genetic algorithm typically converges within 10-50 generations depending on the hyperparameters.

### MountainCar-v0
The neural network learns to build momentum by rocking back and forth to reach the goal at the top of the hill. This environment is more challenging due to the sparse reward structure and the need for strategic action sequences.

### Training Performance
- **Reproducible Results**: Deterministic training with seed-based randomization
- **Automatic Model Saving**: Best and latest models saved during training
- **Professional Evaluation**: Statistical analysis with multiple episode testing

## Technical Details

### Neural Network Architectures

#### CartPole Network
- **Input layer**: 4 neurons (cart position, cart velocity, pole angle, pole angular velocity)
- **Hidden layers**: 4 and 3 neurons with ReLU activation
- **Output layer**: 1 neuron with threshold-based binary decision (left/right)

#### MountainCar Network  
- **Input layer**: 2 neurons (car position, car velocity)
- **Hidden layers**: 8 and 6 neurons with ReLU activation
- **Output layer**: 3 neurons for discrete action selection (push left, no action, push right)

### Genetic Algorithm Features
- **Encoding**: Direct weight encoding in chromosomes
- **Selection**: Roulette wheel (fitness-proportionate) selection
- **Crossover**: Single-point crossover with configurable probability
- **Mutation**: Gaussian noise addition to weights
- **Replacement**: Generational replacement with elitism

### Environment Adaptation
The neural network automatically adapts its decision-making based on the environment:
- **CartPole**: Threshold-based binary output (>0.5 = right, ≤0.5 = left)
- **MountainCar**: Multi-output with argmax selection or threshold-based mapping

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- OpenAI Gymnasium for the CartPole environment
- Python Course EU for neural network implementation guidance
- The broader machine learning and evolutionary computation communities
