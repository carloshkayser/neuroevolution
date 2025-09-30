from matplotlib import pyplot as plt
from ga import GeneticAlgorithm

import numpy as np

# @author Carlos Henrique Kayser
# Static Neural Network
# Supporting material used https://www.python-course.eu/neural_networks_with_python_numpy.php

class NeuralNetwork:

    def __init__(self, nodes, env, genetics = None, model_prefix = "model"):

        self.layers = len(nodes)
        self.nodes = nodes
        self.env = env
        self.model_prefix = model_prefix
        if genetics is None:
            self.genetics = GeneticAlgorithm(self.nodes, 15, 0.5, 0.5, generation_interval=0.5) # genetic default case is not passed
        else:
            self.genetics = genetics

    def feedforward(self, input_vector, weights):

        weights_matrices, bias = self.prepareWeights(weights)

        for layer in weights_matrices:

            output = self.reLu(np.dot(input_vector, layer) + bias)
            input_vector = output

        # Determine action based on environment type
        env_id = self.env.spec.id if hasattr(self.env, 'spec') else str(self.env)
        
        if 'CartPole' in env_id:
            # CartPole: binary decision (left/right)
            if output > 0.50:
                action = 1
            else:
                action = 0

        elif 'MountainCar' in env_id:
            # MountainCar: 3 discrete actions (left, no action, right)
            if len(output) == 3:
                # If network outputs 3 values, use argmax
                action = np.argmax(output)
            else:
                # If network outputs 1 value, map to 3 actions
                if output < -0.33:
                    action = 0  # Push left
                elif output > 0.33:
                    action = 2  # Push right
                else:
                    action = 1  # No push

        else:
            # Default behavior for unknown environments
            if hasattr(output, '__len__') and len(output) > 1:
                action = np.argmax(output)
            else:
                action = 1 if output > 0.5 else 0

        return action
    
    # Use reLu
    def reLu(self, x):
        return np.maximum(0,x)

    # Sigmoid Activation Function
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x))
    
    # Function prepares the weights to execute in the network
    def prepareWeights(self, weights):

        weights_matrices = []
        bias = 0
        matrix_size = 0
        last_index = 0

        for x, y in zip(self.nodes[:-1], self.nodes[1:]):
            matrix_size = (x * y) + last_index
            weights_matrices.append(np.reshape(weights[last_index:matrix_size], (x, y)))
            last_index = matrix_size

        # getting the bias
        bias = weights[last_index:]

        return weights_matrices, bias[0]

    def train(self, max_generations=100):

        learned_generations = []
        historic_score = []
        n_episodes = 0
        generation_count = 0
        episode_seed = 42  # Base seed for reproducibility

        while generation_count < max_generations:
            number_generation = (len(self.genetics.populations) - 1)
            generations = self.genetics.populations
            populations = generations[number_generation]
            individuals = populations.individuals

            score_individuals = []
            
            for ind_idx, ind in enumerate(individuals):

                obs, _ = self.env.reset(seed=episode_seed + n_episodes)
                total_reward = 0
                while True:
                    action = self.feedforward(obs, ind.weights)
                    obs, reward, done, truncated, info = self.env.step(action)
                    total_reward += reward
                    if done or truncated:
                        n_episodes = n_episodes + 1
                        break

                ind.fitness = total_reward
                score_individuals.append(total_reward)

                # Save the latest model after each generation
                current_generation = len(self.genetics.populations) - 1
                latest_individual = individuals[np.argmax([ind.fitness for ind in individuals])]
                np.save(f"checkpoints/{self.model_prefix}_latest", latest_individual.weights)

                # Receive the first individual
                if self.genetics.the_best is None:
                    self.genetics.the_best = ind
                    np.save(f"checkpoints/{self.model_prefix}_best", ind.weights)

                if self.genetics.the_best.fitness < ind.fitness:
                    self.genetics.the_best = ind
                    np.save(f"checkpoints/{self.model_prefix}_best", ind.weights)

            print("Generation: ", generation_count, " Score: ", np.amax(score_individuals), "Episodes: ", n_episodes)

            learned_generations = np.append(learned_generations, np.amax(score_individuals))

            historic_score.append(np.amax(score_individuals))

            # Comes after because at the moment of network creation the code already creates the first population
            self.genetics.evolution()
            generation_count += 1

        # Save final models (overwrite with final versions)
        np.save(f"checkpoints/{self.model_prefix}_best", self.genetics.the_best.weights)
        np.save(f"checkpoints/{self.model_prefix}_latest", self.genetics.the_best.weights)
        
        # Generate learning curve plot
        env_name = self.env.spec.id if hasattr(self.env, 'spec') else str(self.env)
        plot_filename = self.plot_learning_curve(learned_generations, env_name, self.model_prefix)
        
        print(f"Models saved:")
        print(f"- Best model: checkpoints/{self.model_prefix}_best.npy")
        print(f"- Latest model: checkpoints/{self.model_prefix}_latest.npy")
        print(f"- Learning curve: {plot_filename}")

        return self.genetics.the_best, historic_score, n_episodes

    def plot_learning_curve(self, learned_generations, env_name, model_prefix):
        """Generate and save learning curve plot with environment and algorithm info"""
        plt.figure(figsize=(10, 6))
        plt.plot(learned_generations, linewidth=2, color='blue')
        plt.title(f'Learning Curve - Genetic Algorithm on {env_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Best Fitness Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add algorithm and environment info as text
        plt.text(0.02, 0.98, f'Algorithm: Genetic Algorithm\nEnvironment: {env_name}\nGenerations: {len(learned_generations)}', 
                transform=plt.gca().transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Save the plot
        plot_filename = f"checkpoints/{model_prefix}_{env_name.split('-')[0].lower()}_learning_curve.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

        return plot_filename
