from matplotlib import pyplot as plt
from ga import Genetics

import numpy as np

# @author Carlos Henrique Kayser
# Rede Neural Estatica
# Material de apoio usado https://www.python-course.eu/neural_networks_with_python_numpy.php

class NeuralNetwork:

    def __init__(self, nodes, env, genetics = None ):

        self.layers = len(nodes)
        self.nodes = nodes
        self.env = env
        if genetics is None:
            self.genetics = Genetics(self.nodes, 15, 0.5, 0.5) # genetic default case is not passed
        else:
            self.genetics = genetics

    def feedforward(self, input_vector, weights):

        weights_matrices, bias = self.prepareWeights(weights)

        for layer in weights_matrices:

            output = self.reLu(np.dot(input_vector, layer) + [bias])
            input_vector = output

        # CartPole
        if output > 0.50:
            output = 1
        else:
            output = 0

        return output
    
    # Use reLu
    def reLu(self, x):
	    return np.maximum(0,x)

    # Função de Ativação Sigmóide
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x))
    
    # Funcao prepara os pesos para executarmos na rede
    def prepareWeights(self, weights):

        weights_matrices = []
        bias = 0
        matrix_size = 0
        last_index = 0

        for x, y in zip(self.nodes[:-1], self.nodes[1:]):
            matrix_size = (x * y) + last_index
            weights_matrices.append(np.reshape(weights[last_index:matrix_size], (x, y)))
            last_index = matrix_size

        # pegando o bias
        bias = weights[last_index:]

        return weights_matrices, bias[0]

    def train(self):

        learned_generations = []
        historic_score = []
        n_episodes = 0

        while(True):        
            number_generation = (len(self.genetics.populations) - 1)
            generations = self.genetics.populations
            populations = generations[number_generation]
            individuals = populations.individuals

            score_individuals = []
            
            for ind in individuals:

                obs = self.env.reset()
                award = 0
                while True:
                    self.env.render()
                    action = self.feedforward(obs, ind.weights)
                    obs, reward, done, info = self.env.step(action)
                    award += reward
                    if done:
                        n_episodes = n_episodes + 1
                        break

                ind.fitness = award
                score_individuals.append(award)

                # Receberá o primeiro individuo
                if self.genetics.the_best is None:
                    self.genetics.the_best = ind
                    np.save("the_best", ind.weights)                
                if self.genetics.the_best.fitness < ind.fitness:
                    self.genetics.the_best = ind
                    np.save("the_best", ind.weights)

            print("Population: ", len(self.genetics.populations) - 1, " Score: ", np.amax(score_individuals), "Episodes: ", n_episodes)

            learned_generations = np.append(learned_generations, np.amax(score_individuals))

            historic_score.append(np.amax(score_individuals))
            if self.stop_function(historic_score):
                break

            # Vem depois pois no momento da criação da rede o code ja cria a primeira população
            self.genetics.evolution()

        # Apresentar curva de aprendizado
        plt.plot(learned_generations)
        plt.xlabel('Numero de populações')
        plt.ylabel('Pontuação')
        plt.title('Numero de populações vs Pontuação')
        plt.grid()
        plt.show()

        return self.genetics.the_best, historic_score, n_episodes

    def stop_function(self, historic_score):

        # Caso o score fique estagnado então interrompe o treinamento
        if len(historic_score) >= 5:
            for i in range(len(historic_score)):
                if historic_score[i] == historic_score[i-1] and historic_score[i-1] == historic_score[i-2] and historic_score[i-2] == historic_score[i-3] and historic_score[i-3] == historic_score[i-4]:
                    print("Stop Function")
                    return True

    def test(self, the_best):

        sucess = False
        obs = self.env.reset()
        award = 0
        while True:
            self.env.render()
            action = self.feedforward(obs, the_best.weights)
            obs, reward, done, info = self.env.step(action)
            award += reward
            print('award', award)
            if done:
                if award == 200.0:
                    sucess = True
                    return sucess
                elif award == 500.0:
                    sucess = True
                    return sucess
                break
