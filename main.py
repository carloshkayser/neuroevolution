from ann import NeuralNetwork
from ga import Genetics

import gymnasium as gym

env = gym.make('CartPole-v1')

number_of_inputs = env.observation_space.shape[0]
number_of_actions = env.action_space.n # number_of_outputs

size_of_network = [number_of_inputs, 4, 3, 1]

genetic = Genetics(size_of_network, 50, 0.01, 0.01, intervalo_geracao = 0.50)
network = NeuralNetwork( size_of_network, env = env, genetics = genetic)

the_best, _, n_episodes = network.train()

sucess = network.test(the_best)

if sucess:
    print('Objetivo alcançado! \nForam necessários ', n_episodes, ' episódios para solucionar o desafio!')

else:
    print('Não foi possível solucionar o desafio em ', n_episodes, ' episódios!')

env.close()
