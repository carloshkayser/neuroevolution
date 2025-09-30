import numpy as np

# @author Carlos Henrique

class Individual:

    def __init__(self, weights):
        self.weights = weights
        self.fitness = 0


class Population:

    def __init__(self, individuals):
        self.individuals = individuals


class GeneticAlgorithm():

    def __init__(self, ann_nodes, no_individuals, mutation_chance, crossover_chance, generation_interval):

        # Set seed for reproducible results
        np.random.seed(42)
        
        self.ann_nodes = ann_nodes
        self.mutation_chance = mutation_chance
        # The number of individuals will be used to define the size of a population
        self.no_individuals = no_individuals
        self.crossover_chance = crossover_chance
        self.populations = []
        self.create_initial_population()
        self.the_best = None
        self.generation_interval = generation_interval # Controls the percentage of population that will be replaced for the next generation

    def createIndividual(self, weights = None):

        if weights is None:

            qtdWeights = 0

            for x, y in zip(self.ann_nodes[:-1], self.ann_nodes[1:]):
                qtdWeights = qtdWeights + (x * y)

            weights = self.create_random_weight( qtdWeights )                                 

        return Individual(weights)

    def create_initial_population(self):

        individuals = []

        # What is the number of individuals that the population has
        size = self.no_individuals

        for i in range(size):
            ind = self.createIndividual()
            individuals.append(ind)

        # Add the created individuals to the population
        population = Population(individuals)

        # Setting the created population
        self.populations.append(population)

    # Generate initial weights
    def create_random_weight(self, qtdWeights):

        # Create the weight vector that will be used in the neural network later
        weights = []

        for _ in range(qtdWeights):
            weights.append(np.random.rand())

        bias = np.random.rand()
        weights.append(bias)
        
        return np.array(weights)

    # Single point crossover
    def crossover(self, list_individuals):

        newlist_individuals = list_individuals[:]

        qtd_cross = int(len(list_individuals) / 2)

        i = 0
        for x, y in zip(list_individuals[:qtd_cross], list_individuals[qtd_cross:]):

            random_rate = np.random.rand()

            if random_rate <= self.crossover_chance:

                j = np.random.randint(0, len(list_individuals[0].weights))
                newlist_individuals[i].weights = np.append(x.weights[:j], y.weights[j:]) # Concatenating weights
                i = i + 1
                newlist_individuals[i].weights = np.append(y.weights[:j], x.weights[j:]) # Concatenating weights
                i = i + 1

        return newlist_individuals

    def mutation(self, list_individuals):

        len_weights = len(list_individuals[0].weights)

        for x in range(len(list_individuals)):
            for i in range(len_weights):
                random_rate = np.random.rand()
                if random_rate <= self.mutation_chance:
                    list_individuals[x].weights[i] = np.random.rand()

        return list_individuals

    # Roulette wheel selection method
    def roulette_selection(self, list_individuals, num):
        """
            Source: https://stackoverflow.com/questions/177271/roulette-selection-in-genetic-algorithms/5315710#5315710
        """

        # Summing all scores to divide later
        total_fitness = 0
        rel_fitness = []
        for x in range(len(list_individuals)):
            total_fitness = total_fitness + list_individuals[x].fitness
            
        for x in range(len(list_individuals)):
            rel_fitness.append( list_individuals[x].fitness / total_fitness )
            
        # Generating probability intervals for each individual in the population
        probs = [sum(rel_fitness[:i+1]) for i in range(len(rel_fitness))]

        # Forming the set of selected individuals
        new_population = []
        for n in range(int(num)):
            r = np.random.rand()
            for (i, individual) in enumerate(list_individuals):
                if r <= probs[i]:
                    new_population.append(individual)
                    break

        return new_population     

    # This method creates the next populations based on iteration of the first population
    # Where the process of selection, mutation and crossover will be done
    def evolution(self):
        
        the_best = []
        newIndividuals = []
        number_population = (len(self.populations) - 1)
        individuals = self.populations[number_population].individuals

        interval = int(self.no_individuals * self.generation_interval)

        # Getting the best individuals from the previous population using the roulette method
        the_best = self.roulette_selection(individuals, interval)

        # Crossover of the best individuals from the previous population
        newIndividuals_cross = self.crossover(the_best)

        # Mutation of individuals after crossover
        newIndividuals_cross_mutation = self.mutation(newIndividuals_cross)

        # Elite Individual
        newIndividuals.append(self.the_best)
        
        # Joining individuals after crossover of individuals from the last population
        for i in range(len(newIndividuals_cross_mutation)):
            newIndividuals.append(newIndividuals_cross_mutation[i])

        # Generating random individuals to complete the population
        no_individuals = self.no_individuals - len(newIndividuals)
        for i in range(no_individuals):
            newIndividuals.append(self.createIndividual())

        # Generate new population
        self.new_population(newIndividuals)

    # the_best will be used to pass the best individuals from the previous population
    def new_population(self, newIndividuals):

        # Add the created individuals to the population
        population = Population(newIndividuals)

        # Setting the created population
        self.populations.append(population)