import numpy as np

# @author Carlos Henrique

class Individual:

    def __init__(self, weights):
        self.weights = weights
        self.fitness = 0


class Population:

    def __init__(self, individuals):
        self.individuals = individuals


class Genetics():

    def __init__(self, nodes, no_individuals, chance_mutacao, chance_cruzamento, intervalo_geracao):

        self.nodes = nodes
        self.chance_mutacao = chance_mutacao
        # O numero de individuos será o demilitados usado para dizer o tamanho de uma população
        self.no_individuals = no_individuals
        self.chance_cruzamento = chance_cruzamento
        self.populations = []
        self.create_initial_population()
        self.the_best = None
        self.intervalo_geracao = intervalo_geracao # Controle a porcentagem da população que será substituída para a próxima geração

    def createIndividual(self, weights = None):

        if weights is None:

            qtdWeights = 0

            for x, y in zip(self.nodes[:-1], self.nodes[1:]):
                qtdWeights = qtdWeights + (x * y)

            weights = self.create_random_weight( qtdWeights )                                 

        return Individual(weights)

    def create_initial_population(self):

        individuals = []

        # Qual é a quantidade de indiviudos que a população possui
        size = self.no_individuals

        for i in range(size):
            ind = self.createIndividual()
            individuals.append(ind)

        # Adicionar os individuos criados a população
        population = Population(individuals)

        # Setando a populacao criada
        self.populations.append(population)

    # Gera os pesos iniciais
    def create_random_weight(self, qtdWeights):

        # Criar o vetor de pesos que serão usados na rede neural posteriormente
        weights = []

        for _ in range(qtdWeights):
            weights.append(np.random.rand())

        bias = np.random.rand()
        weights.append(bias)
        
        return np.array(weights)

    # Crossover de um ponto
    def crossover(self, list_individuals):

        newlist_individuals = list_individuals[:]

        qtd_cross = int(len(list_individuals) / 2)

        i = 0
        for x, y in zip(list_individuals[:qtd_cross], list_individuals[qtd_cross:]):

            taxa_random = np.random.rand()

            if taxa_random <= self.chance_cruzamento:

                j = np.random.randint(0, len(list_individuals[0].weights))
                newlist_individuals[i].weights = np.append(x.weights[:j], y.weights[j:]) # Concatenando os pesos
                i = i + 1
                newlist_individuals[i].weights = np.append(y.weights[:j], x.weights[j:]) # Concatenando os pesos
                i = i + 1

        return newlist_individuals

    def mutation(self, list_individuals):

        len_weights = len(list_individuals[0].weights)

        for x in range(len(list_individuals)):
            for i in range(len_weights):
                taxa_random = np.random.rand()
                if taxa_random <= self.chance_cruzamento:
                    list_individuals[x].weights[i] = np.random.rand()

        return list_individuals

    # Metodo de seleção por roleta
    def selecao_roleta(self, list_individuals, num):
        """
            Fonte: https://stackoverflow.com/questions/177271/roulette-selection-in-genetic-algorithms/5315710#5315710
        """

        # Somando todas as notas para dividir posteriormente
        total_fitness = 0
        rel_fitness = []
        for x in range(len(list_individuals)):
            total_fitness = total_fitness + list_individuals[x].fitness
            
        for x in range(len(list_individuals)):
            rel_fitness.append( list_individuals[x].fitness / total_fitness )
            
        # Gerando intervalos de probabilidade para cada individuo da população
        probs = [sum(rel_fitness[:i+1]) for i in range(len(rel_fitness))]

        # Formando o conjunto de individuos selecionados
        new_population = []
        for n in range(int(num)):
            r = np.random.rand()
            for (i, individual) in enumerate(list_individuals):
                if r <= probs[i]:
                    new_population.append(individual)
                    break

        return new_population     

    # Esse metodo cria as proximas populaçoes com base na iteração da primeira população
    # Onde será feito o processo de selection mutation e crossover
    def evolution(self):
        
        the_best = []
        newIndividuals = []
        number_population = (len(self.populations) - 1)
        individuals = self.populations[number_population].individuals

        intervalo = int(self.no_individuals * self.intervalo_geracao)

        # Pegando os melhores individuos da população anterior utilizando o método de roleta
        the_best = self.selecao_roleta(individuals, intervalo)

        # Crossover dos melhores individuos da população anterior
        newIndividuals_cross = self.crossover(the_best)

        # Mutação dos indivíduos após o crossover
        newIndividuals_cross_mutation = self.mutation(newIndividuals_cross)

        # Individuo Elite
        newIndividuals.append(self.the_best)
        
        # Unindo os individuos após o crossover dos individuos da utlima população
        for i in range(len(newIndividuals_cross_mutation)):
            newIndividuals.append(newIndividuals_cross_mutation[i])

        # Gerando individuos randomicos para completar a população
        no_individuals = self.no_individuals - len(newIndividuals)
        for i in range(no_individuals):
            newIndividuals.append(self.createIndividual())

        # Gerar nova população
        self.new_population(newIndividuals)

    # the_best será usado para passar os melhores individuos da população passada
    def new_population(self, newIndividuals):

        # Adicionar os individuos criados a população
        population = Population(newIndividuals)

        # Setando a populacao criada
        self.populations.append(population)