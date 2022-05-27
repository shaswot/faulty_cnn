import numpy as np
import csv
import time
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

class BaseIndividual(ABC):
    '''
        Abstract base class for implementing individual

        An individual consists of three basics attributes:
            * Genotype (a.k.a chromosome)
            * Phenotype: actual solutions to the optimization problem
            * Fitness

        This class provides an interface to abtract away the implementation of GA
        from the problem-specific details.
    '''

    # A dictionary contains all the necessary info to calculate an individual's fitness
    FITNESS_PARAMS: dict = None

    __slots__ = ['_genotype', '_phenotype', '_fitness']

    def __init__(self, genotype, phenotype):
        self._genotype = genotype
        self._phenotype = phenotype
        self._fitness = self.calculate_fitness(phenotype)

    @classmethod
    def set_individual_attrs(cls, genetic_info, fitness_params):
        '''
            Set up all necessary information for the class
            Should be called before starting GA
            Params:
                genetic_info: a dict contains general info about genotype.
                              A common info is the gene's length.
                fitness_params: a dict contains all necessary params
                                to calculate fitness of individual
        '''
        pass

    @classmethod
    def genotype_to_phenotype(cls, genotype):
        ''' Implement the genotype -> phenotype mapping '''
        pass

    @classmethod
    def phenotype_to_genotype(cls, phenotype):
        ''' Implement the phenotype -> genotype mapping '''
        pass

    @abstractmethod
    def calculate_fitness(self, phenotype):
        '''
        Implement the fitness function.
        Given an individual's phenotype, calculate its fitness value
        '''
        pass

    @classmethod
    def crossover(cls, parent1, parent2):
        ''' Implement the crossover operator '''
        pass

    @classmethod
    def mutate(cls, parent):
        ''' Implement the mutation operator '''
        pass

    @classmethod
    def from_phenotype(cls, phenotype):
        ''' Generate an individual from a give phenotype '''
        genotype = cls.phenotype_to_genotype(phenotype)

        return cls(genotype=genotype, phenotype=phenotype)

    @classmethod
    def from_genotype(cls, genotype):
        ''' Generate an individual from a given genotype '''
        phenotype = cls.genotype_to_phenotype(genotype)

        return cls(genotype=genotype, phenotype=phenotype)

    @classmethod
    def random_initialize(cls):
        '''
        Implement the way to generate a random individual,
        usually by generate either a random genotype, or a random phenotype
        '''
        pass

    @property
    def genotype(self):
        return self._genotype

    @property
    def phenotype(self):
        return self._phenotype

    @property
    def fitness(self):
        return self._fitness

    @genotype.setter
    def genotype(self):
        raise NotImplementedError

    @phenotype.setter
    def phenotype(self):
        raise NotImplementedError

    def __eq__(self, another):
        return self.fitness == another.fitness

    def __ne__(self, another):
        return self.fitness != another.fitness

    def __lt__(self, another):
        return self.fitness < another.fitness

    def __gt__(self, another):
        return self.fitness > another.fitness

    def __le__(self, another):
        return self.fitness <= another.fitness

    def __ge__(self, another):
        return self.fitness >= another.fitness

    def __repr__(self):
        return "(Gene:%s; Pheno:%s; Fitness:%.6f)" % \
            (self.genotype, self.phenotype, self.fitness)


class Population(object):
    '''
        Population in each GA iteration
    '''

    INDIVIDUAL_TYPE = None

    def __init__(self, pop_size: int, random: bool):
        self.pop_size = pop_size

        self._individuals = []
        self._best_individual = None

        if random:
            self._random_initialize()

    def __len__(self):
        return len(self._individuals)

    @property
    def individuals(self):
        return self._individuals

    def add_individual(self, new_individual):
        if new_individual:
            self._individuals.append(new_individual)
        else:
            raise ValueError('Cannot add a None individual')

    @property
    def best_individual(self):
        if not self._best_individual:
            self.update_best()

        return self._best_individual
############################################
    def sort_population(self):
      # sort by descending order
      return np.sort(self._individuals).tolist()
    
    def check_duplicate(self, new_individual):
        for indv in self._individuals:
            if (np.abs(new_individual._genotype - indv._genotype)).sum == 0:
                return True
            else:
                return False
    
#     def check_duplicate(self, new_individual):
#         for indv in self._individuals:
#             if np.array_equal(new_individual._genotype,indv._genotype):
#                 return True
#             else:
#                 return False
############################################
    def _find_best(self):
        return max(self._individuals)

    def update_best(self):
        self._best_individual = self._find_best()

    def is_empty(self):
        return len(self) == 0

    def _random_initialize(self):
        assert self.is_empty(), 'Cannot initialize non-empty population'

        for _ in range(self.pop_size):
            ind = self.INDIVIDUAL_TYPE.random_initialize()
            self.add_individual(ind)

    def build_roulette_wheel(self):
        """
        Build a roulette wheel based on current population
        In order to do a roulette wheel selection
        """

        roulette_wheel = np.empty((len(self), ))
        sum_fitness = sum(i.fitness for i in self.individuals)

        prob = 0.0
        for i, ind in enumerate(self.individuals):
            prob += ind.fitness / sum_fitness
            roulette_wheel[i] = prob

        return roulette_wheel

    def roll_roulette_wheel(self, roulette_wheel):
        """
        Roll the roulette wheel to choose individuals
        for a new population
        """
        assert len(roulette_wheel) == len(self), \
            "Invalid wheel length (expect %d, get %d)" % (len(self), len(roulette_wheel))

        prob = np.random.random_sample()
        output = None

        # assume the roulette wheel is strictly increasing,
        # perform a binary search
        left, right = 0, int(len(roulette_wheel)) - 1
        while True:
            mid = int(left + (right - left)/2)

            if roulette_wheel[mid] < prob and prob <= roulette_wheel[mid+1]:
                # found
                output = self._individuals[mid+1]
                break

            if left == 0 and right == 0:
                # special case
                output = self._individuals[0]
                break

            if prob <= roulette_wheel[mid]:
                right = mid
            elif prob > roulette_wheel[mid+1]:
                left = mid+1

        return output

############################################
#     @classmethod
#     def population_selection(cls, current_pop):
#         """
#         Create new population by choosing elite individuals
#         from the current population, using roulette wheel
#         """
#     ## higher fitness implies higher probability of being chosen for breeding
#         pop_size = current_pop.pop_size

#         new_pop = cls(pop_size, random=False)

#         new_pop.add_individual(current_pop.best_individual)
#         roulette_wheel = current_pop.build_roulette_wheel()

#         while len(new_pop) < pop_size:
#             ind = current_pop.roll_roulette_wheel(roulette_wheel)
#             new_pop.add_individual(ind)

#         return new_pop

    @classmethod
    def population_selection(cls, current_pop):
        # choose the best of the population and use them for breeding
        sorted_population = current_pop.sort_population()
        pop_size = current_pop.pop_size
#         q1_pop = int(pop_size/4)
#         q2_pop = int(pop_size/2)
        new_pop = cls(pop_size, random=False)
    
        for indv in sorted_population[-pop_size:]:
            new_pop.add_individual(indv)
        
#         for indv in sorted_population[-q2_pop:]:
#           new_pop.add_individual(indv)

#         # add the most N fit individuals in the parent population
#         N = 1
#         for indv in sorted_population[-N:]:
#             new_pop.add_individual(indv)
#         no_of_new_indv = N
#         counter = N
#         while no_of_new_indv < q2_pop and counter < pop_size-N:
#             selected_indv = sorted_population[-1-counter]
#             if not new_pop.check_duplicate(selected_indv):
#                 new_pop.add_individual(selected_indv)
#                 no_of_new_indv += 1
#             counter += 1
        
#         if no_of_new_indv < q2_pop:
#             for indv in sorted_population[-no_of_new_indv -(q2_pop - no_of_new_indv):-no_of_new_indv]:
#                 new_pop.add_individual(indv)
        return new_pop
############################################

class GAConfig(object):

    def __init__(self, 
                 pop_size,
                 n_generations,
                 crossover_rate,
                 mutation_rate,
                 individual_type,
                 genetic_info,
                 fitness_params,
                 experiment_tag,
                 log_file):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        self.individual_type = individual_type
        self.genetic_info = genetic_info
        self.fitness_params = fitness_params
        
        self.experiment_tag = experiment_tag
        self.log_file = log_file


class GeneticAlgorithm:

    def __init__(self, config):
        self.pop_size = config.pop_size
        self.n_generations = config.n_generations
        self.crossover_rate = config.crossover_rate
        self.mutation_rate = config.mutation_rate

        self.individual_type = config.individual_type
        Population.INDIVIDUAL_TYPE = config.individual_type
        Population.INDIVIDUAL_TYPE.set_individual_attrs(config.genetic_info,
                                                        config.fitness_params)
        self._population = None
        self.experiment_tag = config.experiment_tag
        
        self.log_file = config.log_file

    def do_crossover(self):
        # create new individuals by crossover
        assert self._population, "Population is empty"

        prob = np.random.uniform()
        if prob >= self.crossover_rate:
            return (None, None)

        n = len(self._population)
        i, j = 0, 0
        while i == j:
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)

        parent1 = self._population.individuals[i]
        parent2 = self._population.individuals[j]

        return self.individual_type.crossover(parent1, parent2)

    def do_mutation(self):
        # create new individual by mutation
        assert self._population, "Population is empty"

        if np.random.uniform() >= self.mutation_rate:
            return None

        n = len(self._population)
        i = np.random.randint(0, n)
        parent = self._population.individuals[i]

        return self.individual_type.mutate(parent)

    def breed_new_offsprings(self):
        # expand the population by creating new individuals
        for _ in range(self.pop_size):
#         for _ in range(self._population.pop_size):
            offspring1, offspring2 = self.do_crossover()
            if offspring1 and offspring2:
                self._population.add_individual(offspring1)
                self._population.add_individual(offspring2)

            offspring = self.do_mutation()
            if offspring:
                self._population.add_individual(offspring)

    @property
    def best_individual(self):
        return self._population.best_individual

    @property
    def best_fitness(self):
        return self.best_individual.fitness

    def search(self, verbose=True, 
               # log_dir='./logging'
              ):
        # this is where genetic algorithm run

        # # check the log directory
        # if not os.path.exists(log_dir):
        #     os.mkdir(log_dir)
        logfilename = Path(self.log_file)
        log_file = logfilename.with_suffix('.log')
        with open(log_file, 'w') as f:
            f.write('GA config:\n')
            f.write('Number of generations: %d\n' % self.n_generations)
            f.write('Population size:       %d\n' % self.pop_size)
            f.write('Crossover rate:        %f\n' % self.crossover_rate)
            f.write('Mutation rate:         %f\n' % self.mutation_rate)
            f.write('Experiment Tag:        %s\n' % self.experiment_tag)


        # set print options for logging numpy arrays
        np.set_printoptions(
            precision=6,
            threshold=sys.maxsize,
            linewidth=1<<32,
            suppress=True
        )

        if verbose:
            print('GA: initializing first population')

        # first, initialize the first population
        self._population = Population(self.pop_size, random=True)

        current_best_fitness = self.best_fitness
        n_unimprove = 0
        history = []

        # log info about first population (generation 0)
        with open(log_file, 'a') as f:
            f.write('Generation 0: Chromosome, Fitness\n')
            for ind in self._population.individuals:
                f.write('%s;\t%s\n' % (ind.genotype, ind._fitness))

        if verbose:
            print('GA: start main loop...')

        # main loop
        for generation in range(self.n_generations):
            start_t = time.perf_counter()

            # expand current population
            self.breed_new_offsprings()

            # selection on current expanded population
            self._population = Population.population_selection(self._population)

            # update best individual in this generation
            self._population.update_best()

            if self.best_fitness > current_best_fitness:
                n_unimprove = 0
            else:
                n_unimprove += 1

            current_best_fitness = self.best_fitness
            history.append(current_best_fitness)

            generation_t = time.perf_counter() - start_t

            # log info about this generation
            with open(log_file, 'a') as f:
                f.write("#####################################\n")
                f.write('Generation %d: Chromosome, Fitness\n' % (generation+1))
                for ind in self._population.individuals:
                    f.write('%s;\t%s\n' % (ind.genotype, ind._fitness))

            if verbose:
                print('GA: Generation %d: took %.2f(s)' % (generation+1, generation_t))
                print('\tBest fitness: %f' % (self.best_fitness))
                if n_unimprove > 0:
                    print('\tNo improvement for %d gen(s)' % (n_unimprove))

        return history
