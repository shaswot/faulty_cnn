import os
import sys
import git
import pathlib

import random
import numpy as np
from bitarray import bitarray, util
import math
from collections import namedtuple

import csv
import time

PROJ_ROOT_PATH = pathlib.Path(git.Repo('.', search_parent_directories=True).working_tree_dir)
PROJ_ROOT =  str(PROJ_ROOT_PATH)
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)
    
from libs.ga.genetic_algorithm import GAConfig, GeneticAlgorithm
from libs.ga.individual import StatisticalFitness

from libs.ga.individual import INDV_c0layer_lenet_3hidden_ERRexpbitflips
from libs.ga.individual import INDV_h2layer_lenet_3hidden_ERRexpbitflips
from libs.ga.individual import INDV_oplayer_lenet_3hidden_ERRexpbitflips


#########################################################################
# Shuffling rows of convolutional layer
class RowShuffle_c0layer_lenet_3hidden_ERRexpbitflips:
    def __init__(self,
                 model,
                 error_profile,
                 ERR_PARAM,
                 test_set,
                 log_file: str,
                 pop_size: int = 10,
                 n_generations: int = 100,
                 crossover_rate: float = 0.25,
                 mutation_rate: float = 0.01,
                 experiment_tag: str = time.strftime("%m%d_%H%M%S")):

        if not log_file.endswith('.csv'):
            raise ValueError('Log file should be .csv file')

        self.model = model
        self.error_profile = error_profile
        self.ERR_PARAM = ERR_PARAM
        self.test_set = test_set
        self.log_file = log_file
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.experiment_tag = experiment_tag

    def run(self):
        genetic_info = {
            'gene_length': self.model.layers[0].output_shape[-1]
             # no. of rows of (transposed) weight matrix of output
             # for convolution layer, the no. of rows = no. of filter kernels (i.e. output channels)
        }

        fitness_params = {
            'model': self.model,
            'error_profile': self.error_profile,
            'error_param': self.ERR_PARAM,
            'test_set': self.test_set
        }
    
        self.run_GA_evaluation(
            individual_type=INDV_c0layer_lenet_3hidden_ERRexpbitflips,
            genetic_info=genetic_info,
            fitness_params=fitness_params
        )

    def run_GA_evaluation(self, individual_type, genetic_info, fitness_params):
        # initialize and run GA evaluation

        config = GAConfig(
            pop_size=self.pop_size,
            n_generations=self.n_generations,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
            individual_type=individual_type,
            genetic_info=genetic_info,
            fitness_params=fitness_params,
            experiment_tag=self.experiment_tag,
            log_file = self.log_file
        )

        print('Start running GA...')
        ga = GeneticAlgorithm(config)
        history = ga.search(verbose=True)

        # write best finess history to a csv file
        print('Write result to csv file...')
        with open(self.log_file, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Generation', 'Accuracy'])
            for i, acc in enumerate(history):
                writer.writerow([i+1, acc])

        print('Done')
#########################################################################
# Shuffling rows of fc_2 layer
class RowShuffle_h2layer_lenet_3hidden_ERRexpbitflips:
    def __init__(self,
                 model,
                 error_profile,
                 ERR_PARAM,
                 test_set,
                 log_file: str,
                 pop_size: int = 10,
                 n_generations: int = 100,
                 crossover_rate: float = 0.25,
                 mutation_rate: float = 0.01,
                 experiment_tag: str = time.strftime("%m%d_%H%M%S")):

        if not log_file.endswith('.csv'):
            raise ValueError('Log file should be .csv file')

        self.model = model
        self.error_profile = error_profile
        self.ERR_PARAM = ERR_PARAM
        self.test_set = test_set
        self.log_file = log_file
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.experiment_tag = experiment_tag

    def run(self):
        genetic_info = {
            'gene_length': self.model.layers[5].output_shape[-1]
             # no. of rows of (transposed) weight matrix of output
        }

        fitness_params = {
            'model': self.model,
            'error_profile': self.error_profile,
            'error_param': self.ERR_PARAM,
            'test_set': self.test_set
        }
    
        self.run_GA_evaluation(
            individual_type=INDV_h2layer_lenet_3hidden_ERRexpbitflips,
            genetic_info=genetic_info,
            fitness_params=fitness_params
        )

    def run_GA_evaluation(self, individual_type, genetic_info, fitness_params):
        # initialize and run GA evaluation

        config = GAConfig(
            pop_size=self.pop_size,
            n_generations=self.n_generations,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
            individual_type=individual_type,
            genetic_info=genetic_info,
            fitness_params=fitness_params,
            experiment_tag=self.experiment_tag,
            log_file = self.log_file
        )

        print('Start running GA...')
        ga = GeneticAlgorithm(config)
        history = ga.search(verbose=True)

        # write best finess history to a csv file
        print('Write result to csv file...')
        with open(self.log_file, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Generation', 'Accuracy'])
            for i, acc in enumerate(history):
                writer.writerow([i+1, acc])

        print('Done')
#########################################################################
# Shuffling rows of output layer
class RowShuffle_oplayer_lenet_3hidden_ERRexpbitflips:
    def __init__(self,
                 model,
                 error_profile,
                 ERR_PARAM,
                 test_set,
                 log_file: str,
                 pop_size: int = 10,
                 n_generations: int = 100,
                 crossover_rate: float = 0.25,
                 mutation_rate: float = 0.01,
                 experiment_tag: str = time.strftime("%m%d_%H%M%S")):

        if not log_file.endswith('.csv'):
            raise ValueError('Log file should be .csv file')

        self.model = model
        self.error_profile = error_profile
        self.ERR_PARAM = ERR_PARAM
        self.test_set = test_set
        self.log_file = log_file
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.experiment_tag = experiment_tag

    def run(self):
        genetic_info = {
            'gene_length': self.model.layers[-1].output_shape[-1]
             # no. of rows of (transposed) weight matrix of output
        }

        fitness_params = {
            'model': self.model,
            'error_profile': self.error_profile,
            'error_param': self.ERR_PARAM,
            'test_set': self.test_set
        }
    
        self.run_GA_evaluation(
            individual_type=INDV_oplayer_lenet_3hidden_ERRexpbitflips,
            genetic_info=genetic_info,
            fitness_params=fitness_params
        )

    def run_GA_evaluation(self, individual_type, genetic_info, fitness_params):
        # initialize and run GA evaluation

        config = GAConfig(
            pop_size=self.pop_size,
            n_generations=self.n_generations,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
            individual_type=individual_type,
            genetic_info=genetic_info,
            fitness_params=fitness_params,
            experiment_tag=self.experiment_tag,
            log_file = self.log_file
        )

        print('Start running GA...')
        ga = GeneticAlgorithm(config)
        history = ga.search(verbose=True)

        # write best finess history to a csv file
        print('Write result to csv file...')
        with open(self.log_file, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Generation', 'Accuracy'])
            for i, acc in enumerate(history):
                writer.writerow([i+1, acc])

        print('Done')