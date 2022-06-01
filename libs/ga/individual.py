import os
import sys
import git
import pathlib

import random
import numpy as np
from bitarray import bitarray, util
import math
from collections import namedtuple

PROJ_ROOT_PATH = pathlib.Path(git.Repo('.', search_parent_directories=True).working_tree_dir)
PROJ_ROOT =  str(PROJ_ROOT_PATH)
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)
    
from libs.ga.genetic_algorithm import BaseIndividual
from libs.fitnessfns import eval_lenet_3hidden_ERRexpbitflips

#########################################################################
class StatisticalFitness(namedtuple('StatisticalFitness', ['mean', 'std'])):

    def __repr__(self):
        return 'Fitness(mean=%.6f, std=%.6f)' % (self.mean, self.std)

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented

        return self.mean == other.mean and self.std == other.std

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if type(other) is not type(self):
            return NotImplemented

        if self.mean < other.mean:
            return True
        elif self.mean == other.mean:
            return self.std > other.std
        else:
            return False

    def __gt__(self, other):
        if type(other) is not type(self):
            return NotImplemented

        if self.mean > other.mean:
            return True
        elif self.mean == other.mean:
            return self.std < other.std
        else:
            return False

    def __le__(self, other):
        if type(other) is not type(self):
            return NotImplemented

        return self.mean <= other.mean

    def __ge__(self, other):
        if type(other) is not type(self):
            return NotImplemented

        return self.mean >= other.mean
#########################################################################

# Shuffling rows of convolutional layer
class INDV_mnist32_cnn_3hidden_c0layer(BaseIndividual):
class INDV_c0layer_lenet_3hidden_ERRexpbitflips(BaseIndividual):

    @classmethod
    def set_individual_attrs(cls, genetic_info, fitness_params):
        GENETIC_INFO_KEYS = ['gene_length' # required to generated shuffled row order
                            ]
        FITNESS_PARAMS_KEYS = ['model', # weights and biases (keras seq object)
                               'error_profile', # error profile
                               'error_param', # value of stuck-at-k
                               'test_set' # test set of images
                              ]

        for key in GENETIC_INFO_KEYS:
            assert key in genetic_info, 'Missing genetic info: %s' % (key)

        for key in FITNESS_PARAMS_KEYS:
            assert key in fitness_params, 'Missing fitness params: %s' % (key)

        cls.GENE_LENGTH = genetic_info['gene_length']
        cls.FITNESS_PARAMS = fitness_params

    @classmethod
    def genotype_to_phenotype(cls, genotype):
        return genotype # the phenotype is the same as genotype

    @classmethod
    def phenotype_to_genotype(cls, phenotype):
        # this method would not be used anyways
        return NotImplemented

    def calculate_fitness(self, phenotype):
        assert self.FITNESS_PARAMS is not None, 'Missing fitness parameter'

        model = self.FITNESS_PARAMS['model']
        test_set = self.FITNESS_PARAMS['test_set']
        error_profile = self.FITNESS_PARAMS['error_profile']
        ERR_PARAM = self.FITNESS_PARAMS['error_param']

        # evaluate the phenotype (row shuffle order) with 
        # each error_profile
        # using the test_set
        shuffle_order = phenotype
        
        mean_acc, std_acc = eval_lenet_3hidden_ERRexpbitflips(model=model,
                                                                error_profile_c0=error_profile,
                                                                error_profile_h0=None,
                                                                error_profile_h1=None,
                                                                error_profile_h2=None,
                                                                error_profile_op=None,
                                                                ERR_PARAM=ERR_PARAM,
                                                                clayer0_shuffle_order=shuffle_order,
                                                                hlayer0_shuffle_order=None,
                                                                hlayer1_shuffle_order=None,
                                                                hlayer2_shuffle_order=None,
                                                                oplayer_shuffle_order=None,
                                                                test_set=test_set)   
        return StatisticalFitness(mean_acc, std_acc)

    @classmethod
    def crossover(cls, parent1, parent2):
        # implementing order crossover (OX)
        gene1 = parent1.genotype.copy()
        gene2 = parent2.genotype.copy()
        l = parent1.GENE_LENGTH

        # choose the cut point for genes
        start, end = 0, 0
        while start >= end:
            # make sure start < end
            start = np.random.randint(0, l)
            end = np.random.randint(0, l)

        new_gene1 = gene1[start:end+1]
        new_gene2 = gene2[start:end+1]

        # mark inplace alleles
        marked1 = np.zeros(l, dtype=np.bool)
        marked2 = np.zeros(l, dtype=np.bool)
        for i in range(start, end+1):
            marked1[gene1[i]] = True
            marked2[gene2[i]] = True

        for i in range(l):
            j = (end + i) % l

            allele1 = gene1[j]
            allele2 = gene2[j]

            if not marked1[allele2]:
                new_gene1 = np.append(new_gene1, allele2)

            if not marked2[allele1]:
                new_gene2 = np.append(new_gene2, allele1)

        new_gene1 = np.roll(new_gene1, start)
        new_gene2 = np.roll(new_gene2, start)

        offspring1 = cls.from_genotype(new_gene1)
        offspring2 = cls.from_genotype(new_gene2)

        return offspring1, offspring2

    @classmethod
    def mutate(cls, parent):
        # implementing random-swap mutation
        new_gene = parent.genotype.copy()
        l = parent.GENE_LENGTH

        i, j = 0, 0
        while i == j:
            # make sure 2 swap position are different
            i = np.random.randint(0, l)
            j = np.random.randint(0, l)

        new_gene[i], new_gene[j] = new_gene[j], new_gene[i]

        offspring = cls.from_genotype(new_gene)

        return offspring

    @classmethod
    def random_initialize(cls):
        random_gene = np.arange(cls.GENE_LENGTH, dtype=np.int32)
        np.random.shuffle(random_gene)
        return cls.from_genotype(random_gene)

    @property
    def fitness(self):
        return self._fitness.mean

    def __repr__(self):
        return 'Gene:%s; Fitness: Mean:%.6f; Std:%.6f' % \
            (self.genotype, self._fitness.mean, self._fitness.std)
#########################################################################

# Shuffling rows of fc_2 layer
class INDV_mnist32_cnn_3hidden_h2layer(BaseIndividual):
class INDV_h2layer_lenet_3hidden_ERRexpbitflips(BaseIndividual):

    @classmethod
    def set_individual_attrs(cls, genetic_info, fitness_params):
        GENETIC_INFO_KEYS = ['gene_length' # required to generated shuffled row order
                            ]
        FITNESS_PARAMS_KEYS = ['model', # weights and biases (keras seq object)
                               'error_profile', # error profile
                               'error_param', # value of stuck-at-k
                               'test_set' # test set of images
                              ]

        for key in GENETIC_INFO_KEYS:
            assert key in genetic_info, 'Missing genetic info: %s' % (key)

        for key in FITNESS_PARAMS_KEYS:
            assert key in fitness_params, 'Missing fitness params: %s' % (key)

        cls.GENE_LENGTH = genetic_info['gene_length']
        cls.FITNESS_PARAMS = fitness_params

    @classmethod
    def genotype_to_phenotype(cls, genotype):
        return genotype # the phenotype is the same as genotype

    @classmethod
    def phenotype_to_genotype(cls, phenotype):
        # this method would not be used anyways
        return NotImplemented

    def calculate_fitness(self, phenotype):
        assert self.FITNESS_PARAMS is not None, 'Missing fitness parameter'

        model = self.FITNESS_PARAMS['model']
        test_set = self.FITNESS_PARAMS['test_set']
        error_profile = self.FITNESS_PARAMS['error_profile']
        ERR_PARAM = self.FITNESS_PARAMS['error_param']

        # evaluate the phenotype (row shuffle order) with 
        # each error_profile
        # using the test_set
        shuffle_order = phenotype
        
        mean_acc, std_acc = eval_lenet_3hidden_ERRexpbitflips(model=model,
                                                                error_profile_c0=None,
                                                                error_profile_h0=None,
                                                                error_profile_h1=None,
                                                                error_profile_h2=error_profile,
                                                                error_profile_op=None,
                                                                ERR_PARAM=ERR_PARAM,
                                                                clayer0_shuffle_order=None,
                                                                hlayer0_shuffle_order=None,
                                                                hlayer1_shuffle_order=None,
                                                                hlayer2_shuffle_order=shuffle_order,
                                                                oplayer_shuffle_order=None,
                                                                test_set=test_set)   
        return StatisticalFitness(mean_acc, std_acc)

    @classmethod
    def crossover(cls, parent1, parent2):
        # implementing order crossover (OX)
        gene1 = parent1.genotype.copy()
        gene2 = parent2.genotype.copy()
        l = parent1.GENE_LENGTH

        # choose the cut point for genes
        start, end = 0, 0
        while start >= end:
            # make sure start < end
            start = np.random.randint(0, l)
            end = np.random.randint(0, l)

        new_gene1 = gene1[start:end+1]
        new_gene2 = gene2[start:end+1]

        # mark inplace alleles
        marked1 = np.zeros(l, dtype=np.bool)
        marked2 = np.zeros(l, dtype=np.bool)
        for i in range(start, end+1):
            marked1[gene1[i]] = True
            marked2[gene2[i]] = True

        for i in range(l):
            j = (end + i) % l

            allele1 = gene1[j]
            allele2 = gene2[j]

            if not marked1[allele2]:
                new_gene1 = np.append(new_gene1, allele2)

            if not marked2[allele1]:
                new_gene2 = np.append(new_gene2, allele1)

        new_gene1 = np.roll(new_gene1, start)
        new_gene2 = np.roll(new_gene2, start)

        offspring1 = cls.from_genotype(new_gene1)
        offspring2 = cls.from_genotype(new_gene2)

        return offspring1, offspring2

    @classmethod
    def mutate(cls, parent):
        # implementing random-swap mutation
        new_gene = parent.genotype.copy()
        l = parent.GENE_LENGTH

        i, j = 0, 0
        while i == j:
            # make sure 2 swap position are different
            i = np.random.randint(0, l)
            j = np.random.randint(0, l)

        new_gene[i], new_gene[j] = new_gene[j], new_gene[i]

        offspring = cls.from_genotype(new_gene)

        return offspring

    @classmethod
    def random_initialize(cls):
        random_gene = np.arange(cls.GENE_LENGTH, dtype=np.int32)
        np.random.shuffle(random_gene)
        return cls.from_genotype(random_gene)

    @property
    def fitness(self):
        return self._fitness.mean

    def __repr__(self):
        return 'Gene:%s; Fitness: Mean:%.6f; Std:%.6f' % \
            (self.genotype, self._fitness.mean, self._fitness.std)
#########################################################################

# Shuffling rows of output layer
class INDV_mnist32_cnn_3hidden_oplayer(BaseIndividual):
class INDV_oplayer_lenet_3hidden_ERRexpbitflips(BaseIndividual):

    @classmethod
    def set_individual_attrs(cls, genetic_info, fitness_params):
        GENETIC_INFO_KEYS = ['gene_length' # required to generated shuffled row order
                            ]
        FITNESS_PARAMS_KEYS = ['model', # weights and biases (keras seq object)
                               'error_profile', # error profile
                               'error_param', # value of stuck-at-k
                               'test_set' # test set of images
                              ]

        for key in GENETIC_INFO_KEYS:
            assert key in genetic_info, 'Missing genetic info: %s' % (key)

        for key in FITNESS_PARAMS_KEYS:
            assert key in fitness_params, 'Missing fitness params: %s' % (key)

        cls.GENE_LENGTH = genetic_info['gene_length']
        cls.FITNESS_PARAMS = fitness_params

    @classmethod
    def genotype_to_phenotype(cls, genotype):
        return genotype # the phenotype is the same as genotype

    @classmethod
    def phenotype_to_genotype(cls, phenotype):
        # this method would not be used anyways
        return NotImplemented

    def calculate_fitness(self, phenotype):
        assert self.FITNESS_PARAMS is not None, 'Missing fitness parameter'

        model = self.FITNESS_PARAMS['model']
        test_set = self.FITNESS_PARAMS['test_set']
        error_profile = self.FITNESS_PARAMS['error_profile']
        ERR_PARAM = self.FITNESS_PARAMS['error_param']

        # evaluate the phenotype (row shuffle order) with 
        # each error_profile
        # using the test_set
        shuffle_order = phenotype
        
        mean_acc, std_acc = eval_lenet_3hidden_ERRexpbitflips(model=model,
                                                                error_profile_c0=None,
                                                                error_profile_h0=None,
                                                                error_profile_h1=None,
                                                                error_profile_h2=None,
                                                                error_profile_op=error_profile,
                                                                ERR_PARAM=ERR_PARAM,
                                                                clayer0_shuffle_order=None,
                                                                hlayer0_shuffle_order=None,
                                                                hlayer1_shuffle_order=None,
                                                                hlayer2_shuffle_order=None,
                                                                oplayer_shuffle_order=shuffle_order,
                                                                test_set=test_set)   
        return StatisticalFitness(mean_acc, std_acc)

    @classmethod
    def crossover(cls, parent1, parent2):
        # implementing order crossover (OX)
        gene1 = parent1.genotype.copy()
        gene2 = parent2.genotype.copy()
        l = parent1.GENE_LENGTH

        # choose the cut point for genes
        start, end = 0, 0
        while start >= end:
            # make sure start < end
            start = np.random.randint(0, l)
            end = np.random.randint(0, l)

        new_gene1 = gene1[start:end+1]
        new_gene2 = gene2[start:end+1]

        # mark inplace alleles
        marked1 = np.zeros(l, dtype=np.bool)
        marked2 = np.zeros(l, dtype=np.bool)
        for i in range(start, end+1):
            marked1[gene1[i]] = True
            marked2[gene2[i]] = True

        for i in range(l):
            j = (end + i) % l

            allele1 = gene1[j]
            allele2 = gene2[j]

            if not marked1[allele2]:
                new_gene1 = np.append(new_gene1, allele2)

            if not marked2[allele1]:
                new_gene2 = np.append(new_gene2, allele1)

        new_gene1 = np.roll(new_gene1, start)
        new_gene2 = np.roll(new_gene2, start)

        offspring1 = cls.from_genotype(new_gene1)
        offspring2 = cls.from_genotype(new_gene2)

        return offspring1, offspring2

    @classmethod
    def mutate(cls, parent):
        # implementing random-swap mutation
        new_gene = parent.genotype.copy()
        l = parent.GENE_LENGTH

        i, j = 0, 0
        while i == j:
            # make sure 2 swap position are different
            i = np.random.randint(0, l)
            j = np.random.randint(0, l)

        new_gene[i], new_gene[j] = new_gene[j], new_gene[i]

        offspring = cls.from_genotype(new_gene)

        return offspring

    @classmethod
    def random_initialize(cls):
        random_gene = np.arange(cls.GENE_LENGTH, dtype=np.int32)
        np.random.shuffle(random_gene)
        return cls.from_genotype(random_gene)

    @property
    def fitness(self):
        return self._fitness.mean

    def __repr__(self):
        return 'Gene:%s; Fitness: Mean:%.6f; Std:%.6f' % \
            (self.genotype, self._fitness.mean, self._fitness.std)
