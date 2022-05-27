import numpy as np
import os

def logstring2chromosome(logstring):
    return np.array([int(x) for x in logstring.split(";")[0][1:-1].split()])
##############################################################################
#https://stackoverflow.com/questions/46258499/how-to-read-the-last-line-of-a-file-in-python
def get_chromosomes(ga_output_files):
    chromosome_list = []
    fitness_list = []
    for file in ga_output_files:
        with open(file, 'rb') as f:
            try:  # catch OSError in case of a one line file 
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b'\n':
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            last_line = f.readline().decode()
            chromosome, fitness_string = last_line.strip('\n').split(";")
            fitness = float(fitness_string[14:22])
            chromosome_list.append(chromosome)
            fitness_list.append(fitness)

    best_chromosome = logstring2chromosome(chromosome_list[np.argmax(fitness_list)])
    worst_chromosome = logstring2chromosome(chromosome_list[np.argmin(fitness_list)])
    return best_chromosome, worst_chromosome
##############################################################################

