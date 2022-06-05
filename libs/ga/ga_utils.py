import os
import sys
import git
import pathlib

import numpy as np
import glob

PROJ_ROOT_PATH = pathlib.Path(git.Repo('.', search_parent_directories=True).working_tree_dir)
PROJ_ROOT =  str(PROJ_ROOT_PATH)
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)
    
from libs.utils import metadata2instancenames, instancename2metadata

##############################################################################
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
def extract_best_worst_chromosomes(dataset_seg_type, # (all, qmean1, qent2)
                                   layer, 
                                   error_param, 
                                   model_instance, 
                                   error_profile_tag, 
                                   this_seed):
    
    dataset, model_arch, model_config, layer_widths, seed = instancename2metadata(model_instance)
    # model_instance = dataset-modelarch_modelconfig-seed
    model_meta_type, model_type, model_instance = metadata2instancenames(dataset, 
                                                                               model_arch, 
                                                                               layer_widths, 
                                                                               seed)
    # model_meta_type: 'mnist32-cnn'
    error_type = "ERR_"+str(error_param)

    error_lim, error_seed = error_profile_tag.split('-')

    EXP_TYPE           = dataset_seg_type + "_"  + model_meta_type + "_" + layer + "_" + error_type
    experiment_name    = model_instance   + '--' + error_profile_tag
    ga_experiment_name = dataset_seg_type + "_"  + experiment_name + "--" + error_type + '--' + layer + '--ga_' + str(this_seed)
    # File/Folder to load log files
    logging_folder = pathlib.Path(PROJ_ROOT_PATH / "logging" / dataset_seg_type / model_type / model_instance / error_lim / error_profile_tag / error_type / layer )
    logging_filename_tag = pathlib.Path(logging_folder / ga_experiment_name)

    # ga outputs
    ga_output_files = glob.glob(str(logging_filename_tag) + "*.log")

    # get chromosomes strings
    best_chromosome, worst_chromosome = get_chromosomes(ga_output_files)
    
    return(best_chromosome, worst_chromosome)
#############################################################################################