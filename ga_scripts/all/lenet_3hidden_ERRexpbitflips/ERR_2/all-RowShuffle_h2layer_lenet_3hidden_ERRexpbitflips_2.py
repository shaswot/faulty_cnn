import os
import sys
import git
import pathlib

import random
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import argparse

PROJ_ROOT_PATH = pathlib.Path(git.Repo('.', search_parent_directories=True).working_tree_dir)
PROJ_ROOT =  str(PROJ_ROOT_PATH)
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)
    
from libs import utils

# set seed
this_seed = 84376
tf.random.set_seed(this_seed)
np.random.seed(this_seed)
random.seed(this_seed)
os.environ['PYTHONHASHSEED']=str(this_seed)

# Tensorflow Limit GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
# Extract model
parser = argparse.ArgumentParser()
parser.add_argument('model_instance')
parser.add_argument('error_profile_tag')
args= parser.parse_args()
model_instance    = args.model_instance    #"mnist32-cnn_1024_256_64-1023"   
error_profile_tag = args.error_profile_tag #"LIM_01-2188"                 

# Prepare dataset
# Combine test and train images together into one dataset
DATASET_PATH = str(pathlib.Path(PROJ_ROOT_PATH / "datasets" / "mnist.npz" ))
(train_images, train_labels), (test_images, test_labels) = mnist.load_data(path=DATASET_PATH)
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0  

all_images =np.concatenate([train_images, test_images], axis=0)
all_labels =np.concatenate([train_labels, test_labels], axis=0)
all_images = np.expand_dims(all_images, axis=-1)

# resize the input shape , i.e. old shape: 28, new shape: 32
image_x_size = 32
image_y_size = 32
all_images = tf.image.resize(all_images, [image_x_size, image_y_size]) 

# Get model
dataset, model_arch, model_config, layer_widths, seed = utils.instancename2metadata(model_instance)
model_meta_type, model_type, model_instance = utils.metadata2instancenames(dataset, model_arch, layer_widths, seed)
model_folder = pathlib.Path(PROJ_ROOT_PATH / "models" / model_meta_type / model_type)
model_filename = model_instance + ".h5"
model_file = pathlib.Path(model_folder/ model_filename)

# Load model
model = tf.keras.models.load_model(model_file)

# Load error profile
error_profile_folder = pathlib.Path(PROJ_ROOT_PATH / "error_profiles")
error_profile_filename = error_profile_tag + ".npy"
error_profile_file = pathlib.Path(error_profile_folder/ error_profile_filename)
error_profile = np.load(error_profile_file)

# Batchsize for evaluation
EVAL_BATCHSIZE = 128

# Load test set
# Testing with only the last 12_80 images
test_set = (all_images[-12_80:], all_labels[-12_80:])

# No. of times to repeat experiments
K = 3

POP_SIZE = 20
N_GENERATIONS = 100

CR = 0.6
MR = 0.2

print("Population: ", POP_SIZE)
print("Generations: ", N_GENERATIONS)
print("Crossover Rate: ", CR)
print("Mutation Rate: ", MR)

# Run GA experiment
from libs.ga.ga_experiments import RowShuffle_h2layer_lenet_3hidden_ERRexpbitflips

error_param = 2 # TF32 truncation
optim_type = "all" # (q1, q2, q3, q4)
layer = "h2layer"

META_RS = "vanilla" # type of dataset segregation (qent, qmean)
archerr_type = "lenet_3hidden_ERRexpbitflips"
meta_experiment_name = model_instance + '--' + error_profile_tag
RS_experiment_type = optim_type + "-" + "RowShuffle_" + layer + "_" + archerr_type + "_" + str(error_param)
ga_experiment_name = meta_experiment_name + '--ga_' + str(this_seed)

# Folder to save log files
logging_folder = pathlib.Path(PROJ_ROOT_PATH / "logging" / META_RS / archerr_type /  meta_experiment_name / RS_experiment_type )
if not os.path.exists(logging_folder):
    os.makedirs(logging_folder)
    
logging_filename_tag = pathlib.Path(logging_folder / ga_experiment_name)

for i in range(K):
    print('#' * 20)
    print('Experiment number %d' % (i+1))
    # time_tag = time.strftime("%m%d_%H%M%S")
    logging_filename = str(logging_filename_tag)  + '_' + str(i) + '.csv'
    print("Logfile: ",logging_filename)

    experiment = RowShuffle_h2layer_lenet_3hidden_ERRexpbitflips(   model=model,
                                                                    error_profile=error_profile,
                                                                    ERR_PARAM=error_param,
                                                                    test_set=test_set,
                                                                    log_file=logging_filename,
                                                                    pop_size=POP_SIZE,
                                                                    n_generations=N_GENERATIONS,
                                                                    crossover_rate=CR,
                                                                    mutation_rate=MR,
                                                                    experiment_tag=str(i))

    start = time.perf_counter()
    experiment.run()
    t = time.perf_counter() - start
    print('Experiment number %d: took %f(s)' % (i+1, t))