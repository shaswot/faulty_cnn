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
import importlib

PROJ_ROOT_PATH = pathlib.Path(git.Repo('.', search_parent_directories=True).working_tree_dir)
PROJ_ROOT =  str(PROJ_ROOT_PATH)
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)
    
from libs import utils
from libs.utils import segregate_dataset_4Q_mean

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
model_instance = args.model_instance #"mnist32-cnn_1024_256_64-1023"#args.model_instance
error_profile_tag = args.error_profile_tag #"LIM_01-2188"#args.error_profile_tag

# Prepare dataset
# Use only test images
# Prepare dataset
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
all_images = tf.image.resize(all_images, [image_x_size, image_y_size]).numpy()

# Get model
dataset, model_arch, model_config, layer_widths, seed = utils.instancename2metadata(model_instance)
model_meta_type, model_type, model_instance = utils.metadata2instancenames(dataset, model_arch, layer_widths, seed)
model_folder = pathlib.Path(PROJ_ROOT_PATH / "models" / model_type)
model_filename = model_instance + ".h5"
model_file = pathlib.Path(model_folder/ model_filename)

# Load model
model = tf.keras.models.load_model(model_file)

# Load error profile
error_profile_folder = pathlib.Path(PROJ_ROOT_PATH / "error_profiles")
error_profile_filename = error_profile_tag + ".npy"
error_profile_file = pathlib.Path(error_profile_folder/ error_profile_filename)
error_profile = np.load(error_profile_file)
error_lim, error_seed = error_profile_tag.split('-')
# Batchsize for evaluation
EVAL_BATCHSIZE = 128

# Load test set
# Testing with only im_num images
im_num = 128*40#128*40=5120 #[128*78 = 9984]
(qmean1_images, qmean1_labels,
qmean2_images, qmean2_labels,
qmean3_images, qmean3_labels,
qmean4_images, qmean4_labels) = segregate_dataset_4Q_mean(all_images, all_labels, half_size=int(image_x_size/2))

qmean1_images = qmean1_images[-im_num:]
qmean1_labels = qmean1_labels[-im_num:]

qmean2_images = qmean2_images[-im_num:]
qmean2_labels = qmean2_labels[-im_num:]

qmean3_images = qmean3_images[-im_num:]
qmean3_labels = qmean3_labels[-im_num:]

qmean4_images = qmean4_images[-im_num:]
qmean4_labels = qmean4_labels[-im_num:]

test_set = (qmean2_images, 
            qmean2_labels)

error_params = [-1] # types of error to optimize for
layers = ["c0"] # layers to optimize with GA optimization

meta_optim = "mean" #, "ent"
dataset_seg_type = "qmean2" # "all" # type of dataset segregation 
        
for error_param in error_params:
    error_type = "ERR_"+str(error_param)
    print('+'*40)
    print(error_type)
    print('+'*40)
    
    for layer in layers:
        print('~'*30)
        print("Layer: ", layer)
        print('~'*30)

        # RUN Experiment
        # Load GA experiment from module depending on layer name
        ga_func_name = "EXP_mnist32_cnn_ERR_" + layer
        module_name = 'libs.ga.ga_experiments'
        module = importlib.import_module(module_name)
        ga_func = getattr(module, ga_func_name)

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
        dataset, model_arch, model_config, layer_widths, seed = utils.instancename2metadata(model_instance)
        # model_instance = dataset-modelarch_modelconfig-seed
        model_meta_type, model_type, model_instance = utils.metadata2instancenames(dataset, 
                                                                                   model_arch, 
                                                                                   layer_widths, 
                                                                                   seed)

        # model_arch = 'cnn'
        # model_config = '1024_256_64'
        # model_type: 'mnist32-cnn_1024_256_64'
        # model_meta_type: 'mnist32-cnn'

        EXP_TYPE           = dataset_seg_type + "_"  + model_meta_type + "_" + layer + "_" + error_type
        experiment_name    = model_instance   + '--' + error_profile_tag
        ga_experiment_name = dataset_seg_type + "_"  + experiment_name + "--" + error_type + '--' + layer + '--ga_' + str(this_seed)
        # File/Folder to save log files
        logging_folder = pathlib.Path(PROJ_ROOT_PATH / "logging" / dataset_seg_type / model_type / model_instance / error_lim / error_profile_tag / error_type / layer )
        
        if not os.path.exists(logging_folder):
            os.makedirs(logging_folder, exist_ok = True)
        logging_filename_tag = pathlib.Path(logging_folder / ga_experiment_name)

        for i in range(K):
            print('#' * 20)
            print('Experiment number %d' % (i+1))
            # time_tag = time.strftime("%m%d_%H%M%S")
            logging_filename = str(logging_filename_tag)  + '--' + str(i) + '.csv'
            print("Logfile: ",logging_filename)

            experiment = ga_func(model=model,
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
            print()