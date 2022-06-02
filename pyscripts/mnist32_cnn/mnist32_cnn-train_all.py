# train model on MNIST32 using CNN

import os
import sys
import git
import pathlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # don't use GPU
# Using GPU results in different results although the seeds have been set

PROJ_ROOT_PATH = pathlib.Path(git.Repo('.', search_parent_directories=True).working_tree_dir)
PROJ_ROOT =  str(PROJ_ROOT_PATH)
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)
    
from libs import utils, mnist32_cnn
from libs.constants import model_seeds

model_type = "mnist32-cnn_1024_256_64"

# Train models
for seed in model_seeds:
    model_instance = model_type + "-" + str(seed)
    model_file = mnist32_cnn.train_mnist32(model_instance, show_summary=False)
    
