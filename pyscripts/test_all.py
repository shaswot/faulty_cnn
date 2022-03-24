# test CNN MNIST32 saved models

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

# Evaluate models
for seed in model_seeds:
    model_instance = model_type + "-" + str(seed)
    dataset, model_arch, model_config, layer_widths, seed = utils.instancename2metadata(model_instance)
    model_meta_type, model_type, model_instance = utils.metadata2instancenames(dataset, model_arch, layer_widths, seed)
    
    model_folder = pathlib.Path(PROJ_ROOT_PATH / "models" / model_meta_type / model_type)
    model_filename = model_instance + ".h5"
    model_file = pathlib.Path(model_folder/ model_filename)
    
    [accuracy, conf_matrix] = mnist32_cnn.test_mnist32(model_file, show_summary=False)
    print(f"Model: {model_instance} \t Accuracy:{accuracy*100:0.3f}%")
