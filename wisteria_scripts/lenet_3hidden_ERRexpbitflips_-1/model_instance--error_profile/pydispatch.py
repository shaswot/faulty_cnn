import subprocess
import os
import sys
import git
import pathlib

PROJ_ROOT_PATH = pathlib.Path(git.Repo('.', search_parent_directories=True).working_tree_dir)
PROJ_ROOT =  str(PROJ_ROOT_PATH)
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)
    

model_instance = "mnist32-cnn_1024_256_64-1023"
error_profile  = "LIM_01-3987"

exp_script_list = [ 
                    "all-RowShuffle_c0layer_lenet_3hidden_ERRexpbitflips_-1",
                    "all-RowShuffle_h2layer_lenet_3hidden_ERRexpbitflips_-1",
                  ]

for script_name in exp_script_list:
    cmd_script = "exp_name="+script_name + " model_instance="+model_instance + " error_profile="+error_profile +" bash dispatch.sh"
    print(cmd_script)
    subprocess.call(cmd_script, shell=True)
    
