import os
import sys
import git
import pathlib
PROJ_ROOT_PATH = pathlib.Path(git.Repo('.', search_parent_directories=True).working_tree_dir)
PROJ_ROOT =  str(PROJ_ROOT_PATH)
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)
    
import numpy as np
SEEDS_FOLDER = pathlib.Path(PROJ_ROOT_PATH / "seedfiles" )
pathlib.Path(SEEDS_FOLDER).mkdir(parents=True, exist_ok=True)

##################################
# model seeds
##################################
# model seed PATHS
model_seeds_filename = "model_seeds.dat"
model_seeds_file = pathlib.Path(SEEDS_FOLDER / model_seeds_filename)
    
# Generate model seeds
def generate_model_seeds(root_seed=20230403,
                        NO_OF_SEEDS=5):
    rng = np.random.default_rng(root_seed)
    print("Generating model seeds")
    model_seeds = sorted(rng.integers(low=1000, 
                            high=9999, 
                            size=NO_OF_SEEDS))
    
    np.savetxt(model_seeds_file, model_seeds, fmt="%d")
    
# load model seeds
def load_model_seeds():
    if not model_seeds_file.is_file():
        generate_model_seeds()
    model_seeds = np.loadtxt(model_seeds_file, dtype=np.uintc).tolist()
    return model_seeds
##################################