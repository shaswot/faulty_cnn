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

# load model seeds
model_seeds_filename = "model_seeds.dat"
model_seeds_file = pathlib.Path(SEEDS_FOLDER / model_seeds_filename)
if model_seeds_file.is_file():
    model_seeds = np.loadtxt(model_seeds_file, dtype=np.uintc).tolist()
else:
    from seedfiles.seedgen import model_seeds, error_seeds
    

# load error_profile seeds
error_seeds_filename = "error_seeds.dat"
error_seeds_file = pathlib.Path(SEEDS_FOLDER / error_seeds_filename)
if error_seeds_file.is_file():
    error_seeds = np.loadtxt(error_seeds_file, dtype=np.uintc).tolist()
else:
    from seedfiles.seedgen import model_seeds, error_seeds