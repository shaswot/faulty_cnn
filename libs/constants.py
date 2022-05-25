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
if not model_seeds_file.is_file():
    from seedfiles.seedgen import model_seeds, error_seeds
model_seeds = np.loadtxt(model_seeds_file, dtype=np.uintc).tolist()
    

# load error_profile seeds
error_seeds_filename = "error_seeds.dat"
error_seeds_file = pathlib.Path(SEEDS_FOLDER / error_seeds_filename)
if not error_seeds_file.is_file():
    from seedfiles.seedgen import model_seeds, error_seeds
error_seeds = np.loadtxt(error_seeds_file, dtype=np.uintc).tolist()