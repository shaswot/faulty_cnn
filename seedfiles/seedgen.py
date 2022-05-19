import os
import sys
import git
import pathlib
PROJ_ROOT_PATH = pathlib.Path(git.Repo('.', search_parent_directories=True).working_tree_dir)
PROJ_ROOT =  str(PROJ_ROOT_PATH)
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)
    


import numpy as np

SEEDS_FOLDER = pathlib.Path(PROJ_ROOT_PATH / "seedfiles")
pathlib.Path(SEEDS_FOLDER).mkdir(parents=True, exist_ok=True)

NO_OF_SEEDS = 5
rng = np.random.default_rng(27)

# model seeds
print("Generating model seeds")
model_seeds = sorted(rng.integers(low=1000, 
                        high=9999, 
                        size=NO_OF_SEEDS))
model_seeds_filename = "model_seeds.dat"
model_seeds_file = pathlib.Path(SEEDS_FOLDER / model_seeds_filename)
np.savetxt(model_seeds_file, model_seeds, fmt="%d")

# error_profile seeds
print("Generating error seeds")
error_seeds = sorted(rng.integers(low=1111, 
                        high=9999, 
                        size=NO_OF_SEEDS))
error_seeds_filename = "error_seeds.dat"
error_seeds_file = pathlib.Path(SEEDS_FOLDER / error_seeds_filename)
np.savetxt(error_seeds_file, error_seeds, fmt="%d")