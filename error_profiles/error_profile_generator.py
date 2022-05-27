import os
import sys
import git
import pathlib
import numpy as np

PROJ_ROOT_PATH = pathlib.Path(git.Repo('.', search_parent_directories=True).working_tree_dir)
PROJ_ROOT =  str(PROJ_ROOT_PATH)
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)

from libs.constants import error_seeds

MAX_FAULT_PROB_LIST = [1E-3, 2E-3, 5E-3, 10E-3, 20E-3]

# GPU with many SM Blocks (faulty)
# No SM blocks are repeated during one matmul
GPU_NO_OF_BLOCKS = 20_000
N_THREADS_PER_BLOCK  = 32

for MAX_FAULT_PROB in MAX_FAULT_PROB_LIST:
    for seed in error_seeds:
        rng_ =  np.random.default_rng(seed)
        ERR_PROFILE = rng_.uniform(low=0.0, 
                                        high=MAX_FAULT_PROB,
                                        size = (GPU_NO_OF_BLOCKS,
                                                N_THREADS_PER_BLOCK)).astype(np.float32)
        
        error_tag = "INF_" + f"{int(MAX_FAULT_PROB*1E3):02d}" + "-"
        np.save(error_tag + str(seed), ERR_PROFILE)
    
# GPU with limited SM Blocks (faulty)
# SM blocks are repeated during one matmul
GPU_NO_OF_BLOCKS = 10
N_THREADS_PER_BLOCK  = 32

for MAX_FAULT_PROB in MAX_FAULT_PROB_LIST:
    for seed in error_seeds:
        rng_ =  np.random.default_rng(seed)
        ERR_PROFILE = rng_.uniform(low=0.0, 
                                        high=MAX_FAULT_PROB,
                                        size = (GPU_NO_OF_BLOCKS,
                                                N_THREADS_PER_BLOCK)).astype(np.float32)
        error_tag = "LIM_" + f"{int(MAX_FAULT_PROB*1E3):02d}" + "-"
        TILED_ERR_PROFILE = np.tile(ERR_PROFILE,(20_000//GPU_NO_OF_BLOCKS,1))
        np.save(error_tag + str(seed), TILED_ERR_PROFILE)
