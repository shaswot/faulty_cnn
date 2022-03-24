#!/bin/bash
#PJM -g gq50
#PJM -L rg=debug 
#PJM -L node=1
#PJM -L elapse=30:00
#PJM -j

module load singularity
REPO=/work/gq50/q50002/faulty_cnn
SIF=/work/gq50/q50002/faulty_cnn_ni.file

cd ${REPO}

singularity exec \
    --bind `pwd` \
    -H ${REPO} \
    ${SIF} \
    python ${REPO}/pyscripts/train_all.py 
