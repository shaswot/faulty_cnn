#!/bin/bash

# create directory for job instance
mkdir -p ${exp_name}--${model_instance}--${error_profile}

# enter the directory
cd ${exp_name}--${model_instance}--${error_profile} 

# create job script
# https://stackoverflow.com/questions/2500436/how-does-cat-eof-work-in-bash
cat << EOF > ${exp_name}--${model_instance}--${error_profile}.job 
#!/bin/bash
#PJM -g gq50
#PJM -L rg=share 
#PJM -L gpu=1
#PJM -L jobenv=singularity
#PJM -L elapse=48:00:00
#PJM -j

module load singularity

cd /work/gq50/q50002/faulty_cnn/ga_scripts/all/lenet_3hidden_ERRexpbitflips/ERR_2

singularity exec \
	--bind `pwd` \
	-H /work/gq50/q50002/faulty_cnn\
	/work/gq50/q50002/faulty_cnn_san.file \
	python ${exp_name}.py ${model_instance} ${error_profile}

EOF

# submit job script
pjsub ${exp_name}--${model_instance}--${error_profile}.job
 
