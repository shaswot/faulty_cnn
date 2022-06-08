#!/bin/bash

# create directory for job instance
mkdir -p ${model_instance}--${error_lim}

# enter the directory
cd ${model_instance}--${error_lim}

# create job script
# https://stackoverflow.com/questions/2500436/how-does-cat-eof-work-in-bash

log_folder='/work/gq50/q50002/faulty_cnn/logfiles/qent/mnist32-cnn/'${exp_name}
mkdir -p ${log_folder}

cat << EOF > ${exp_name}--${model_instance}--${error_lim}.job 
#!/bin/bash
#PJM -g gq50
#PJM -L rg=share 
#PJM -L gpu=1
#PJM -L jobenv=singularity
#PJM -L elapse=48:00:00
#PJM -j

module load singularity

cd /work/gq50/q50002/faulty_cnn/ga_scripts/qent/mnist32-cnn/

singularity exec \
	--bind `pwd` \
	-H /work/gq50/q50002/faulty_cnn\
	/work/gq50/q50002/faulty_cnn_san.file \
	python ${exp_name}.py ${model_instance} ${error_lim}-2188 > ${log_folder}/${model_instance}--${error_lim}-2188".log" 2>&1 &

singularity exec \
	--bind `pwd` \
	-H /work/gq50/q50002/faulty_cnn\
	/work/gq50/q50002/faulty_cnn_san.file \
	python ${exp_name}.py ${model_instance} ${error_lim}-4981 > ${log_folder}/${model_instance}--${error_lim}-4981".log" 2>&1 &


singularity exec \
	--bind `pwd` \
	-H /work/gq50/q50002/faulty_cnn\
	/work/gq50/q50002/faulty_cnn_san.file \
    python ${exp_name}.py ${model_instance} ${error_lim}-3987 > ${log_folder}/${model_instance}--${error_lim}-3987".log" 2>&1



EOF

# ERR_SEEDS = 2188, 3987, 4981, 6404, 9387

# submit job script
pjsub ${exp_name}--${model_instance}--${error_lim}.job
 
