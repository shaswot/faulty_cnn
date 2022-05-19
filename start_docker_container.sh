# START DOCKER CONTAINER
docker run --gpus all \
        -dit \
        -v ~/stash:/stash \
        --name sunday \
        -p 7750:8888 \
        -p 7751:6006 \
	bhootmali/faulty_cnn:san \
	screen -S jlab jupyter lab --no-browser --ip=0.0.0.0 --port 8888 --allow-root --LabApp.token=''
