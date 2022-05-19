FROM tensorflow/tensorflow:2.8.0-gpu-jupyter

MAINTAINER shaswot <shaswot@acsl.ics.keio.ac.jp>

# install system-wide deps for python and node
# The yqq flag is used to suppress output and assumes "Yes" to all prompts.

RUN apt-get -yqq update --fix-missing

RUN apt-get -yqq install \
        screen \
        vim \
        git \
        openssh-server \
        htop \
        wget \
        net-tools \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*


# set a directory for the repos 
RUN mkdir -p /repos/faulty_cnn


# copy files to the container
COPY . /repos/faulty_cnn
COPY ./docker/requirements.txt /tmp/

# install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# folder to share files with host machine
RUN mkdir -p /stash
VOLUME /stash

# folder to store the project
RUN mkdir -p /repos/faulty_cnn
COPY . /repos/faulty_cnn

# define the port number the container should expose
# All running containers expose 8888
# They are redirected to other ports using docker -p <machine_port>:8888
EXPOSE 8888

# Change workdir to root so that Jupyter Lab can access the /stash folder
# /stash folder stores large dumpfiles that can be accessed from the host computer
WORKDIR /

# run bash shell
# start jupyter lab once inside the container using screen
# this allows you to access the bash shell and run multiple screens for multiple notebooks / tensorboards 
RUN /bin/bash
