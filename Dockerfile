# docker build -f Dockerfile -t minimax .
# Base Image
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# Setup basic packages 
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    cmake \
    unzip \
    bzip2 \
    wget \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    libopenmpi-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    ffmpeg \
    xpra \
    libglfw3 \
    xserver-xorg-dev \
    python3.9 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
#     && chmod +x /usr/local/bin/patchelf

# ENV LANG C.UTF-8
WORKDIR /user

# install jax
RUN pip3 install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# RUN pip install minimax-lib

# Copy the code in the very end
COPY src /user/minimax/src
# COPY setup.py /user/metamorph/
# RUN /bin/bash -c ". activate docker_env; cd metamorph; pip install -e ."

# RUN conda init
# RUN /bin/bash -c ". activate docker_env"

# Change permissions
RUN useradd --shell /bin/bash -u `id -u` -o -d /user user
# RUN chown -R user /user/miniconda/envs/docker_env/lib/python3.8/site-packages/mujoco_py/

# Set python ENV variables
# ENV PYTHONUNBUFFERED=1