# docker build -f Dockerfile --build-arg USER_ID=$UID -t minimax .
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install conda 
WORKDIR /user
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -p /user/miniconda -b
RUN rm /user/Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/user/miniconda/bin:${PATH}

# Create the conda env. Don't change the name as the alias in .bashrc is hardcoded.
RUN conda create -n docker_env python=3.9
# Make python default to env python
ENV PATH=/user/miniconda/envs/docker_env/bin:${PATH}

# Copy the code in the very end
COPY . /user/minimax
WORKDIR /user/minimax
RUN /bin/bash -c ". activate docker_env; python3 -m pip install --upgrade pip; python3 -m pip install -e ."

RUN python3 -m pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# add new user
ARG USER_ID
RUN useradd --shell /bin/bash -u ${USER_ID} -o -d /user user
USER user

# Set python ENV variables
ENV PYTHONUNBUFFERED=1