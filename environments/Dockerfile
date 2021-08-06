# pull a prebuilt image
FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04

# setup timezone
ENV TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    unzip \
    libx11-6 \
    nano \
    wget \
    graphviz \
    libgl1-mesa-glx \
    openssh-server \
 && rm -rf /var/lib/apt/lists/*

# Install other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgtk2.0-0 \
    libcanberra-gtk-module \
    libsm6 libxext6 libxrender-dev \
 && sudo rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

ENV CONDA_DEFAULT_ENV=${project}
ENV CONDA_PREFIX=/root/miniconda3/envs/$CONDA_DEFAULT_ENV
ENV PATH=/root/miniconda3/bin:$CONDA_PREFIX/bin:$PATH
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/root/miniconda3/envs/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64/:$LD_LIBRARY_PATH

# install python 3.6
RUN conda install python=3.6
RUN alias python='/root/miniconda3/envs/bin/python3.6 '

# install pytorch
RUN python -m pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# install tensorflow
RUN python -m pip install tensorflow-gpu==2.4.1

# install other pip requirements
COPY pip-extra.txt /tmp/pip-extra.txt
RUN python -m pip install -r /tmp/pip-extra.txt

# libraries path
RUN ln -s /usr/local/cuda/lib64/libcusolver.so.11 /usr/local/cuda/lib64/libcusolver.so.10