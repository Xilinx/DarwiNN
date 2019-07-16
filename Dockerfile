FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV LC_ALL C

# Linux libraries
RUN apt-get -y update && apt-get -y install --allow-downgrades  --allow-change-held-packages --no-install-recommends make \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libxml2-dev \
    cmake \
    git \
    curl \
    vim \
    wget \
    nano \
    libjpeg-dev \
    libpng-dev \
    gfortran \
    ca-certificates \
    build-essential \
    software-properties-common 

# Install Open MPI
RUN apt-get -y update && apt-get install -y openmpi-bin libopenmpi-dev

# Set default NCCL parameters
RUN echo NCCL_DEBUG=INFO >> /etc/nccl.conf && \
    echo NCCL_SOCKET_IFNAME=^docker0 >> /etc/nccl.conf

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

ENV PYTORCH_VERSION=1.1.0
ENV PYTHON_VERSION=3.6

# Install Python 3.6 on top of Miniconda, all in a single pass
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=${PYTHON_VERSION} 

# Add Conda exe to the PATH
ENV PATH /opt/conda/bin:$PATH

# Conda-level dependecies for Pytorch (from source)
RUN conda install numpy pyyaml scipy ipython mkl mkl-include cython typing && \
    conda clean -ya

RUN conda install -y -c pytorch pytorch=${PYTORCH_VERSION} torchvision && \
    conda clean -ya 

# Install DEAP for the BBO compatibility optimizer
RUN pip install deap scoop