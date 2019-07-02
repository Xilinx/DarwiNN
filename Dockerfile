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
RUN apt-get install -y openmpi-bin

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

# Install PDT, for TAU
#get and install PDT
RUN wget https://www.cs.uoregon.edu/research/tau/pdt_releases/pdtoolkit-3.25.1.tar.gz && tar -xf pdtoolkit-3.25.1.tar.gz && \
    cd pdtoolkit-3.25.1/ && ./configure -prefix=/usr/local && make clean install && \
    cd ../ && rm -rf pdtoolkit*

RUN apt-get install -y libopenmpi-dev

# Install TAU
RUN wget https://www.cs.uoregon.edu/research/tau/tau_releases/tau-2.28.1.tar.gz && \ 
    tar -xf tau-2.28.1.tar.gz && \
    PYTHON_INCLUDE_PATH=$(python -c "from sysconfig import get_paths as gp; print(gp()[\"include\"])"); \
    PYTHON_LIB_PATH=/opt/conda/lib/python$PYTHON_VERSION; \
    cd tau-2.28.1/ && \
    ./configure -pdt=/usr/local -prefix=/usr/local -bfd=download -pythoninc=$PYTHON_INCLUDE_PATH -pythonlib=$PYTHON_LIB_PATH -mpi -opari -cuda=/usr/local/cuda-10.0 && \
    make clean install && \
    cd ../ && rm -rf tau*

# Export TAU paths
ENV PATH="/usr/local/x86_64/bin:${PATH}"
ENV PYTHONPATH="/usr/local/x86_64/lib/bindings-mpi-python:${PYTHONPATH}"

# Install DEAP for the BBO compatibility optimizer
RUN pip install deap