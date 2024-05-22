Bootstrap: docker
From: nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

%post

addgroup --system --gid 300 slurmadmin
adduser --system --uid 300 --gid 300 --home /opt/slurm/slurm \
        --shell /bin/false slurmadmin

apt-get -y update
apt-get -y install python3.8
apt-get -y install python3-pip
apt-get -y install git

pip3 install --upgrade pip
pip3 install --upgrade setuptools

pip3 install \
    cython \
    gitpython \
    h5py \
    ipykernel \
    jinja2 \
    jupyterlab>=4.0.5 \
    jupyterhub==2.3.1 \
    ipyparallel>=8.6.1 \
    notebook>=7.0.3 \
    matplotlib \
    numpy \
    pandas \
    protobuf \
    pandas \
    six==1.12.0 \
    seaborn \
    tqdm \
    torch==1.12.1 \
    torchvision==0.13.1 \

%environment

PATH=${PATH}:${LSF_BINDIR}:/cm/local/apps/cuda/libs/current/bin
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-10.1/lib64:/cm/local/apps/cuda-driver/libs/current/lib64
CUDA_PATH=/usr/local/cuda-11.3.1
CUDA_ROOT=/usr/local/cuda-11.3.1