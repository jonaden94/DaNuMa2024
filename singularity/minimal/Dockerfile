Bootstrap: docker
From: nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

%post
################################################################################# Append custom content to .bashrc using heredoc
addgroup --system --gid 300 slurmadmin
adduser --system --uid 300 --gid 300 --home /opt/slurm/slurm \
        --shell /bin/false slurmadmin

















        # Bootstrap: docker
        # From: nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
        
        # %post
        
        # apt-get -y update
        # apt-get -y install python3.8
        # apt-get -y install python3-pip
        # apt-get -y install git
        
        # pip3 install --upgrade pip
        # pip3 install --upgrade setuptools
        
        # pip3 install \
        #     cython \
        #     gitpython \
        #     h5py \
        #     ipykernel \
        #     jinja2 \
        #     jupyter \
        #     jupyterlab \
        #     matplotlib \
        #     numpy==1.18.5 \
        #     pandas \
        #     protobuf \
        #     sklearn \
        #     pandas \
        #     scikit-image \
        #     scikit-learn \
        #     scipy==1.4.1 \
        #     six==1.12.0 \
        #     seaborn \
        #     tensorflow-gpu \
        #     tqdm \
        #     torch==1.2.0 \
        #     torchvision==0.4.0 \
        
        # %environment
        
        # PATH=${PATH}:${LSF_BINDIR}:/cm/local/apps/cuda/libs/current/bin
        # LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-10.1/lib64:/cm/local/apps/cuda-driver/libs/current/lib64
        # CUDA_PATH=/usr/local/cuda-10.1
        # CUDA_ROOT=/usr/local/cuda-10.1
        