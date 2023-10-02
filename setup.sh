ENV_NAME="exenv"
conda env remove -n $ENV_NAME
conda create --name $ENV_NAME python=3.10 -y
conda activate $ENV_NAME

# packages
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge tensorboard -y
conda install -c anaconda jupyter -y
conda install -c conda-forge tqdm -y
conda install -c conda-forge torchinfo -y
conda install -c anaconda pandas -y
conda install -c conda-forge matplotlib -y

# install custom package from setup.py
pip install -v -e .

# further packages