ENV_NAME="exenv"
conda env remove -n $ENV_NAME -y
conda create --name $ENV_NAME python=3.10 -y
conda activate $ENV_NAME

# packages
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
# conda install -c conda-forge tensorboard -y
conda install -c conda-forge tqdm -y
conda install -c conda-forge torchinfo -y
conda install -c conda-forge matplotlib -y
conda install anaconda::scikit-learn -y

# install custom package from setup.py
pip install -r setup/requirements.txt