ENV_NAME="danuma"
conda update -n base -c conda-forge conda -y
conda env remove -n $ENV_NAME -y
conda create --name $ENV_NAME python=3.10 -y
conda activate $ENV_NAME

# packages also in PigDetect
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install -c conda-forge tensorboard -y
conda install mkl=2024.0.0 -y # needs to downgraded because of some bug in other version

# other packages
conda install -c conda-forge tqdm -y
conda install -c conda-forge torchinfo -y
conda install -c conda-forge matplotlib -y
conda install anaconda::scikit-learn -y
conda install jupyter -y
conda install pandas -y

# pip
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org -r setup/requirements.txt
