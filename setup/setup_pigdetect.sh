############## 2) get further required packages
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org -U openmim
mim install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org --trusted-host download.openmmlab.com mmengine
mim install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org --trusted-host download.openmmlab.com "mmcv>=2.0.0rc4,<2.2.0"
# mim install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org "mmcv>=2.0.0"

############## 3) initialize submodules to the commits specified in this repo
############## mmdetection commit hash: cfd5d3a985b0249de009b67d04f37263e11cdf3d
############## mmyolo commit hash: 8c4d9dc503dc8e327bec8147e8dc97124052f693
git submodule update --init --recursive

####### optionally update submodules if newer versions of mmdetection or mmyolo are available with new detection models (has not been tested and might require debugging)
# cd mmdetection
# git fetch
# git checkout main
# git pull
# cd ..

# cd mmyolo
# git fetch
# git checkout main
# git pull
# cd ..

############## 4) build mmdetection, mmyolo and detection_utils
cd mmdetection
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org -v -e .
cd ..

cd mmyolo
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org -r requirements/albu.txt
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org albumentations==1.3.1 # this downgrade is necessary to avoid conflicts with mmyolo
mim install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org --trusted-host download.openmmlab.com -v -e .
cd ..

pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org -v -e .