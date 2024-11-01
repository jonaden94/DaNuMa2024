## General information

This repository contains the exercises and solutions of the DaNuMa 2024 summer school 'machine learning in animal science'. It also contains functionality to automatically download all data required to run the exercises. The repository can be used in two ways:

1\) You can install the required packages and download all required data to your own machine. This is the recommended way in the long-term if you want to modify and extend research code, but it requires a linux-based machine with access to GPUs (usually some kind of compute server). Instructions for installation and data download are given [here](#using-the-repository-on-your-own-machine)

2\) Exercises 1-6 can also be run without any manual setup using Google Colab. This is a convenient way to try out methods and for smaller projects. It does not require any access to a GPU server since all resources are provided by Google. Instructions for how to run exercises 1-6 using Google Colab are given [here](#using-the-repository-with-google-colab). Unfortunately, exercises 7-9 canno be run on Colab since they require the manual installation of some packages which did not work on Colab.

If you have any questions about the repository or if you encounter an error in the setup process or the exercises, feel free to open an issue directly here on github or write me a mail.

## Using the repository on your own machine
To get this repository running on your own machine, you need to have *git* and *conda* installed. Furthermore, the setup has only been tested on linux-based machines. If these requirements are met, you first have to clone the repository to a location of your choice:
```
git clone https://github.com/jonaden94/DaNuMa2024
```
Then, navigate to the root directory of the repository, make sure that conda is activated, and run the following command to create a new conda environment called *danuma* with all required packages:
```
source setup/setup.sh
```
Note that this setup will also clone and install the PigDetect repository, which is required for the exercises, to the same location as the DaNuMa2024 repository. Finally, you will need to download the data required to work on the exercises. For this, you once again need to be in the root directory of the repository, and then run the following download command:
```
python tools/download_data.py
```
The download might take a couple of minutes. If it is finished everything, should be properly set up to run the exercises. To work on an exercise, you need to open the notebook and select the kernel of the *danuma* environment. You can then just start running the cells of the notebook since all data should already be in the appropriate locations! It should be noted that setting up conda environments can sometimes be hard since you might need to adjust some installation commands depending on your specific system parameters. Feel free to contact me in case of issues.

## Using the repository with Google Colab










<!-- 


All commands should be run while being in the TreeLearn root directory. In the following, we explain how to run the segmentation pipeline on our benchmark dataset L1W. Running the segmentation pipeline on a custom forest point cloud works analogously. You can change the configuration of running the pipeline by editing the configuration file located at ``configs/pipeline/pipeline.yaml``. However, the default configuration should be adequate for the majority of cases.

*1\) Download pre-trained model weights and L1W forest point cloud*
```
python tree_learn/util/download.py --dataset_name model_weights_diverse_training_data --root_folder data/model_weights
```
```
python tree_learn/util/download.py --dataset_name benchmark_dataset --root_folder data/pipeline/L1W/forest
```

*2\) Prepare forest point cloud to be segmented* (This is already fulfilled for L1W)
* The forest point cloud must be provided as a las, laz, npy, npz or a space-delimited txt file. 
* The coordinates must be provided in meter scale and have a minimum resolution of one point per (0.1 m)<sup>3</sup>. It is especially important that the trunks of the trees have a sufficiently high resolution.
* Terrain and low-vegetation points must still be part of the point cloud. Only rough noise filtering has to be performed in advance (e.g. to remove scanned particles in the air). See L1W as an example.
* The point cloud file must be placed in ``data/pipeline/L1W/forest``
* Change the argument 'forest_path' in the pipeline configuration at ``configs/pipeline/pipeline.yaml`` to ``data/pipeline/L1W/forest/L1W.laz``
* We strongly recommend retaining a buffer around the point cloud that is of interest. E.g. for an area of interest of 100 m x 100 m, retain a buffer of ~13.5 m to each side so that input is 127 m x 127 m.
* The pipeline automatically removes the buffer which is only needed as context for network prediction. The xy-shape of the point cloud does not have to be square. Arbitrary shapes are allowed.

*3\) Run segmentation pipeline*
* To execute the segmentation pipeline, run the following command:
```
python tools/pipeline/pipeline.py --config configs/pipeline/pipeline.yaml
```


## Training

Here we explain how to train your own networks for semantic and offset prediction using the automatically segmented point clouds introduced in the paper. Training the network on custom forest point clouds works analogously.

*1\) Download training/validation point clouds and pretrained model weights*
```
python tree_learn/util/download.py --dataset_name automatically_segmented_data --root_folder data/train/forests
```
```
python tree_learn/util/download.py --dataset_name benchmark_dataset --root_folder data/val/forest
```
* Download the pretrained model weights provided by [SoftGroup](https://drive.google.com/file/d/1FABsCUnxfO_VlItAzDYAwurdfcdK-scs/view?usp=sharing). Save the file to ``data/model_weights/hais_ckpt_spconv2.pth``.

*2\) Generate training crops for semantic and offset prediction*
* The forest point clouds from which to generate training data must fulfil the same resolution and noise filtering requirement as in the segmentation pipeline.
* Additionally, the point clouds must contain individual tree and semantic labels. We recommend you to provide the labels as part of .las or .laz files, in which case you need to adhere to the labeling scheme proposed by [this paper](https://doi.org/10.48550/arXiv.2309.01279). See also the description of our [dataset](https://doi.org/10.25625/VPMPID).
* Alternatively, you can provide the point clouds as .npy or .txt files where the first three columns are the x, y and z coordinates and the last column is the label. In this case, unclassified points should be labeled as -1, non-tree points should be labeled as 0, and trees should be labeled starting from 1. Unclassified points are ignored during training.
* To generate random crops from the forest point clouds, run the following command. Please note that generating 25000 random crops as training data takes up a large amount of space (~800Gb). You can adjust the number of crops to be generated in the configuration file.
```
python tools/data_gen/gen_train_data.py --config configs/data_gen/gen_train_data.yaml
```

*3\) Generate validation data for semantic and offset prediction:*
* The forest point cloud used to generate validation data must fulfil the same properties as for the training data.
* To generate tiles used for validation, run the following command:
```
python tools/data_gen/gen_val_data.py --config configs/data_gen/gen_val_data.yaml
```

*4\) Train the network for semantic and offset prediction with the following command:*
```
python tools/training/train.py --config configs/training/train.yaml
```


## Evaluation on benchmark dataset

To evaluate the performance of an arbitrary segmentation method on the benchmark dataset in the same way as in the paper, you need to do the following:

*1\) Download benchmark dataset in voxelized form and evaluated_trees.txt*
```
python tree_learn/util/download.py --dataset_name benchmark_dataset_voxelized --root_folder data/benchmark
```
```
python tree_learn/util/download.py --dataset_name evaluated_trees --root_folder data/extra
```


*2\) Obtain prediction results on the benchmark dataset with an arbitrary method*
* It is ok if the method does not exactly return the same coordinates as in the voxelized benchmark dataset, e.g. different number of points is allowed.
* It only has to be ensured that the coordinates are not in some way shifted because the predictions are propagated to the benchmark dataset using k-nearest-neighbors
* Predictions only have to be made for the inner part of the benchmark dataset but it is also ok if predictions are made for the outer parts
* The prediction results can be supplied as a .las or .laz file with the same labeling scheme as in [this paper](https://doi.org/10.48550/arXiv.2309.01279). See also the description of our [dataset](https://doi.org/10.25625/VPMPID).
* Alternatively, you can provide the prediction results as a .npy or .txt file where the first three columns are the x, y and z coordinates and the last column is the label. In this case, non-tree points should be labeled as 0 and trees should be labeled starting from 1.
* Change the argument 'pred_forest_path' in the evaluate_benchmark configuration at ``configs/evaluation/evaluate_benchmark.yaml`` to where your predicted forest is located.

*3\) Run the evaluation and inspect the evaluation results*
```
python tools/evaluation/evaluate_benchmark.py --config configs/evaluation/evaluate_benchmark.yaml
```
* To take a look at the evaluation results, we prepared a notebook that can be found at ``tools/evaluation/evaluate_benchmark.ipynb``


## Acknowledgements

This work was funded with NextGenerationEU funds from the European Union by the Federal Ministry of Education and Research under the funding code 16DKWN038. The responsibility for the content of this publication lies with the authors. -->
