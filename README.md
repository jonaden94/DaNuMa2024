## General information
![combined_image](https://github.com/user-attachments/assets/22de7f10-d7cb-42c1-be5e-0d31360dba24)

This repository contains the exercises and solutions of the DaNuMa 2024 summer school "machine learning in animal science". The data required to run the exercises can be found [here](https://doi.org/10.25625/9AIY3V). However, this repository also contains functionality to automatically download all data, so you do not have to manually download it to run the exercises. This repository can be used in two ways:

1\) You can install the required packages and download all required data to your own machine. This is the recommended way in the long-term if you want to modify and extend research code, but it requires a linux-based machine with access to GPUs (usually some kind of compute server) for faster computation. Instructions for installation and data download are given below under **Using the repository on your own machine**

2\) Exercises 1-6 can also be run without any manual setup using Google Colab. This is a convenient way to try out methods and for smaller projects. It does not require any access to a GPU server since all resources are provided by Google. Instructions for how to run exercises 1-6 using Google Colab are given below under **Using the repository with Google Colab**. Unfortunately, exercises 7-9 cannot be run on Colab since they require the manual installation of some packages which did not work on Colab.

If you have any questions about the repository or if you encounter errors/difficulties in the setup process, the exercises or when using Colab, feel free to open an issue directly here on GitHub or write me a mail.

## Using the repository on your own machine
To get this repository running on your own machine, you need to have *git* and *conda* installed. Furthermore, the setup has only been tested on linux-based machines. If these requirements are met, you first have to clone the repository to a location of your choice:
```
git clone https://github.com/jonaden94/DaNuMa2024
```
Then, navigate to the root directory of the repository, make sure that conda is activated, and run the following command to create a new conda environment called *danuma* with all required packages:
```
source setup/setup.sh
```
Note that this setup takes a while since many packages need to be installed. It will also clone and install the PigDetect repository, which is required for the exercises, to the same location as the DaNuMa2024 repository. If the environment is set up successfully, you will need to download the data required to work on the exercises. For this, you once again need to be in the root directory of the repository, and then run the following download command:
```
python tools/download_data.py
```
The download might take a couple of minutes. If it is finished, everything should be properly set up to run the exercises. To work on an exercise, you need to open the notebook and select the kernel of the *danuma* environment. You can then just start running the cells of the notebook since all data should already be in the appropriate locations. Remember to set `colab = False` at the start of the notebooks. It should be noted that setting up conda environments can sometimes be cumbersome since you might need to adjust some installation commands depending on your specific system parameters.

## Using the repository with Google Colab
Once again, exercises 7-9 unfortunately do not work on Colab. Running the other exercises on Colab is relatively simple. Make sure that you have a Google account and then go to the Google Colab website:
```
https://colab.research.google.com/
```
You should be immediately prompted to open a notebook. Opening a notebook also works by selecting file &#8594; open notebook. In the menu that opens you can then select "GitHub". You should then copy the link to the DaNuMa2024 repository in the field that is shown:
```
https://github.com/jonaden94/DaNuMa2024
```
All notebooks from the repository should then be shown and you can simply select which notebook you want to run. To run the notebook on a GPU, you still have to change one thing. Select runtime &#8594; change runtime and then select a T4 GPU. In the free version of Colab, you have limited computational resources. You can also connect to your own GitHub repositories with Colab (for example your own version of the summer school repository). In that case you can directly save changes that you make to your notebooks as commits, which might be a convenient way to update your projects.

## Acknowledgements
This work was funded with NextGenerationEU funds from the European Union by the Federal Ministry of Education and Research under the funding code 16DKWN038. The responsibility for the content of this publication lies with the authors.

![329264093-812c4f7b-13fd-4493-8347-3be8d50c22b4](https://github.com/user-attachments/assets/9544edfb-2dfd-4c5c-8550-a0fb4294561d)
