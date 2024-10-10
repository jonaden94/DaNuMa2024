# PARTIALLY TAKEN FROM https://github.com/pdebench/PDEBench (MIT LICENSE)

import argparse
from torchvision.datasets.utils import download_url
import os	

BASE_PATH = 'https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/9AIY3V'

class danuma_data:
    files = [
        ["3ZFUWQ", "1_introduction"],
        ["3QVFJT", "3_mlp"],
        ["ARY2XR", "5_weight_regression"],
        ["6ZNSYZ", "6_keypoint_detection"],
        ["GHEB4J", "7_instance_segmentation"],
        ["MWFJN8", "8_natural_language_inference"],
        ["GSULQY", "9_tracking"],

    ]
    
class third_party_data:
    files = [
        
    ]

def download_data(root_folder, dataset_name):
    """ "
    Download data splits specific to a given setting.

    Args:
    root_folder: The root folder where the data will be downloaded
    dataset_name: The name of the dataset to download, must be defined in this python file.  """

    print(f"Downloading data for {dataset_name} ...")

    # Load and parse metadata csv file
    files = get_ids(dataset_name)
    os.makedirs(root_folder, exist_ok=True)

    # Iterate ids and download the files
    for id, name in files:
        url = BASE_PATH + id
        download_url(url, root_folder, name)


if __name__ == "__main__":

    download_data(args.root_folder, args.dataset_name)