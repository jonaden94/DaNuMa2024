from torchvision.datasets.utils import download_url
import os	
import zipfile
import shutil

BASE_PATH = 'https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/9AIY3V/'

danuma_data = {
    "1_introduction.zip": "3ZFUWQ",
    "3_mlp.zip": "3QVFJT",
    "5_weight_regression.zip": "ARY2XR",
    "6_keypoint_detection.zip": "6ZNSYZ",
    "7_instance_segmentation.zip": "GHEB4J",
    "8_natural_language_inference.zip": "MWFJN8",
    "9_tracking.zip": "GSULQY"
}
    
third_party_data = {
    'sam_vit_h_4b8939.pth': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
    'glove.6B.zip': 'https://nlp.stanford.edu/data/glove.6B.zip',
    'snli_1.0.zip': 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
}

def download_danuma(name, root_dir):
    id = danuma_data[name]
    url = BASE_PATH + id
    download_url(url, root_dir, name)
    
def download_third_party(name, root_dir):
    url = third_party_data[name]
    download_url(url, root_dir)

def unzip(source_file, output_dir):
    zip_file = os.path.join(source_file)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    os.remove(zip_file)

if __name__ == "__main__":
    raw_data_dir = 'data/raw_data'
    output_data_dir = 'data/output_data'
    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(raw_data_dir, exist_ok=True)

    for danuma_name in danuma_data:
        download_danuma(danuma_name, raw_data_dir)
        unzip(os.path.join(raw_data_dir, danuma_name), output_data_dir)
    
    os.makedirs(os.path.join(raw_data_dir, '8_natural_language_inference'), exist_ok=True)
    os.makedirs(os.path.join(output_data_dir, '7_instance_segmentation'), exist_ok=True)
    for third_party_name in third_party_data:
        
        if 'sam' in third_party_name:
            pass
            # download_third_party(third_party_name, os.path.join(raw_data_dir, '7_instance_segmentation'))
        elif 'glove' in third_party_name:
            download_third_party(third_party_name, raw_data_dir)
            unzip(os.path.join(raw_data_dir, third_party_name), os.path.join(raw_data_dir, '8_natural_language_inference'))
            os.remove(os.path.join(raw_data_dir, '8_natural_language_inference', 'glove.6B.50d.txt'))
            os.remove(os.path.join(raw_data_dir, '8_natural_language_inference', 'glove.6B.200d.txt'))
            os.remove(os.path.join(raw_data_dir, '8_natural_language_inference', 'glove.6B.300d.txt'))
            pass
        elif 'snli' in third_party_name:
            download_third_party(third_party_name, raw_data_dir)
            unzip(os.path.join(raw_data_dir, third_party_name), os.path.join(raw_data_dir, '8_natural_language_inference'))
            shutil.move(os.path.join(raw_data_dir, '8_natural_language_inference', 'snli_1.0', 'snli_1.0_dev.jsonl'), os.path.join(raw_data_dir, '8_natural_language_inference'))
            shutil.move(os.path.join(raw_data_dir, '8_natural_language_inference', 'snli_1.0', 'snli_1.0_test.jsonl'), os.path.join(raw_data_dir, '8_natural_language_inference'))
            shutil.move(os.path.join(raw_data_dir, '8_natural_language_inference', 'snli_1.0', 'snli_1.0_train.jsonl'), os.path.join(raw_data_dir, '8_natural_language_inference'))
            shutil.rmtree(os.path.join(raw_data_dir, '8_natural_language_inference', 'snli_1.0'))
            shutil.rmtree(os.path.join(raw_data_dir, '8_natural_language_inference', '__MACOSX'))
                          
        
    