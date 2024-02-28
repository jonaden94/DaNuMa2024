from datasets import load_dataset
import os

########################### get dataset
base_dir = '/usr/users/henrich1/exercises_summer_school/data/detect_and_segment'
train_path = os.path.join(base_dir, 'train.csv')
val_path = os.path.join(base_dir, 'val.csv')
data_files = {
    'train': train_path,
    'val': val_path
}

dataset = load_dataset("csv", data_files=data_files)

id2label = {0: 'pig'}
label2id = {'pig': 0}



########################### get image_processor
from transformers import AutoImageProcessor, YolosImageProcessor

# checkpoint = "facebook/detr-resnet-50"
# image_processor = AutoImageProcessor.from_pretrained(checkpoint)

checkpoint = "hustvl/yolos-tiny"
image_processor = YolosImageProcessor.from_pretrained(checkpoint)


########################### augment
import albumentations
import numpy as np

transform = albumentations.Compose(
    [
        # albumentations.Resize(400, 640),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightnessContrast(p=0.5),
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
)


########################### transform input from dataset
def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(len(category)):

        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0, # no background class
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations


from PIL import Image
def transform_aug_ann(examples, base_dir):
    images_dir = os.path.join(base_dir, 'images')
    labels_dir = os.path.join(base_dir, 'labels')
    image_ids = examples["image_id"]
    images, bboxes, areas, categories = [], [], [], []
    
    for image_name in examples["image_name"]:
        image = Image.open(os.path.join(images_dir, image_name + '.jpg'))
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        
        bbox = np.loadtxt(os.path.join(labels_dir, image_name + '.txt'))
        if bbox.ndim == 1:
            bbox = bbox[None, :]
        category = [0 for _ in range(len(bbox))] # only one class
        out = transform(image=image, bboxes=bbox, category=category)
        area = np.array(out["bboxes"])[:, 2] * np.array(out["bboxes"])[:, 3]

        areas.append(area)
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, areas, bboxes)
    ]

    return image_processor(images=images, annotations=targets, return_tensors="pt")


dataset['train'] = dataset['train'].with_transform(lambda examples: transform_aug_ann(examples, base_dir))



########################### collate
def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    # batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch






########################### model
from transformers import AutoModelForObjectDetection, AutoModel, YolosModel

model = AutoModelForObjectDetection.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)




########################### train 

from transformers import TrainingArguments

training_args = TrainingArguments(
    # output_dir="detr-resnet-50_finetuned_pigs",
    output_dir="yolo_ft",
    # output_dir="defdetr-resnet-50_finetuned_pigs",
    # output_dir="conddetr-resnet-50_finetuned_pigs",
    per_device_train_batch_size=16,
    num_train_epochs=50,
    fp16=True,
    save_steps=200,
    logging_steps=200,
    learning_rate=1e-4,
    weight_decay=1e-4,
    save_total_limit=20,
    remove_unused_columns=False,
    push_to_hub=False,
)


from transformers import Trainer
# torch.cuda.set_device(7)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=dataset["train"],
    tokenizer=image_processor,
)

# trainer.args._n_gpu = 1
trainer.train()