import torch
from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from pathlib import Path
import json
import os
from PIL import Image
from torch.utils.data import Dataset

BASE_DIR = Path(__file__).resolve().parent.parent
# Change this to where you have saved the BCCD dataset locally
DATA_SET_PATH = "../BCCD_Dataset-master/BCCD/JPEGImages"

# ---------------- Training ----------------

# Paths
TRAIN_JSON = str (BASE_DIR / "coco_Data/instances_train.json")
VAL_JSON = str (BASE_DIR / "coco_Data/instances_val.json")
IMAGE_ROOT = DATA_SET_PATH

class COCODataset(Dataset):
    def __init__(self, annotation_file, image_root, processor):
        with open(annotation_file, "r") as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.annotations = coco["annotations"]
        self.image_root = image_root
        self.processor = processor

        self.img_to_anns = {}
        for ann in self.annotations:
            self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

        self.image_ids = list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        image_path = os.path.join(self.image_root, image_info["file_name"])

        image = Image.open(image_path).convert("RGB")
        anns = self.img_to_anns.get(image_id, [])

        target = {
            "image_id": image_id,
            "annotations": anns,
        }

        encoding = self.processor(
            images=image,
            annotations=target,
            return_tensors="pt",
        )

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"][0],
        }

#CUDA Check
print("Is CUDA available: ", torch.cuda.is_available())

# Image processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

id2label = {
    0: "RBC",
    1: "WBC",
    2: "Platelets"
}

label2id = {v: k for k, v in id2label.items()}

# Model
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

for name, param in model.model.backbone.named_parameters():
    if "layer4" not in name:  # only last ResNet block trainable
        param.requires_grad = False 

model.config.num_queries = 100

# Explicit no-object handling
# model.config.num_labels = 3
'''
#model.config.background_label = -1

# Stabilize Hungarian matching
model.config.class_cost = 1.0
model.config.bbox_cost = 5.0
model.config.giou_cost = 2.0'''

train_dataset = COCODataset(
    annotation_file=TRAIN_JSON,
    image_root=IMAGE_ROOT,
    processor=processor,
)

val_dataset = COCODataset(
    annotation_file=VAL_JSON,
    image_root=IMAGE_ROOT,
    processor=processor,
)

# Data collator
def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]

    return {
        "pixel_values": torch.stack(pixel_values),
        "labels": labels,
    }

# Training arguments
training_args = TrainingArguments(
    output_dir=str(BASE_DIR / "training_outputs/runs/"),
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=10,
    learning_rate=5e-5,
    weight_decay=1e-4,
    logging_steps=10,
    save_steps=500,
    eval_strategy="epoch",
    remove_unused_columns=False,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

trainer.train()
trainer.save_model(BASE_DIR / "training_outputs/detr_bccd")
processor.save_pretrained(BASE_DIR / "training_outputs/detr_bccd")


print("Training complete. Model saved.")

#-----------------ONNX Export----------------
print("Exporting model to ONNX format for C++ and Python comparison")

dummy_input = torch.randn(1, 3, 800, 800)

if torch.cuda.is_available():
    dummy_input = dummy_input.cuda()
    
onnx_out_path = str(BASE_DIR / "training_outputs/detr_bccd/model.onnx")
torch.onnx.export(
    model,
    (dummy_input,),
    onnx_out_path,
    export_params=True,        # Store the trained parameter weights inside the file
    opset_version=16,          # Stable opset version supporting Transformer layers
    input_names=["input"],     # Name of input layer for your C++ code
    output_names=["output"],   # Name of output layer for your C++ code
    dynamic_axes={             # Allows C++ to pass variable image dimensions later
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size"}
    }
)
print(f"ONNX model successfully saved to: {onnx_out_path}")

# ---------------- Evaluation ----------------

metrics = trainer.evaluate()
eval_dir = BASE_DIR / "training_outputs/detr_bccd_eval"

print("Evaluation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v}")

# Save metrics to file
with open(eval_dir / "eval_metrics.txt", "w") as f:
    for k, v in metrics.items():
        f.write(f"{k}: {v}\n")