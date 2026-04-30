import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from transformers import DetrForObjectDetection, DetrImageProcessor
from pycocotools.coco import COCO
from pathlib import Path
import random

BASE_DIR = Path(__file__).resolve().parent.parent

VAL_JSON = BASE_DIR / "coco_Data/instances_val.json"
IMAGE_ROOT = BASE_DIR / "BCCD_Dataset-master/BCCD/JPEGImages"
MODEL_DIR = BASE_DIR / "outputs/detr_bccd"
OUTPUT_DIR = BASE_DIR / "outputs/detr_bccd_eval/visualization"


device = "cuda" if torch.cuda.is_available() else "cpu"

processor = DetrImageProcessor.from_pretrained(MODEL_DIR)
model = DetrForObjectDetection.from_pretrained(MODEL_DIR, ignore_mismatched_sizes=True).to(device)

# Enforce same architecture as training
model.config.num_queries = 100

# Reinitialize query embeddings
model.model.query_embed = torch.nn.Embedding(
    model.config.num_queries,
    model.config.hidden_size
)

model.eval()

coco = COCO(str(VAL_JSON))
image_ids = coco.getImgIds()
selected_ids = random.sample(image_ids, 10)

id2label = {
    0: "RBC",
    1: "WBC",
    2: "Platelets",
}

for img_id in selected_ids:
    img_info = coco.loadImgs(img_id)[0]
    img_path = IMAGE_ROOT / img_info["file_name"]

    image = Image.open(img_path).convert("RGB")

    encoding = processor(images=image, return_tensors="pt")
    pixel_values = encoding["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values)

    target_sizes = torch.tensor([[image.height, image.width]])
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=0.3,
    )[0]

    draw = ImageDraw.Draw(image)

    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        if label == 0: #RBC
            x_min, y_min, x_max, y_max = box.tolist()
            draw.rectangle(
                [(x_min, y_min), (x_max, y_max)],
                outline="green",
                width=2,
            )
            text = f"{id2label[int(label)]}: {score:.2f}"
            draw.text((x_min, y_min), text, fill="green")
        elif label == 1: #WBC
            x_min, y_min, x_max, y_max = box.tolist()
            draw.rectangle(
                [(x_min, y_min), (x_max, y_max)],
                outline="red",
                width=2,
            )
            text = f"{id2label[int(label)]}: {score:.2f}"
            draw.text((x_min, y_min), text, fill="red")
        elif label == 2: #Platelets
            x_min, y_min, x_max, y_max = box.tolist()
            draw.rectangle(
                [(x_min, y_min), (x_max, y_max)],
                outline="blue",
                width=2,
            )
            text = f"{id2label[int(label)]}: {score:.2f}"
            draw.text((x_min, y_min), text, fill="blue")

    save_path = OUTPUT_DIR / img_info["file_name"]
    image.save(save_path)

print("Visualization images saved.")
