import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import DetrForObjectDetection, DetrImageProcessor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pathlib import Path
import json
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent.parent

VAL_JSON = BASE_DIR / "coco_Data/instances_val.json"
IMAGE_ROOT = BASE_DIR / "BCCD_Dataset-master/BCCD/JPEGImages"
MODEL_DIR = BASE_DIR / "outputs/detr_bccd"
OUTPUT_DIR = BASE_DIR / "outputs/detr_bccd_eval"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = DetrImageProcessor.from_pretrained(MODEL_DIR)
model = DetrForObjectDetection.from_pretrained(MODEL_DIR, ignore_mismatched_sizes=True).to(device)
model.eval()

coco_gt = COCO(str(VAL_JSON))

results = []

for img_id in coco_gt.getImgIds():
    img_info = coco_gt.loadImgs(img_id)[0]
    img_path = IMAGE_ROOT / img_info["file_name"]

    img = Image.open(img_path).convert("RGB")
    encoding = processor(images=img, return_tensors="pt")
    image = encoding["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(image)

    target_sizes = torch.tensor([[img_info["height"], img_info["width"]]])
    processed = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.3
    )[0]

    for score, label, box in zip(
        processed["scores"], processed["labels"], processed["boxes"]
    ):
        x_min, y_min, x_max, y_max = box.tolist()
        results.append(
            {
                "image_id": img_id,
                "category_id": int(label),
                "bbox": [
                    x_min,
                    y_min,
                    x_max - x_min,
                    y_max - y_min,
                ],
                "score": float(score),
            }
        )

# Save predictions
pred_json = OUTPUT_DIR / "predictions.json"
with open(pred_json, "w") as f:
    json.dump(results, f)

# COCO Evaluation
coco_dt = coco_gt.loadRes(str(pred_json))
coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

with open(OUTPUT_DIR / "map_metrics.txt", "w") as f:
    f.write(f"mAP@[0.5:0.95]: {coco_eval.stats[0]:.4f}\n")
    f.write(f"mAP@0.5: {coco_eval.stats[1]:.4f}\n")
    f.write(f"mAP@0.75: {coco_eval.stats[2]:.4f}\n")


precision = coco_eval.eval["precision"]

# IoU = 0.5
iou_idx = 0

# precision: [R, K]
pr = precision[iou_idx, :, :, 0, -1]

# Remove invalid entries (-1)
pr = np.where(pr < 0, np.nan, pr)

# Mean over classes, ignoring NaNs
pr_mean = np.nanmean(pr, axis=1)

recall = np.linspace(0, 1, len(pr_mean))

plt.figure()
plt.plot(recall, pr_mean)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (IoU=0.5)")
plt.grid()
plt.savefig(OUTPUT_DIR / "precision_recall_curve.png")
plt.close()

print("Precision–Recall curve saved.")
