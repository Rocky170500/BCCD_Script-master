import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import DetrForObjectDetection, DetrImageProcessor
from pycocotools.coco import COCO
from pathlib import Path
from sklearn.metrics import confusion_matrix
from PIL import Image
import itertools

BASE_DIR = Path(__file__).resolve().parent.parent

VAL_JSON = BASE_DIR / "coco_Data/instances_val.json"
IMAGE_ROOT = BASE_DIR / "BCCD_Dataset-master/BCCD/JPEGImages"
MODEL_DIR = BASE_DIR / "outputs/detr_bccd"
OUTPUT_DIR = BASE_DIR / "outputs/detr_bccd_eval"

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = DetrImageProcessor.from_pretrained(MODEL_DIR)
model = DetrForObjectDetection.from_pretrained(MODEL_DIR).to(device)
model.eval()

coco = COCO(str(VAL_JSON))

iou_threshold = 0.5
score_threshold = 0.5   # IMPORTANT

gt_labels = []
pred_labels = []

def compute_iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])

    return inter / (areaA + areaB - inter + 1e-6)

for img_id in coco.getImgIds():
    img_info = coco.loadImgs(img_id)[0]
    img_path = IMAGE_ROOT / img_info["file_name"]

    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    preds = processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([[image.height, image.width]]),
        threshold=score_threshold
    )[0]

    pred_boxes = preds["boxes"].cpu().numpy()
    pred_labels_img = preds["labels"].cpu().numpy()
    used_preds = set()

    gt_anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

    for gt in gt_anns:
        gt_box = [
            gt["bbox"][0],
            gt["bbox"][1],
            gt["bbox"][0] + gt["bbox"][2],
            gt["bbox"][1] + gt["bbox"][3],
        ]
        gt_label = gt["category_id"]

        best_iou = 0
        best_idx = -1

        for i, pred_box in enumerate(pred_boxes):
            if i in used_preds:
                continue
            iou = compute_iou(gt_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= iou_threshold:
            used_preds.add(best_idx)
            gt_labels.append(gt_label)
            pred_labels.append(pred_labels_img[best_idx])

labels = ["RBC", "WBC", "Platelets"]
cm = confusion_matrix(gt_labels, pred_labels, labels=[0, 1, 2])

plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)

for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], ha="center")

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
plt.close()

print("Confusion matrix saved correctly.")
