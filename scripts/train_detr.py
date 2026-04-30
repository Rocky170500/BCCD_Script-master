from transformers import DetrForObjectDetection

model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=3,
    ignore_mismatched_sizes=True
)

print("DETR model loaded successfully with 3 classes.")
