import os
import json
import xml.etree.ElementTree as ET


BASE_DIR = "C:/Users/Hardika Menghani/Desktop/Ravin/HS_Mannheim/BCCD_Dataset-master/BCCD"
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "Annotations")
IMAGES_DIR = os.path.join(BASE_DIR, "JPEGImages")
SPLITS_DIR = os.path.join(BASE_DIR, "ImageSets", "Main")

OUTPUT_DIR = "C:/Users/Hardika Menghani/Desktop/Ravin/HS_Mannheim/coco_Data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- CLASS MAPPING ----
CATEGORIES = {
    "RBC": 0,
    "WBC": 1,
    "Platelets": 2
}

def convert_split(split_name):
    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    split_file = os.path.join(SPLITS_DIR, f"{split_name}.txt")
    with open(split_file, "r") as f:
        image_ids = [line.strip() for line in f.readlines()]

    for image_name in image_ids:
        xml_path = os.path.join(ANNOTATIONS_DIR, f"{image_name}.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find("filename").text
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        images.append({
            "id": img_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        for obj in root.findall("object"):
            label = obj.find("name").text
            if label not in CATEGORIES:
                continue

            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": CATEGORIES[label],
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "area": (xmax - xmin) * (ymax - ymin),
                "iscrowd": 0
            })
            ann_id += 1

        img_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 0, "name": "RBC"},
            {"id": 1, "name": "WBC"},
            {"id": 2, "name": "Platelets"}
        ]
    }

    output_path = os.path.join(OUTPUT_DIR, f"instances_{split_name}.json")
    with open(output_path, "w") as f:
        json.dump(coco_dict, f, indent=2)

    print(f"Saved {output_path}")

# ---- RUN ----
convert_split("train")
convert_split("val")
