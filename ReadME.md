# Blood Cell Detection using DETR (Transformer-based Object Detection)

## Overview
This project fine-tunes a Transformer-based object detection model (DETR) on the BCCD (Blood Cell Count and Detection) dataset.
The goal is to detect and classify blood cells into:
- RBC (Red Blood Cells)
- WBC (White Blood Cells)
- Platelets

The project uses a pretrained DETR model from Hugging Face and adapts it to a medical imaging domain.The project can be reproduced by creating a virtual environment and installing dependencies via a single command.

---

### Repository Structure
BCCD_Script-master/
├── coco_Data/             # COCO annotations
├── scripts/               # Training & evaluation scripts
├── outputs/               # Saved model checkpoints
├── requirements.txt       # List of dependencies
└── README.md              # Description of whole project

## Data Source
This project uses the [BCCD (Blood Cell Count and Detection) Dataset] provided by [Shenggan](https://github.com/Shenggan/BCCD_Dataset).

>**Note:** The dataset is not included in this repository. To run this project, please download the data from the original source linked above and replace the path in "finetune_detr.py".

## Dataset
- Dataset: BCCD (Blood Cell Count and Detection) 
- Original annotations: Pascal VOC (XML)
- Converted format: COCO JSON
- Image resolution: 640 × 480
- Train / Validation split applied

Classes:
| ID | Class |
|----|-------|
| 0  | background |
| 1  | RBC |
| 2  | WBC |
| 3  | Platelets |

---

## Model
- Architecture: DETR (Detection Transformer)
- Backbone: ResNet-50
- Pretrained on: MS-COCO
- Framework: Hugging Face Transformers

Transfer learning strategy:
- Backbone and transformer encoder-decoder initialized from pretrained weights
- Classification head reinitialized for BCCD classes

---

## Environment Setup

### Python Version
- Python 3.12

<!--
### Create Virtual Environment and Installation
```bash
python -m venv venv

.\venv\Scripts\activate

pip install -r requirements.txt

```
-->
### Training details:
Epochs: 10
Batch size: 2
Optimizer: AdamW
Learning rate: 5e-5
Loss: Hungarian matching loss (DETR)

### Commands to run pipeline
```bash
python .\scripts\run_pipeline.py

```

### GPU Note
> **Note:** If a CUDA-compatible GPU is available, PyTorch will automatically use it.
> The project also runs correctly on CPU.