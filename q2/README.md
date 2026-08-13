# Q2 — Human Parsing & Garment Segmentation

## Objective

The goal of Q2 is to generate the intermediate representations required for the virtual try-on pipeline:

- Human parsing maps
- Person segmentation masks
- Garment segmentation masks
- Agnostic person representations

These outputs are later used by **Q3 (Virtual Try-On)**, **Q4 (Evaluation)**, and **Q5 (Web Demo)**.

---

# Models Used

### 1. Grounding DINO
Used for detecting the **person** and **garment** regions in the image.

### 2. Segment Anything Model (SAM ViT-B)
Used to generate accurate segmentation masks from the detected regions.

### 3. Human Parsing Pipeline
Used to create body-part parsing maps and agnostic representations.

---

# Technologies

- Python 3.11
- PyTorch
- OpenCV
- Pillow
- NumPy

---

# How to Run

Run all commands from the **project root**.

## Person Parsing & Agnostic Generation

```bash id=
