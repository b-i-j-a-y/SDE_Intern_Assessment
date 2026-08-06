# Q2 – Human Parsing & Garment Segmentation

## Overview

This module implements a complete preprocessing pipeline for virtual try-on systems using open-source vision models.

The pipeline performs:

- Person Detection
- Person Segmentation
- Human Parsing
- Agnostic Image Generation
- Garment Segmentation
- Edge Case Evaluation

The generated outputs are later used as inputs for the virtual try-on pipeline in Q3.

---

# Project Structure

```
q2/

├── models/
│   └── sam_vit_b_01ec64.pth
│
├── scripts/
│   ├── detector.py
│   ├── segmenter.py
│   ├── human_parsing.py
│   ├── create_agnostic.py
│   ├── segment_garment.py
│   ├── pipeline.py
│   └── evaluate_edge_cases.py
│
├── outputs/
│   ├── person_masks/
│   ├── garment_masks/
│   ├── parsing_maps/
│   ├── agnostic/
│   ├── edge_cases/
│   └── visualization/
│
└── README.md
```

---

# Models Used

## Grounding DINO

Purpose:

- Open-vocabulary person detection
- Garment detection

Repository:

https://github.com/IDEA-Research/GroundingDINO

---

## Segment Anything Model (SAM)

Purpose:

- High-quality person segmentation
- Garment mask generation

Model:

ViT-B

---

## SegFormer Human Parsing

Model:

matei-dorian/segformer-b5-finetuned-human-parsing

Purpose:

- Semantic body-part parsing
- Upper-clothes detection
- Arm segmentation
- Face detection
- Hair detection

---

# Pipeline

```
Person Image
        │
        ▼
Grounding DINO
        │
        ▼
Bounding Box
        │
        ▼
SAM
        │
        ▼
Person Mask
        │
        ├──────────────┐
        ▼              ▼
Human Parsing      Agnostic Image
        │
        ▼
Parsing Map

Garment Image
        │
        ▼
Grounding DINO
        │
        ▼
SAM
        │
        ▼
Garment Mask
```

---

# Outputs

The pipeline generates:

- Person Masks
- Parsing Maps
- Agnostic Images
- Garment Masks

---

# Edge Cases

The following challenging images were evaluated:

- Crossed Arms
- Side Pose
- Seated Person
- No Person

---

# How to Run

Generate person preprocessing:

```bash
python q2/scripts/pipeline.py
```

Generate garment masks:

```bash
python q2/scripts/segment_garment.py
```

Evaluate edge cases:

```bash
python q2/scripts/evaluate_edge_cases.py
```

---

# Technologies

- Python
- PyTorch
- HuggingFace Transformers
- Grounding DINO
- Segment Anything (SAM)
- SegFormer
- OpenCV
- NumPy