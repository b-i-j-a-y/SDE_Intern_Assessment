# SDE Intern Technical Assessment

## Candidate

**Name:** Bijay K B

---

# Project Overview

This repository contains my solutions for the SDE Intern Technical Assessment.

Currently Completed:

- ✅ Q1 – Garment & Person Understanding using a Vision-Language Model (Florence-2)

Remaining Questions:

- ⏳ Q2 – Human Parsing & Garment Segmentation
- ⏳ Q3 – Virtual Try-On
- ⏳ Q4 – Evaluation Pipeline
- ⏳ Q5 – Edge Cases & Guardrails

---

# Q1 Overview

The objective of Q1 is to extract structured information from garment and person images using a Vision-Language Model.

The pipeline performs:

- Garment attribute extraction
- Person pose estimation
- Upper body visibility detection
- Lower body visibility detection
- JSON generation for each image pair

---

# Model Used

### Florence-2-base

Repository:
https://huggingface.co/microsoft/Florence-2-base

Purpose:

- Image Captioning
- Vision-Language Understanding
- Garment Description
- Person Description

License:

MIT License

---

# Programming Language

- Python 3.13

---

# Libraries Used

- transformers
- torch
- pillow
- huggingface_hub
- safetensors
- einops

---

# Folder Structure

```
SDE_Interen_Assesment/

│
├── data/
│   ├── garment/
│   ├── person/
│   └── edge_cases/
│
├── q1/
│   ├── florence.py
│   ├── parser.py
│   ├── main.py
│   └── outputs/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation

Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Run

```bash
python3 q1/main.py
```

---

# Output

The program generates JSON files inside

```
q1/outputs/
```

Example:

```
pair_01.json
pair_02.json
pair_03.json
pair_04.json
pair_05.json
```

---

# Example Output

```json
{
    "person_image": "person_01.png",
    "garment_image": "garment_01.jpg",

    "garment_attributes": {
        "type": "t-shirt",
        "sleeve_length": "short",
        "neckline": "round",
        "primary_color": "light purple",
        "pattern": "logo"
    },

    "person_attributes": {
        "pose_category": "side",
        "upper_body_visible": true,
        "lower_body_visible": true
    },

    "model_used": "Florence-2-base",

    "confidence_notes": "Generated using Florence-2 and rule-based parsing."
}
```

---

# Approach

1. Load Florence-2.
2. Generate detailed captions for garment images.
3. Extract garment attributes using a rule-based parser.
4. Generate detailed captions for person images.
5. Extract pose and visibility information.
6. Combine both outputs into a structured JSON file.

---

# Limitations

- Attribute extraction uses rule-based parsing from Florence descriptions.
- Results depend on caption quality.
- Some garment details (neckline, sleeve type) may be unavailable if not described by the model.

---

# Models & Licenses

| Model | Purpose | License |
|-------|---------|---------|
| Florence-2-base | Vision-Language Understanding | MIT |

---

# Notes

This project uses only open-source models with freely downloadable weights, in accordance with the assessment requirements.

No paid or hosted inference APIs were used.

All inference was performed locally.