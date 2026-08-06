# SDE Intern Technical Assessment

## Candidate

**Name:** Bijay K B

---

# Project Overview

This repository contains my solutions for the SDE Intern Technical Assessment.

At the moment, I have completed **Question 1**, which focuses on understanding garment and person images using a Vision-Language Model.

The remaining questions (Q2–Q5) will be added as I progress through the assessment.

---

# Question 1 – Garment & Person Understanding

For this task, I used **Microsoft Florence-2-base**, an open-source Vision-Language Model, to understand both garment and person images.

The workflow includes:

- Generating captions for garment images
- Extracting garment attributes such as type, sleeve length, neckline, color, and pattern
- Understanding person images
- Identifying pose and body visibility
- Handling the provided edge-case images
- Saving the results as JSON files

---

# Model Used

**Model:** Florence-2-base

**Repository:**
https://huggingface.co/microsoft/Florence-2-base

**Purpose:**

- Image captioning
- Garment understanding
- Person understanding

**License:** MIT License

---

# Technologies Used

- Python 3.13
- PyTorch
- Transformers
- Pillow
- Hugging Face

---

# Project Structure

```
SDE_Intern_Assessment/

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

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# Running the Project

```bash
python3 q1/main.py
```

The generated JSON files will be saved inside:

```
q1/outputs/
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

The overall workflow is straightforward:

1. Load the Florence-2 model.
2. Generate captions for garment images.
3. Extract garment attributes using simple rule-based parsing.
4. Generate captions for person images.
5. Extract pose and body visibility information.
6. Handle the provided edge-case images.
7. Save all extracted information as structured JSON files.

---

# Edge Cases

The following edge-case images are also processed:

- `person_side_pose.jpg`
- `person_seated.jpg`
- `person_crossed_arms.jpg`
- `no_person.jpg`

Separate JSON files are generated for these images.

---

# Limitations

This project uses Florence-2 as a general Vision-Language Model. While it performs well on most standard images, it may produce inaccurate descriptions for some challenging edge cases. Because the attribute extraction is based on the generated captions, the final results depend on the quality of those captions.

A dedicated pose estimation model or human parsing model would improve the robustness of the system.

---

# Models & License

| Model | Purpose | License |
|--------|---------|---------|
| Florence-2-base | Vision-Language Understanding | MIT |

---

# Assessment Compliance

This solution follows the assessment guidelines:

- Uses only open-source models.
- No paid or hosted APIs were used.
- All outputs are generated locally.
- Results are saved in JSON format.
- Edge-case images are included in the evaluation.

---



# Limitations

Florence-2 performs well on most standard images, but some challenging edge-case images may produce inaccurate descriptions. Since the final attributes are extracted from these descriptions using rule-based parsing, the overall accuracy depends on the quality of the generated captions.

A dedicated pose estimation or human parsing model could improve the performance for these cases.
