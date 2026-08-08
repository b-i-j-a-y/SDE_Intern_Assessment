# SDE Intern Technical Assessment

## Candidate Details

- **Name:** Bijay K B
- **Role:** Software Engineering Intern
- **Repository:** https://github.com/b-i-j-a-y/SDE_Intern_Assessment
- **Demo Video:**https://drive.google.com/file/d/1p6oEkqpZxlT3-tY7Mv15hfpE22Wj8jGc/view?usp=drive_link
---

# Project Overview

This repository contains my solutions for all questions in the XIPL Software Engineering Intern Technical Assessment.

## Folder Structure

```text
q1/  # Garment & Person Understanding using Florence-2
q2/  # Human Parsing & Garment Segmentation
q3/  # Virtual Try-On Pipeline
q4/  # Automated Quality Evaluation
q5/  # Web Demo / Integration
```

---

# Environment

- **OS:** macOS (Apple Silicon)
- **Python:** 3.11
- **Frameworks:** PyTorch, OpenCV, Transformers, Gradio/Streamlit

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Question 1 — Garment & Person Understanding

## Objective
Detect and describe the person and garment using the Florence-2 vision-language model.

## Run

```bash
python q1/run.py
```

## Output
- Person description
- Garment description
- Structured attributes

---

# Question 2 — Human Parsing & Garment Segmentation

## Objective
Generate segmentation masks for the person and garment regions.

## Run

```bash
python q2/run.py
```

## Output
- Human parsing mask
- Garment mask
- Visualization image

---

# Question 3 — Virtual Try-On

## Objective
Generate a virtual try-on result by combining the selected person and garment.

## Run

```bash
python q3/run.py
```

## Output
- Final try-on image
- Intermediate preprocessing outputs

---

# Question 4 — Automated Quality Evaluation

## Objective
Evaluate generated results automatically using image quality metrics.

## Run

```bash
python q4/run.py
```

## Output
- `evaluation_template_q4.csv`
- Quality metrics summary

---

# Question 5 — Web Demo

## Objective
Provide a simple web interface for running the pipeline.

## Run

```bash
streamlit run q5/app.py
```

## Features
- Upload person image
- Upload garment image
- Run virtual try-on
- View generated output

---

# Demo Video

A short walkthrough of the complete pipeline and web demo is available here:

**Video Link:** YOUR_GOOGLE_DRIVE_VIDEO_LINK

---

# Colab Notebooks

No Google Colab notebooks were used. All experiments were executed locally on macOS.

---

# Notes and Trade-offs

- Open-source models only; no paid APIs were used.
- Optimized for CPU execution where possible.
- Higher-quality virtual try-on models require larger checkpoints and more memory.
- The pipeline prioritizes reproducibility and clear folder organization.

---

# Reproducibility

After cloning the repository:

```bash
git clone https://github.com/b-i-j-a-y/SDE_Intern_Assessment.git
cd SDE_Intern_Assessment
pip install -r requirements.txt
```

Run any question independently using the commands listed above.

---

# Submission Checklist

- [x] Public GitHub repository
- [x] One folder per question
- [x] Root README completed
- [x] Demo video link added
- [x] `evaluation_template_q4.csv` committed
- [x] Run instructions provided

---

Thank you for reviewing my submission.
