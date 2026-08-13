# SDE Intern Technical Assessment

## Candidate Details

- **Name:** Bijay K B
- **Role:** Software Engineering Intern
- **Repository:** https://github.com/b-i-j-a-y/SDE_Intern_Assessment
- **Demo Video:** https://drive.google.com/file/d/14Bsk5MRteWK4lfCXO8nvCxhotV56reLc/view?usp=drive_link

---

# Project Overview

This repository contains my solutions for all questions in the XIPL Software Engineering Intern Technical Assessment.

## Folder Structure

```text
q1/  # Garment & Person Understanding using Florence-2
q2/  # Human Parsing & Garment Segmentation
q3/  # Virtual Try-On Pipeline
q4/  # Automated Quality Evaluation
q5/  # Streamlit Demo / Integration
```

---

# Environment

- **OS:** macOS (Apple Silicon)
- **Python:** 3.11
- **Frameworks:** PyTorch, OpenCV, Transformers, Streamlit

## Setup

```bash
git clone https://github.com/b-i-j-a-y/SDE_Intern_Assessment.git
cd SDE_Intern_Assessment

python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

# Question 1 — Garment & Person Understanding

## Objective

Use the Florence-2 vision-language model to analyze a person image and a garment image and extract structured attributes.

## Run

```bash
python q1/main.py
```

## Output

- Person attributes
- Garment attributes
- Structured JSON-style results

---

# Question 2 — Human Parsing & Garment Segmentation

## Objective

Generate human parsing maps, person masks, garment masks, and agnostic representations.

## Run

### Human parsing and person masks

```bash
python q2/scripts/segment_person.py
```

### Garment segmentation

```bash
python q2/scripts/segment_garment.py
```

## Output

- `q2/outputs/parsing_maps/`
- `q2/outputs/person_masks/`
- `q2/outputs/garment_masks/`
- `q2/outputs/agnostic/`

Edge-case outputs are stored in:

- `q2/outputs/edge_cases/`

---

# Question 3 — Virtual Try-On

## Objective

Generate virtual try-on results for the evaluation pairs.

## Evaluation Pairs

### Official pairs

- pair_01
- pair_02
- pair_03

### Candidate-sourced pairs

- pair_04
- pair_05

The pair definitions are listed in:

```text
q4/pairs_manifest.csv
```

## Run

The repository contains pre-generated outputs for the five evaluation pairs.

Outputs are stored in:

```text
q3/outputs/
```

Generated files:

- `pair_01_output.png`
- `pair_02_output.png`
- `pair_03_output.png`
- `pair_04_output.png`
- `pair_05_output.png`

---

# Question 4 — Automated Quality Evaluation

## Objective

Evaluate the generated try-on results using garment similarity, identity preservation, and a Florence-2 based judge.

## Run

```bash
python q4/evaluate.py
```

## Output

Generated automatically:

```text
q4/outputs/evaluation_results.csv
```

Committed evaluation template:

```text
q4/evaluation_template_q4.csv
```

---

## Judge Rubric Prompt

The Florence-2 based judge evaluates the generated image using the following rubric:

- **Fit Realism (4 points)** — Does the garment appear naturally worn and aligned with the body pose?
- **Texture Transfer (3 points)** — Are garment textures, logos, and visual details preserved?
- **Artifact Detection (3 points)** — Are visible distortions, broken edges, or unrealistic regions present?

The model first produces a descriptive analysis of the generated image. A rule-based scorer then converts that analysis into a final score from **1 to 10**.

---

# Question 5 — Streamlit Demo

## Objective

Provide a lightweight web interface that integrates the earlier stages of the pipeline.

## Run

```bash
streamlit run q5/app.py
```

## Implemented Features

- Upload person image
- Upload garment image
- Florence-2 attribute extraction
- Display available Q2 parsing outputs
- Content-based no-person rejection
- Side-pose warning
- Seated-pose warning
- Estimated processing time
- Quality score display

---

## Important Implementation Note

The Streamlit demo **does not run live IDM-VTON inference**.

For the five known evaluation pairs, the app displays the corresponding **pre-generated Q3 outputs**:

- person_01 → pair_01_output
- person_02 → pair_02_output
- person_03 → pair_03_output
- custom_person_01 → pair_04_output
- custom_person_02 → pair_05_output

For unknown uploads, the app falls back to the original person image.

This design keeps the demo lightweight and reproducible on CPU-only environments.

---

# Demo Video

A short walkthrough of the complete pipeline and web demo:

**Video:** https://drive.google.com/file/d/14Bsk5MRteWK4lfCXO8nvCxhotV56reLc/view?usp=drive_link

---

# Colab Usage

No Google Colab notebooks were used. All experiments were executed locally on macOS.

---

# Notes and Trade-offs

- Only open-source models and libraries were used.
- The repository is optimized for CPU execution where possible.
- High-quality live virtual try-on models require significantly larger checkpoints and GPU memory.
- Q3 outputs are included in the repository to ensure reproducibility.
- Q5 is a demonstration wrapper around the generated Q3 results rather than a full live try-on service.

---

# Reproducibility

After cloning the repository:

```bash
git clone https://github.com/b-i-j-a-y/SDE_Intern_Assessment.git
cd SDE_Intern_Assessment

python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Run each question independently using the commands listed above.

---

# Submission Checklist

- [x] Public GitHub repository
- [x] One folder per question
- [x] Working requirements file
- [x] All imported source files committed
- [x] Three official pairs included
- [x] Two candidate-sourced pairs included
- [x] Q4 evaluation template committed
- [x] Judge rubric documented
- [x] Q5 content-based guardrails implemented
- [x] README claims match the implementation
- [x] Demo video link included

---

Thank you for reviewing my submission.
