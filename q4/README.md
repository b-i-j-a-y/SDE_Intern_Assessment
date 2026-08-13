# Q4 — Automated Quality Evaluation

## Objective

The goal of Q4 is to automatically evaluate the virtual try-on results generated in **Q3**.

The evaluation focuses on:

- **Garment Fidelity**
- **Identity Preservation**
- **Overall Visual Quality**

The pipeline combines embedding-based metrics with a Florence-2 based visual judge.

---

# Models Used

### 1. OpenCLIP ViT-B-32
Used to measure **garment similarity** between the reference garment and the generated try-on result.

### 2. InsightFace (Buffalo-L)
Used to measure **identity preservation** using face embeddings.

### 3. Florence-2 Base
Used as a **Vision-Language Judge (VLM-as-Judge)** to analyze the generated image and produce a descriptive assessment.

---

# Technologies

- Python 3.11
- PyTorch
- OpenCLIP
- InsightFace
- ONNX Runtime
- Pandas
- Pillow
- NumPy

---

# Evaluation Pairs

The evaluation uses:

## Official Pairs

- pair_01
- pair_02
- pair_03

## Candidate-Sourced Pairs

- pair_04
- pair_05

Pair definitions are stored in:

```text id=
