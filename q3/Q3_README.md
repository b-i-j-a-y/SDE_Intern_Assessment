# Q3 — Virtual Try-On Pipeline

## Objective

The goal of Q3 is to generate virtual try-on results by combining:

- a **person image**
- a **garment image**
- the structural information produced in **Q2**

The generated outputs are later used for **Q4 automated evaluation** and **Q5 web demonstration**.

---

# Models Used

### 1. Florence-2
Used for garment and person attribute understanding.

### 2. Human Parsing & Agnostic Representation (from Q2)
Provides body structure and clothing-free upper-body representation.

### 3. CatVTON / IDM-VTON Style Workflow
Used for the try-on generation stage.

The repository includes the generated outputs for the required evaluation pairs.

---

# Technologies

- Python 3.11
- PyTorch
- OpenCV
- Pillow
- NumPy

---

# Evaluation Pairs

## Official Pairs

- pair_01
- pair_02
- pair_03

## Candidate-Sourced Pairs

- pair_04 → `custom_person_01.png` + `custom_garment_01.png`
- pair_05 → `custom_person_02.png` + `custom_garment_02.png`

Pair definitions are stored in:

```text id=
