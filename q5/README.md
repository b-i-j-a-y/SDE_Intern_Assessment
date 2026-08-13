# Q5 — Streamlit Web Demo

## Objective

The goal of Q5 is to provide a lightweight interactive web interface that demonstrates the complete virtual try-on workflow.

The demo integrates:

- Q1 attribute extraction
- Q2 parsing visualization
- Q3 try-on results
- Q4 quality evaluation
- Content-based guardrails

---

# Technologies Used

- **Streamlit** — web interface
- **Florence-2** — attribute extraction
- **OpenCV** — image handling
- **Pillow** — image processing
- **PyTorch** — model backend
- **NumPy** — utility operations

---

# How to Run

Run from the **project root**.

```bash
streamlit run q5/app.py
```

After launching, open:

```text
http://localhost:8501
```

---

# Features Implemented

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

# Pipeline

<Code value=
