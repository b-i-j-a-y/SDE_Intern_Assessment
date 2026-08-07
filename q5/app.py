# ==================================================
# Q5 AI Virtual Try-On System
# ==================================================

import sys
from pathlib import Path
import tempfile


# --------------------------------------------------
# Add project root
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


import streamlit as st


# --------------------------------------------------
# Modules
# --------------------------------------------------

from q5.modules.detector import detect_person
from q5.modules.pose import detect_pose
from q5.modules.tryon import run_tryon
from q5.modules.evaluation import evaluate_result



# ==================================================
# Page Configuration
# ==================================================

st.set_page_config(
    page_title="AI Virtual Try-On",
    page_icon="👕",
    layout="wide"
)



# ==================================================
# Header
# ==================================================

st.title("👕 AI Virtual Try-On System")

st.markdown(
"""
### AI Clothing Transfer Pipeline

**Models Used**

- 👁️ Florence-2  
  Image understanding and attribute extraction

- 🧩 SCHP  
  Self-Correction for Human Parsing

- 👕 IDM-VTON  
  Virtual try-on generation

- 📊 CLIP  
  Quality evaluation
"""
)


st.divider()



# ==================================================
# Upload
# ==================================================

st.subheader("📥 Input Images")


col1, col2 = st.columns(2)


with col1:

    person_file = st.file_uploader(
        "Upload Person Image",
        type=[
            "png",
            "jpg",
            "jpeg"
        ]
    )


with col2:

    garment_file = st.file_uploader(
        "Upload Garment Image",
        type=[
            "png",
            "jpg",
            "jpeg"
        ]
    )



st.divider()



run_button = st.button(
    "🚀 Generate Try-On",
    use_container_width=True
)



# ==================================================
# Pipeline
# ==================================================

if run_button:


    if person_file is None or garment_file is None:

        st.error(
            "Please upload both images"
        )

        st.stop()



    # ----------------------------------------------
    # Save uploaded files
    # ----------------------------------------------

    person_temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    )


    garment_temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    )


    person_temp.write(
        person_file.read()
    )


    garment_temp.write(
        garment_file.read()
    )


    person_temp.close()
    garment_temp.close()



    person_path = Path(
        person_temp.name
    )


    garment_path = Path(
        garment_temp.name
    )



    # ----------------------------------------------
    # Input Preview
    # ----------------------------------------------

    st.subheader("🖼️ Input Preview")


    a,b = st.columns(2)


    with a:

        st.image(
            person_path,
            caption="Person Image",
            use_container_width=True
        )


    with b:

        st.image(
            garment_path,
            caption="Garment Image",
            use_container_width=True
        )



    st.divider()



    # ----------------------------------------------
    # Person Detection
    # ----------------------------------------------

    st.subheader("⚙️ Pipeline Status")


    status = st.empty()


    status.info(
        "Checking person detection..."
    )


    detected, message = detect_person(
        person_path
    )


    if not detected:

        status.error(
            message
        )

        st.stop()


    status.success(
        "✅ Person detection completed"
    )



    # ----------------------------------------------
    # Pose Guardrail
    # ----------------------------------------------

    pose_result = detect_pose(
        person_path
    )


    if pose_result["warning"]:

        st.warning(
            pose_result["warning"]
        )

    else:

        st.success(
            f"✅ Pose detected: {pose_result['pose_category']}"
        )



    # ----------------------------------------------
    # Florence Analysis
    # ----------------------------------------------

    status.info(
        "Running Florence-2 analysis..."
    )


    from q1.main import analyze_images


    analysis_result = analyze_images(
        person_path,
        garment_path
    )


    status.success(
        "✅ Florence analysis completed"
    )



    # ----------------------------------------------
    # Q2 Intermediate Outputs
    # ----------------------------------------------

    st.divider()

    st.subheader(
        "🧩 SCHP Human Parsing Outputs"
    )


    person_id = Path(
        person_file.name
    ).stem



    parsing_path = Path(
        f"q2/outputs/parsing_maps/{person_id}_parsing.png"
    )


    mask_path = Path(
        f"q2/outputs/person_masks/{person_id}_mask.png"
    )


    agnostic_path = Path(
        f"q2/outputs/agnostic/{person_id}_agnostic.png"
    )



    p1,p2,p3 = st.columns(3)



    with p1:

        st.write(
            "Human Parsing"
        )


        if parsing_path.exists():

            st.image(
                parsing_path,
                use_container_width=True
            )

        else:

            st.info(
                "Parsing output unavailable"
            )



    with p2:

        st.write(
            "Human Mask"
        )


        if mask_path.exists():

            st.image(
                mask_path,
                use_container_width=True
            )

        else:

            st.info(
                "Mask unavailable"
            )



    with p3:

        st.write(
            "Agnostic Representation"
        )


        if agnostic_path.exists():

            st.image(
                agnostic_path,
                use_container_width=True
            )

        else:

            st.info(
                "Agnostic unavailable"
            )



    # ----------------------------------------------
    # Try-On Result
    # ----------------------------------------------

    st.divider()


    st.subheader(
        "✨ Virtual Try-On Result"
    )


    result_image = run_tryon(
        person_path,
        garment_path
    )



    r1,r2 = st.columns(2)



    with r1:

        st.image(
            person_path,
            caption="Original Person",
            use_container_width=True
        )



    with r2:

        st.image(
            result_image,
            caption="Generated Try-On",
            use_container_width=True
        )



    # ----------------------------------------------
    # Attributes
    # ----------------------------------------------

    st.divider()


    st.subheader(
        "👕 Extracted Attributes"
    )


    x1,x2 = st.columns(2)



    with x1:

        st.write(
            "Garment Attributes"
        )

        st.json(
            analysis_result[
                "garment_attributes"
            ]
        )



    with x2:

        st.write(
            "Person Attributes"
        )

        st.json(
            analysis_result[
                "person_attributes"
            ]
        )



    # ----------------------------------------------
    # Evaluation
    # ----------------------------------------------

    st.divider()


    st.subheader(
        "📊 Quality Evaluation"
    )


    scores = evaluate_result(
        person_path,
        garment_path,
        "q5/outputs/tryon_result.png"
    )



    s1,s2,s3,s4 = st.columns(4)



    s1.metric(
        "CLIP Similarity",
        scores["clip_similarity"]
    )


    s2.metric(
        "Garment Preservation",
        scores["garment_preservation"]
    )


    s3.metric(
        "Alignment Score",
        scores["alignment_score"]
    )


    s4.metric(
        "Overall Score",
        scores["overall_score"]
    )



    st.success(
        "🎉 Pipeline completed successfully"
    )