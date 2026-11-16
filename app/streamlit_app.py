"""Weaver Lab histology analyzer (Streamlit).

Run from the project root:   streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Weaver Lab Histology Analyzer", layout="wide")
st.title("Weaver Lab histology analyzer")

MODES = ["Whole slide (VSI or large scan)", "Single image or region (PNG, JPG, TIFF)"]
with st.sidebar:
    mode = st.radio("What are you analysing?", MODES)
    st.divider()

if mode == MODES[0]:
    st.caption("Finds the tissue sections on a slide, analyses them at cell resolution, and learns from your "
               "corrections: nuclei, cell-state, infiltration, collagen capsule, hydrogel geometry.")
    import slide_mode
    slide_mode.render()
else:
    st.caption("H&E and Masson's trichrome: nuclei, cell-state, infiltration, fibrosis capsule, "
               "construct geometry. Results are CSV-exportable.")
    import image_mode
    image_mode.render()
