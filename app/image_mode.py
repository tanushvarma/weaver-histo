"""Single image / region mode (PNG, JPG, TIFF): one image in memory, one analysis."""
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from weaver_histo import Params, analyze, stains, viz  # noqa: E402
from weaver_histo.loading import load_image  # noqa: E402


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def fmt(v) -> str:
    """Display formatting; keeps the metrics table one dtype (Arrow needs that)."""
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:.4g}"
    return str(v)


def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def render():
    # --------------------------------------------------------------------- sidebar
    with st.sidebar:
        st.header("Settings")
        stain_label = st.radio("Stain", ["H&E", "Masson's trichrome"])
        stain = "he" if stain_label == "H&E" else "trichrome"

        um_in = st.number_input("Microns per pixel (0 = unknown)", min_value=0.0, value=0.0,
                                step=0.01, format="%.4f",
                                help="From your microscope/scanner. If 0, lengths are reported in pixels "
                                     "and densities per million px².")
        max_side = st.select_slider("Max image side (px). Larger is slower",
                                    options=[1500, 2000, 3000, 4000], value=2000)

        st.subheader("Colour normalisation")
        norm_label = st.selectbox("Method", ["None", "Macenko (H&E only)", "Reinhard (needs reference image)"])
        norm = {"None": "none", "Macenko (H&E only)": "macenko",
                "Reinhard (needs reference image)": "reinhard"}[norm_label]
        ref_file = None
        if norm == "reinhard":
            ref_file = st.file_uploader("Reference image (a slide with ideal staining)",
                                        type=["png", "jpg", "jpeg", "tif", "tiff"], key="ref")

        st.subheader("Matrix / hydrogel region")
        mask_files = st.file_uploader(
            "Optional hand-drawn matrix masks (white = matrix). Name each '<image name>_mask.png'. "
            "Recommended for final numbers.",
            type=["png", "tif", "tiff"], accept_multiple_files=True, key="masks")

        with st.expander("Advanced"):
            he_vec = "reference"
            if stain == "he":
                he_vec = st.selectbox("H&E stain vectors", ["reference", "estimate"],
                                      help="'estimate' fits vectors per image (Macenko). It can fail on "
                                           "images dominated by one colour; 'reference' is safer.")
            nuc_scale = st.slider("Nuclei sensitivity (threshold × Otsu; lower = fainter nuclei)",
                                  0.4, 1.4, 0.8, 0.05)
            pyk_dark = st.slider("Pyknotic: min relative darkness", 1.0, 2.0, 1.15, 0.05)
            pyk_area = st.slider("Pyknotic: max relative size", 0.2, 0.9, 0.60, 0.05)
            frag_area = st.slider("Fragment: max relative size", 0.05, 0.4, 0.25, 0.01)
            pale_dark = st.slider("Pale: max relative darkness", 0.2, 0.9, 0.60, 0.05)
            matrix_thr = st.number_input("Matrix threshold override (0 = automatic)", min_value=0.0,
                                         value=0.0, step=0.05)
            capsule_um = st.number_input("Capsule search distance (µm)", min_value=10.0,
                                         value=150.0, step=10.0)
            tissue_thr = st.number_input("Tissue threshold override (0 = automatic)", min_value=0.0,
                                         value=0.0, step=0.01)
            tri_blue = tri_red = None
            if stain == "trichrome":
                st.caption("Trichrome reference colours (leave unchecked for the built-in preset)")
                if st.checkbox("Use my own collagen / cytoplasm colours"):
                    tri_blue = hex_to_rgb(st.color_picker("Collagen (aniline blue)", "#4673B4"))
                    tri_red = hex_to_rgb(st.color_picker("Cytoplasm / muscle (red)", "#C84660"))

        line_w = st.slider("Overlay line width", 1, 4, 2)

    # ------------------------------------------------------------------ main panel
    files = st.file_uploader("Upload histology images", type=["png", "jpg", "jpeg", "tif", "tiff"],
                             accept_multiple_files=True)
    run = st.button("Run analysis", type="primary", disabled=not files)

    if "img_results" not in st.session_state:
        st.session_state["img_results"] = {}

    if run and files:
        reinhard_target = None
        if norm == "reinhard":
            if ref_file is None:
                st.error("Choose a reference image for Reinhard normalisation, or pick another method.")
                st.stop()
            ref_rgb, _ = load_image(ref_file, max_side=1500)
            reinhard_target = stains.lab_stats(ref_rgb)

        masks = {}
        for mf in mask_files or []:
            key = Path(mf.name).stem
            key = key[:-5] if key.endswith("_mask") else key
            masks[key] = np.asarray(Image.open(mf).convert("L")) > 127

        results = {}
        bar = st.progress(0.0, text="Starting...")
        for i, f in enumerate(files):
            bar.progress(i / len(files), text=f"Analysing {f.name} ({i + 1}/{len(files)})...")
            try:
                rgb, down = load_image(f, max_side=max_side)
                m = masks.get(Path(f.name).stem)
                note_mask = None
                if m is not None and m.shape != rgb.shape[:2]:
                    m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(
                        (rgb.shape[1], rgb.shape[0]), Image.NEAREST)) > 127
                    note_mask = "Mask was resized to the analysed image size."
                p = Params(
                    stain=stain,
                    um_per_px=(um_in * down) if um_in > 0 else None,
                    normalization=norm, reinhard_target=reinhard_target, he_vector_source=he_vec,
                    tissue_threshold=tissue_thr or None, nuclei_thresh_scale=nuc_scale,
                    pyknotic_rel_dark=pyk_dark, pyknotic_rel_area=pyk_area,
                    fragment_rel_area=frag_area, pale_rel_dark=pale_dark,
                    matrix_thresh=matrix_thr or None, capsule_search_um=capsule_um,
                    tri_blue_rgb=tri_blue, tri_red_rgb=tri_red)
                res = analyze(rgb, p, matrix_mask=m)
                if down > 1:
                    res.notes.append(f"Image downscaled {down:.2f}× for analysis "
                                     f"(effective {p.um_per_px:.3f} µm/px)." if p.um_per_px
                                     else f"Image downscaled {down:.2f}× for analysis.")
                if note_mask:
                    res.notes.append(note_mask)
                results[f.name] = res
            except Exception as e:  # keep the batch going
                st.warning(f"{f.name}: failed: {type(e).__name__}: {e}")
        bar.progress(1.0, text="Done")
        st.session_state["img_results"] = results

    results = st.session_state["img_results"]
    if not results:
        st.info("Upload one or more images, choose the stain and settings, then press **Run analysis**.")
        st.stop()

    # -------------------------------------------------------------- batch summary
    summary_df = pd.DataFrame([{"image": n, **r.summary} for n, r in results.items()])
    cells_df = pd.concat([r.cells.assign(image=n) for n, r in results.items() if len(r.cells)],
                         ignore_index=True) if any(len(r.cells) for r in results.values()) else pd.DataFrame()

    st.subheader("Batch summary")
    st.dataframe(summary_df, width="stretch", hide_index=True)
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download summary CSV", summary_df.to_csv(index=False).encode(),
                       "summary.csv", "text/csv")
    if len(cells_df):
        c2.download_button("Download per-nucleus CSV", cells_df.to_csv(index=False).encode(),
                           "nuclei.csv", "text/csv")
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        for n, r in results.items():
            z.writestr(f"{Path(n).stem}_overlay.png", png_bytes(viz.make_overlay(r, line_w)))
    c3.download_button("Download all overlays (zip)", zbuf.getvalue(), "overlays.zip", "application/zip")

    # ------------------------------------------------------------------ per image
    st.subheader("Per-image results")
    tabs = st.tabs(list(results))
    for tab, (name, r) in zip(tabs, results.items()):
        with tab:
            for n in r.notes:
                st.info(n)
            o1, o2, o3 = st.columns(3)
            show_matrix = o1.checkbox("Matrix outline", True, key=f"m_{name}")
            show_nuc = o2.checkbox("Nuclei outlines", True, key=f"n_{name}")
            show_col = o3.checkbox("Collagen tint", True, key=f"c_{name}", disabled=r.collagen is None)

            left, right = st.columns(2)
            left.image(r.rgb, caption="Analysed image", width="stretch")
            overlay = viz.make_overlay(r, line_w, show_matrix, show_col, show_nuc)
            right.image(overlay, width="stretch",
                        caption="Overlay: green intact, red pyknotic, blue pale, yellow fragment; "
                                "magenta = matrix edge" + ("; cyan tint = collagen" if r.collagen is not None else ""))
            st.download_button("Download overlay PNG", png_bytes(overlay), f"{Path(name).stem}_overlay.png",
                               "image/png", key=f"dl_{name}")

            a, b = st.columns(2)
            a.image(viz.density_heatmap(r), caption="Nuclear density map", width="stretch")
            sec = "Eosin" if r.params.stain == "he" else "Collagen"
            b.image(viz.channel_image(r.second, r.tissue), caption=f"{sec} channel (dark = strong)",
                    width="stretch")

            p1, p2 = st.columns(2)
            p1.pyplot(viz.distance_histogram(r))
            p2.pyplot(viz.state_bars(r))

            st.markdown("**Summary metrics**")
            st.dataframe(pd.DataFrame({"metric": list(r.summary), "value": [fmt(v) for v in r.summary.values()]}),
                         width="stretch", hide_index=True)
            with st.expander("Per-nucleus table"):
                st.dataframe(r.cells, width="stretch", hide_index=True)

    with st.expander("What these metrics do and don't mean"):
        st.markdown(
            "- **Pyknotic / pale / fragment** are H&E-level *morphological proxies* "
            "(small+dark+round; faint; tiny). They suggest cell death but do not prove apoptosis or "
            "necrosis; confirm with TUNEL or cleaved caspase-3.\n"
            "- **Pale** nuclei (karyolysis proxy) are the least reliable class; out-of-focus nuclei look pale too.\n"
            "- **Matrix region** is found automatically from stain intensity and is typically accurate to a "
            "few pixels on clean slides. For final numbers, upload a hand-drawn mask.\n"
            "- **Infiltration** = nuclei inside the matrix edge; depth is measured from that edge. "
            "**Capsule thickness** is the local width of the collagen layer touching the matrix, capped at "
            "the search distance, and assumes the capsule is thinner than the matrix wall.\n"
            "- **Clark-Evans R**: <1 clustered, ~1 random, >1 evenly spread.\n"
            "- Tested on synthetic slides only; verify thresholds on your own images.")
