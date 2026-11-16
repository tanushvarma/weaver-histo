"""Whole-slide mode: open a .vsi (or any large scan), find the tissue sections, analyse them at
cell resolution, and teach the tool what *your* hydrogel and nuclei look like."""
import base64
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from weaver_histo import ml, viz  # noqa: E402
from weaver_histo.pipeline import Params  # noqa: E402
from weaver_histo.readers import open_slide  # noqa: E402
from weaver_histo.sections import draw_sections, find_sections  # noqa: E402
from weaver_histo.slide import SlideParams, analyze_slide  # noqa: E402

MODELS = ROOT / "models"
CLASS_COLOURS = [(150, 150, 150), (0, 170, 0), (255, 0, 255), (0, 170, 255), (255, 160, 0)]
SPEED = {"Quick preview (up to 8 tiles per section)": 8,
         "Standard (up to 24 tiles per section)": 24,
         "Complete (every tile; slow)": None}


def _png(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _data_uri(arr) -> str:
    return "data:image/png;base64," + base64.b64encode(_png(arr)).decode()


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:.4g}"
    return str(v)


def _model_files(prefix):
    MODELS.mkdir(exist_ok=True)
    return sorted(p.name for p in MODELS.glob(f"{prefix}_*.joblib"))


def _close_reader():
    r = st.session_state.pop("slide_reader", None)
    if r is not None:
        try:
            r.close()
        except Exception:
            pass


def render():
    ss = st.session_state
    ss.setdefault("region_points", [])
    ss.setdefault("slide_res", None)

    # ------------------------------------------------------------------ sidebar
    with st.sidebar:
        st.header("Slide settings")
        stain = "he" if st.radio("Stain", ["H&E", "Masson's trichrome"]) == "H&E" else "trichrome"
        um_override = st.number_input("Microns per pixel override (0 = use the file's own value)",
                                      min_value=0.0, value=0.0, step=0.01, format="%.4f",
                                      help="VSI files carry their pixel size. Only fill this in if the app "
                                           "says the size is unknown, or it is wrong.")
        speed_label = st.selectbox("Speed", list(SPEED), index=1,
                                   help="The app measures nuclei tile by tile. Sampling tiles gives the same "
                                        "percentages and densities for the sampled tissue, much faster.")
        st.subheader("Trained models")
        region_file = st.selectbox("Hydrogel region model", ["None (use rules)"] + _model_files("region"))
        state_file = st.selectbox("Nuclear-state model", ["None (use rules)"] + _model_files("state"))
        with st.expander("Advanced"):
            an_um = st.select_slider("Nuclei resolution (um/px). Finer = slower", options=[0.25, 0.35, 0.5, 0.75, 1.0], value=0.5)
            tile_um = st.select_slider("Tile size (um)", options=[500, 750, 1000, 1500, 2000], value=1000)
            merge = st.slider("Group tissue pieces closer than (fraction of slide width)", 0.005, 0.08, 0.025, 0.005)
            nuc_scale = st.slider("Nuclei sensitivity (lower = fainter nuclei)", 0.4, 1.4, 0.8, 0.05)
            pyk_dark = st.slider("Pyknotic: min relative darkness", 1.0, 2.0, 1.15, 0.05)
            pyk_area = st.slider("Pyknotic: max relative size", 0.2, 0.9, 0.60, 0.05)
            frag_area = st.slider("Fragment: max relative size", 0.05, 0.4, 0.25, 0.01)
            pale_dark = st.slider("Pale: max relative darkness", 0.2, 0.9, 0.60, 0.05)
            capsule_um = st.number_input("Capsule search distance (um)", min_value=10.0, value=150.0, step=10.0)
            refine = st.checkbox("Snap hydrogel edge to the stain edge", True,
                                 help="Corrects the few-pixel offset of rule-based and trained masks.")

    tab_run, tab_region, tab_state, tab_help = st.tabs(
        ["1. Open and analyse", "2. Teach: hydrogel regions", "3. Teach: nuclei states", "How it works"])

    # =================================================================== tab 1
    with tab_run:
        st.markdown("**Step 1 - point the app at your slide.** The app runs on your PC, so give it the file's "
                    "location instead of uploading (VSI files are huge). In File Explorer, hold Shift, "
                    "right-click the `.vsi` file, choose *Copy as path*, and paste it below. "
                    "Keep the `_<name>_` folder that sits next to the `.vsi`.")
        path = st.text_input("Slide file", value=ss.get("slide_path", ""), placeholder=r"C:\Users\you\Documents\...\slide.vsi")
        c1, c2 = st.columns([1, 3])
        if c1.button("Open slide", type="primary", disabled=not path.strip()):
            _close_reader()
            ss["slide_res"] = None
            ss["region_points"] = []
            try:
                p = path.strip().strip('"')
                ss["slide_reader"] = open_slide(p, um_per_px=um_override or None)
                ss["slide_path"] = p
            except Exception as e:
                st.error(f"Could not open that file: {type(e).__name__}: {e}")
                st.info("If this is a VSI that will not open, export the main scan from Fiji or QuPath as a "
                        "pyramidal TIFF and open that instead.")

        reader = ss.get("slide_reader")
        if reader is None:
            st.info("Open a slide to begin.")
        else:
            infos = reader.scenes()
            if len(infos) > 1 and hasattr(reader, "select_scene"):
                labels = [str(i) for i in infos]
                cur = reader.scene_index
                pick = st.selectbox("Which image inside the file is the main scan? (largest is the default)",
                                    range(len(infos)), index=cur, format_func=lambda i: labels[i])
                if pick != cur:
                    reader.select_scene(pick, um_override or None)
                    ss["slide_res"] = None
            if um_override > 0:
                reader.um_per_px = um_override
            if not reader.um_per_px:
                st.warning("This file has no pixel size. Enter microns-per-pixel in the sidebar.")
            else:
                st.success(f"{getattr(reader, 'scene_name', '') or reader.path.name}: {reader.width_px:,} x "
                           f"{reader.height_px:,} px at {reader.um_per_px:.3f} um/px "
                           f"({reader.width_um / 1000:.1f} x {reader.height_um / 1000:.1f} mm)")
                with st.spinner("Reading overview..."):
                    ov, ov_um = reader.overview(1600)
                    secs = find_sections(ov, ov_um, merge_frac=merge, stain=stain)
                st.image(draw_sections(ov, secs, 2), caption=f"{len(secs)} tissue section(s) found "
                         "(magenta outline, numbered)", width="stretch")
                if not secs:
                    st.warning("No tissue found. Check the stain choice in the sidebar.")

                st.markdown("**Step 2 - run the analysis.**")
                if st.button("Run analysis", type="primary", disabled=not secs):
                    p = Params(stain=stain, nuclei_thresh_scale=nuc_scale, pyknotic_rel_dark=pyk_dark,
                               pyknotic_rel_area=pyk_area, fragment_rel_area=frag_area, pale_rel_dark=pale_dark,
                               capsule_search_um=capsule_um)
                    sp = SlideParams(stain=stain, analysis_um_per_px=an_um, tile_um=float(tile_um),
                                     max_tiles_per_section=SPEED[speed_label], section_merge_frac=merge,
                                     refine_matrix_edge=refine, params=p)
                    rm = ml.PixelClassifier.load(MODELS / region_file) if region_file != "None (use rules)" else None
                    sm = ml.StateClassifier.load(MODELS / state_file) if state_file != "None (use rules)" else None
                    bar = st.progress(0.0, text="Starting...")
                    try:
                        ss["slide_res"] = analyze_slide(reader, sp, region_model=rm, state_model=sm,
                                                        progress=lambda f, m: bar.progress(f, text=m))
                        ss["slide_sp"] = sp
                        ss.pop("review_df", None)
                        bar.progress(1.0, text="Done")
                    except Exception as e:
                        bar.empty()
                        st.error(f"Analysis failed: {type(e).__name__}: {e}")

        res = ss.get("slide_res")
        if res is not None:
            _show_results(res)

    # =================================================================== tab 2
    with tab_region:
        _teach_regions(stain)

    # =================================================================== tab 3
    with tab_state:
        _teach_states()

    with tab_help:
        st.markdown(HELP)


def _show_results(res):
    st.divider()
    st.subheader("Results")
    st.caption(res.scene_info)
    sdf = res.summary
    st.dataframe(sdf.map(_fmt) if hasattr(sdf, "map") else sdf.applymap(_fmt), width="stretch", hide_index=True)
    a, b = st.columns(2)
    a.download_button("Download section summary CSV", sdf.to_csv(index=False).encode(), "sections_summary.csv", "text/csv")
    if len(res.cells):
        b.download_button("Download per-nucleus CSV", res.cells.to_csv(index=False).encode(), "nuclei.csv", "text/csv")

    tabs = st.tabs([f"Section {s.id}" for s in res.sections])
    for tab, sr in zip(tabs, res.sections):
        with tab:
            for n in sr.notes:
                st.info(n)
            o1, o2, o3, o4 = st.columns(4)
            sm = o1.checkbox("Hydrogel edge", True, key=f"sm{sr.id}")
            sc = o2.checkbox("Nuclei dots", True, key=f"sc{sr.id}")
            st_ = o3.checkbox("Tile boxes", True, key=f"st{sr.id}")
            sk = o4.checkbox("Collagen tint", True, key=f"sk{sr.id}", disabled=sr.collagen is None)
            dot = st.slider("Dot size", 1, 6, 2, key=f"dot{sr.id}")
            img = viz.section_overlay(sr, dot, st_, sm, sc, sk)
            st.image(img, caption="Green intact, red pyknotic, blue pale, yellow fragment; magenta hydrogel edge; "
                     "cyan collagen; grey boxes = analysed tiles", width="stretch")
            st.download_button("Download this overlay", _png(img), f"section_{sr.id}_overlay.png", "image/png", key=f"dl{sr.id}")
            p1, p2 = st.columns(2)
            p1.pyplot(viz.section_distance_plot(sr))
            counts = sr.cells["state"].value_counts() if len(sr.cells) else {}
            p2.bar_chart(pd.Series({k: int(counts.get(k, 0)) for k in viz.STATE_COLOURS}))
            st.dataframe(pd.DataFrame({"metric": list(sr.summary), "value": [_fmt(v) for v in sr.summary.values()]}),
                         width="stretch", hide_index=True)


# =============================================================== teach: regions
def _teach_regions(stain):
    ss = st.session_state
    res = ss.get("slide_res")
    st.markdown("Teach the app what your **hydrogel** looks like. Click a few spots of each kind on the "
                "section picture; the app learns the colour and texture of each. The more varied your "
                "clicks (pale and dark hydrogel, thin and thick tissue), the better. Aim for 10-20 per class.")
    if res is None:
        st.info("Run an analysis first (tab 1); you teach on the pictures it produced.")
        return
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
    except ImportError:
        st.error("Click-to-label needs one extra package. In the terminal run: pip install streamlit-image-coordinates")
        return

    classes = list(ml.DEFAULT_PIXEL_CLASSES)
    sid = st.selectbox("Section to mark", [s.id for s in res.sections], key="reg_sec")
    sr = next(s for s in res.sections if s.id == sid)
    cls_name = st.radio("What are you clicking?", classes, horizontal=True,
                        format_func=lambda c: {"background": "Empty glass", "host_tissue": "Host tissue",
                                               "matrix": "Hydrogel"}.get(c, c))
    cnum = classes.index(cls_name) + 1
    radius = st.slider("Mark radius (px on this picture)", 2, 12, 5)

    H, W = sr.roi_rgb.shape[:2]
    disp_w = min(900, W)
    f = W / disp_w
    base = sr.roi_rgb.copy()
    import cv2
    for (psid, x, y, k) in ss["region_points"]:
        if psid == sid:
            cv2.circle(base, (int(x), int(y)), radius, CLASS_COLOURS[k - 1], -1)
            cv2.circle(base, (int(x), int(y)), radius, (0, 0, 0), 1)
    shown = Image.fromarray(base).resize((disp_w, int(H / f)))
    click = streamlit_image_coordinates(shown, key=f"reg_click_{sid}")
    if click and click.get("unix_time") != ss.get("reg_last_click"):
        ss["reg_last_click"] = click.get("unix_time")
        scale = W / max(1, click.get("width", disp_w))
        ss["region_points"].append((sid, click["x"] * scale, click["y"] * scale, cnum))
        st.rerun()

    pts = ss["region_points"]
    counts = {c: sum(1 for p in pts if p[3] == i + 1) for i, c in enumerate(classes)}
    st.caption("Marks so far: " + ", ".join(f"{c} {n}" for c, n in counts.items()))
    b1, b2 = st.columns(2)
    if b1.button("Undo last mark", disabled=not pts):
        pts.pop()
        st.rerun()
    if b2.button("Clear all marks", disabled=not pts):
        ss["region_points"] = []
        st.rerun()

    st.markdown("---")
    existing = ["Start a new model"] + _model_files("region")
    base_model = st.selectbox("Add to an existing model?", existing, key="reg_base")
    name = st.text_input("Name for this model", value="my_hydrogel")
    if st.button("Learn from my marks", type="primary", disabled=sum(counts.values()) < 4):
        try:
            sp = ss.get("slide_sp")
            model = (ml.PixelClassifier.load(MODELS / base_model) if base_model != "Start a new model"
                     else ml.PixelClassifier(classes, sr.region_um_per_px, stain))
            samples = []
            for s in res.sections:
                sp_pts = [(x, y, k) for (psid, x, y, k) in pts if psid == s.id]
                if sp_pts:
                    if abs(s.region_um_per_px - model.um_per_px) > 0.05:
                        st.warning(f"Section {s.id} was read at {s.region_um_per_px:.2f} um/px but the model uses "
                                   f"{model.um_per_px:.2f}; skipped.")
                        continue
                    samples.append((s.roi_rgb, ml.annotations_from_points(s.roi_rgb.shape, sp_pts, radius)))
            model.fit(samples)
            fname = "region_" + "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name) + ".joblib"
            MODELS.mkdir(exist_ok=True)
            model.save(MODELS / fname)
            ss["reg_model_preview"] = (sid, model)
            st.success(f"Saved models/{fname}. Accuracy on its own training examples (out-of-bag): "
                       f"{model.oob_accuracy_:.1%}. Choose it in the sidebar and press Run analysis again.")
        except Exception as e:
            st.error(f"Could not train: {type(e).__name__}: {e}")

    prev = ss.get("reg_model_preview")
    if prev and prev[0] == sid:
        pred = prev[1].predict(sr.roi_rgb)
        vis = (0.55 * sr.roi_rgb + 0.45 * np.array(CLASS_COLOURS, np.float32)[np.clip(pred, 1, 5) - 1]).astype(np.uint8)
        st.image(vis, caption="What the new model sees on this section (grey glass, green host tissue, magenta hydrogel)",
                 width="stretch")


# ================================================================ teach: states
def _teach_states():
    ss = st.session_state
    res, reader = ss.get("slide_res"), ss.get("slide_reader")
    st.markdown("Teach the app which nuclei are **intact, pyknotic, pale** or **fragments**. It shows you a sample "
                "of nuclei with its current guess. **Fix the wrong ones**; rows you leave alone count as "
                "confirmed. Include enough examples of the rare states.")
    if res is None or reader is None or not len(res.cells):
        st.info("Run an analysis first (tab 1).")
        return
    n = st.slider("How many nuclei to review", 24, 120, 60, 12)
    if st.button("Show nuclei to review"):
        cells = res.cells
        per = max(3, n // 4)
        parts = [g.sample(min(per, len(g)), random_state=0) for _, g in cells.groupby("state")]
        pick = pd.concat(parts).sample(frac=1, random_state=1).head(n).reset_index(drop=True)
        an_um = ss["slide_sp"].analysis_um_per_px
        crops = []
        for _, r in pick.iterrows():
            tile = reader.read_um(r["x_um"] - 18, r["y_um"] - 18, 36, 36, an_um)
            crops.append(_data_uri(np.asarray(Image.fromarray(tile).resize((tile.shape[1] * 2, tile.shape[0] * 2), Image.BICUBIC))))
        ss["review_df"] = pd.DataFrame({"nucleus": crops, "app guess": pick["state"], "your label": pick["state"],
                                        "section": pick["section_id"], "_idx": pick.index})
        ss["review_cells"] = pick
    rdf = ss.get("review_df")
    if rdf is None:
        return
    options = list(ml.DEFAULT_STATES)
    for extra in rdf["your label"].unique():
        if extra not in options:
            options.append(extra)
    cfg = {"nucleus": st.column_config.ImageColumn("nucleus", width="small"),
           "your label": st.column_config.SelectboxColumn("your label", options=options, required=True)}
    try:
        edited = st.data_editor(rdf.drop(columns=["_idx"]), column_config=cfg, hide_index=True, row_height=84,
                                disabled=["nucleus", "app guess", "section"], key="review_editor", width="stretch")
    except TypeError:
        edited = st.data_editor(rdf.drop(columns=["_idx"]), column_config=cfg, hide_index=True,
                                disabled=["nucleus", "app guess", "section"], key="review_editor", width="stretch")
    changed = int((edited["your label"] != edited["app guess"]).sum())
    st.caption(f"{changed} label(s) corrected of {len(edited)}.")
    existing = ["Start a new model"] + _model_files("state")
    base_model = st.selectbox("Add to an existing model?", existing, key="st_base")
    name = st.text_input("Name for this model", value="my_states")
    if st.button("Learn from these labels", type="primary"):
        try:
            model = ml.StateClassifier.load(MODELS / base_model) if base_model != "Start a new model" else ml.StateClassifier()
            cells = ss["review_cells"].reset_index(drop=True)
            model.fit(cells, edited["your label"].to_numpy())
            fname = "state_" + "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name) + ".joblib"
            MODELS.mkdir(exist_ok=True)
            model.save(MODELS / fname)
            cv = f"{model.cv_accuracy_:.1%}" if model.cv_accuracy_ is not None else "n/a (too few examples of some state)"
            st.success(f"Saved models/{fname}. It now knows {len(model.train_y)} labelled nuclei; "
                       f"cross-validated accuracy {cv}. Choose it in the sidebar and press Run analysis again.")
        except Exception as e:
            st.error(f"Could not train: {type(e).__name__}: {e}")


HELP = """
**What happens when you press Run analysis**

1. The app reads a small picture of the whole slide and finds each tissue section by colour
   (it ignores the label card, ink, and the slide holder).
2. For each section it makes a medium-resolution map (about 2 um per pixel): where tissue is, where the
   hydrogel is, and (trichrome) where collagen is. From that come the hydrogel geometry, collagen capsule and
   every cell's distance to the hydrogel edge.
3. It then reads the section in tiles at cell resolution (about 0.5 um per pixel) and finds every nucleus.
   Tiles overlap slightly so no nucleus is cut or counted twice.
4. Each nucleus is judged against the *typical nucleus of the whole slide*: size, darkness, roundness, texture.
   Small dark round = pyknotic; faint = pale; tiny = fragment.

**Where the learning comes in**

* **Hydrogel region model** (tab 2): you click examples of glass, host tissue and hydrogel. A random-forest
  learns their colour and texture at several scales and then classifies every pixel. This replaces the
  rule "hydrogel is the pale stuff", which breaks when staining varies.
* **Nuclear-state model** (tab 3): you correct the app's guesses on a sample of nuclei. It learns your
  definition of each state. Every time you teach more, it adds to what it already knew.
* Models are saved in the `models` folder. Share the file with the lab to share the training.

**Honest limits**

* Pyknotic / pale / fragment are H&E look-alikes for cell death. Confirm with TUNEL or cleaved caspase-3.
* Counts and percentages in *Quick* or *Standard* speed describe the **sampled tiles**; the summary says how many.
* Teaching needs a first analysis, and a taught model is then used on the *next* run.
"""
