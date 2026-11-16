# Weaver Lab histology analyzer

Give it a whole slide (Olympus .vsi, or a big TIFF/PNG/JPG) of H&E or Masson's trichrome
hydrogel constructs. It finds each tissue section, analyses it at cell resolution, and
reports nuclei, cell-state, infiltration, collagen-capsule and construct-geometry metrics
with overlays and CSV export. Two small machine-learning models learn from *your* labels
(hydrogel vs host tissue vs glass, and nucleus state) and improve as you teach them. No HistomicsTK, Docker or C++ build tools needed:
everything runs on NumPy / SciPy / scikit-image.

## Setup (Windows PowerShell)

```powershell
cd <this folder>
py -3.14 -m venv .venv            # any 64-bit Python 3.11+ works
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install -U pip
pip install --only-binary=:all: -r requirements.txt
streamlit run app/streamlit_app.py
```

## Using it: whole-slide mode

1. Choose **Whole slide**, paste the full path to the .vsi (or large image) and press **Open slide**.
   Pixel size is read from the file; override it if it is missing.
2. Check the overview: detected sections are outlined. Press **Run analysis**.
   Big sections are sampled in tiles (the summary says what % of tissue was analysed).
3. **Teach regions** tab: click example spots of glass / host tissue / hydrogel, then train.
   **Teach states** tab: correct the app's guess for a few dozen nuclei, then learn.
   Models are stored in `models/` and keep earlier examples when you teach more. Re-run afterwards.
4. Download the per-section summary CSV, the per-nucleus CSV and overlays.

If a .vsi will not open, export a pyramidal TIFF from Fiji/QuPath/OlyVIA and open that.

## Using it: single-image mode

1. Pick the stain, enter microns per pixel (from your scope/scanner; otherwise results are in pixels).
2. Upload one or more images, press **Run analysis**.
3. For final numbers, also upload hand-drawn matrix masks (white = hydrogel) named
   `<image name>_mask.png`; the automatic matrix detection is good on clean slides
   but is a heuristic.
4. Download the summary CSV (one row per image), the per-nucleus CSV, and overlay PNGs.

## What it measures

| Group | Metrics |
|---|---|
| Nuclei | count, density, area, shape, darkness, nearest-neighbour spacing, Clark-Evans R |
| Cell state (H&E-level proxies) | pyknosis index, pale-nuclei %, fragments per 100 nuclei, each also split inside vs outside the matrix |
| Infiltration | % cells inside matrix, depth median/p90/max, density in shells inside and outside the edge |
| Fibrosis (trichrome) | collagen area %, collagen inside matrix, capsule thickness median/mean/p90, capsule coverage |
| Construct geometry | matrix area, perimeter, wall thickness, void fraction, number of pieces |

The pixel model's hydrogel decision ignores nuclei (they are painted out first) and the
boundary is snapped to the half-maximum stain edge, so a handful of clicks is enough.

Pyknotic / pale / fragment are *morphological proxies*. They suggest cell death
but do not prove apoptosis vs necrosis; confirm with TUNEL or cleaved caspase-3.

## Tests

```powershell
pip install -r requirements-dev.txt
python -m pytest tests -q
```

The tests use synthetic slides with known ground truth (planted nuclei of each
state, a spiral hydrogel, a collagen capsule) and check counts, matrix overlap,
capsule thickness, calibration, resolution invariance, edge cases and file loading.

## Layout

```
src/weaver_histo/
  loading.py   image intake (PNG/JPG/TIFF, 8/16-bit, downscale)
  stains.py    Macenko / Reinhard normalisation, colour deconvolution
  segment.py   tissue, nuclei, matrix and collagen masks
  metrics.py   per-nucleus features, states, infiltration, capsule, geometry
  pipeline.py  analyze(image, Params) -> Result (single image)
  readers.py   VSI / TIFF slide readers
  sections.py  finds tissue sections on the overview
  slide.py     whole-slide two-scale analysis
  ml.py        trainable pixel and nucleus-state models
  viz.py       overlays, density map, plots
app/streamlit_app.py
tests/
```

## Known limits

- Validated on synthetic images only. Check thresholds on your own slides first
  (sliders under *Advanced*; the overlay shows what was detected).
- Automatic matrix detection assumes the hydrogel is paler / less stained than host
  tissue. If not, use hand-drawn masks.
- Capsule thickness assumes the capsule is thinner than the matrix wall and is capped
  at the search distance.
- Validated on synthetic slides only; reading a real .vsi via SlideIO has not been tested.
- A slide overview alone cannot give cell-level numbers; they come from the cell-resolution tiles.
- Models trained on one stain/scanner should be re-taught for another.
