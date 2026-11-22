# Changelog

## 0.1.0

- Single-image analysis of H&E and Masson's trichrome images: nuclei, cell-state proxies (intact, pyknotic, pale, fragment), infiltration into the hydrogel, collagen capsule and construct geometry.
- Whole-slide analysis of `.vsi` and other formats through SlideIO, with automatic section detection and two-scale tile sampling.
- Trainable hydrogel pixel classifier and nucleus-state classifier (scikit-learn random forests) that keep earlier examples when retrained.
- Streamlit app for both modes, with CSV and overlay export.
- `weaver-histo` command for batch analysis without the app.
- Regression tests on synthetic slides with planted ground truth.
