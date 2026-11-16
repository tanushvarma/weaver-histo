"""Synthetic H&E / trichrome scenes with known ground truth (spiral hydrogel,
host tissue, planted nuclei of each state, optional collagen capsule)."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology

COLOURS = {
    "he": dict(bg=(246, 243, 246), matrix=(236, 208, 228), host=(226, 165, 200),
               capsule=None, intact=(100, 62, 148), pyknotic=(45, 25, 85),
               pale=(165, 115, 185), fragment=(60, 35, 100)),
    "trichrome": dict(bg=(246, 242, 242), matrix=(228, 200, 210), host=(205, 95, 115),
                      capsule=(85, 120, 190), intact=(60, 45, 80), pyknotic=(35, 25, 55),
                      pale=None, fragment=(40, 30, 60)),
}


def build_scene(kind="he", size=900, seed=0, capsule_px=18, n_intact_out=120, n_intact_in=20,
                n_pyk_out=10, n_pyk_in=5, n_frag=25, n_pale=12):
    rng = np.random.default_rng(seed)
    c = COLOURS[kind]
    H = W = size

    # spiral matrix
    theta = np.linspace(0, 4.0 * 2 * np.pi, 5000)
    r = 30 + 38 * theta / (2 * np.pi)
    pts = np.column_stack([W / 2 + r * np.cos(theta), H / 2 + r * np.sin(theta)]).astype(np.int32)
    m = np.zeros((H, W), np.uint8)
    cv2.polylines(m, [pts.reshape(-1, 1, 2)], False, 255, thickness=22)
    matrix = m > 0

    edt_out = ndi.distance_transform_edt(~matrix)
    host_zone = edt_out <= 100 + (capsule_px if c['capsule'] is not None else 0)
    capsule = np.zeros_like(matrix)
    if c["capsule"] is not None:
        capsule = (edt_out <= capsule_px) & ~matrix
    host = host_zone & ~matrix & ~capsule

    img = np.empty((H, W, 3), np.float32)
    img[:] = c["bg"]
    img[host] = c["host"]
    img[matrix] = c["matrix"]
    if c["capsule"] is not None:
        img[capsule] = c["capsule"]
    # texture
    low = ndi.gaussian_filter(rng.normal(0, 1, (H, W)), 6) * 9
    img += low[..., None] + rng.normal(0, 2.0, (H, W, 1))

    nuclei = []   # (type, y, x)
    occupied = np.zeros((H, W), bool)

    def place(zone, n, kind_, radius_margin=8):
        ys, xs = np.nonzero(ndi.binary_erosion(zone, iterations=radius_margin) & ~occupied)
        placed = 0
        tries = 0
        while placed < n and tries < n * 3000:
            tries += 1
            i = rng.integers(len(ys))
            y, x = int(ys[i]), int(xs[i])
            if occupied[max(0, y - 11):y + 12, max(0, x - 11):x + 12].any():
                continue
            occupied[max(0, y - 11):y + 12, max(0, x - 11):x + 12] = True
            nuclei.append((kind_, y, x))
            placed += 1
        assert placed == n, f"could only place {placed}/{n} {kind_}"

    inner = matrix
    # rare types first so the crowded host zone cannot starve them
    place(host, n_pyk_out, "pyknotic")
    place(inner, n_pyk_in, "pyknotic", radius_margin=7)
    place(host, n_frag, "fragment", radius_margin=4)
    if c["pale"] is not None:
        place(host, n_pale, "pale")
    place(inner, n_intact_in, "intact", radius_margin=7)
    place(host, n_intact_out, "intact")

    canvas = np.clip(img, 0, 255).astype(np.uint8).copy()
    for kind_, y, x in nuclei:
        col = tuple(int(v) for v in c[kind_])
        if kind_ in ("intact", "pale"):
            axes = (int(round(7 + rng.normal(0, 0.7))), int(round(5 + rng.normal(0, 0.6))))
            ang = float(rng.uniform(0, 180))
            cv2.ellipse(canvas, (x, y), axes, ang, 0, 360, col, -1, cv2.LINE_AA)
        elif kind_ == "pyknotic":
            cv2.circle(canvas, (x, y), 4, col, -1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (x, y), 2, col, -1, cv2.LINE_AA)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 0.8)

    truth = dict(
        matrix=matrix, capsule=capsule, capsule_px=capsule_px,
        n_intact=n_intact_out + n_intact_in, n_pyknotic=n_pyk_out + n_pyk_in,
        n_fragment=n_frag, n_pale=n_pale if c["pale"] is not None else 0,
        n_inside=n_intact_in + n_pyk_in,  # whole nuclei planted inside the matrix
        nuclei=nuclei,
    )
    return canvas, truth


if __name__ == "__main__":
    import sys
    from PIL import Image
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    for k in ("he", "trichrome"):
        im, t = build_scene(k)
        Image.fromarray(im).save(f"{out}/synthetic_{k}.png")
        print(k, "ok", {kk: v for kk, v in t.items() if kk.startswith("n_")})


# ---------------------------------------------------------------- whole slides
def build_slide(kind="he", n_sections=3, seed=0, upsample=2, gap_px=500,
                pyk_out=(10, 24, 10), pyk_in=(5, 3, 14)):
    """A glass slide (0.5 um/px) holding `n_sections` synthetic sections side by side.
    Each section is a build_scene() upsampled 2x (scene px were 1 um -> slide px are 0.5 um).
    Returns (rgb uint8, truth) with per-section ground truth in slide microns."""
    um_per_px = 1.0 / upsample
    scenes = []
    for i in range(n_sections):
        img, t = build_scene(kind, seed=seed + i, n_pyk_out=pyk_out[i % len(pyk_out)],
                             n_pyk_in=pyk_in[i % len(pyk_in)])
        scenes.append((cv2.resize(img, None, fx=upsample, fy=upsample, interpolation=cv2.INTER_CUBIC), t))
    h = scenes[0][0].shape[0]
    w_total = sum(s.shape[1] for s, _ in scenes) + gap_px * (n_sections + 1)
    canvas = np.full((h + 2 * gap_px // 2, w_total, 3), COLOURS[kind]["bg"], np.uint8)
    y0 = gap_px // 2
    truth = []
    x = gap_px
    for img, t in scenes:
        canvas[y0:y0 + img.shape[0], x:x + img.shape[1]] = img
        m = cv2.resize(t["matrix"].astype(np.uint8), None, fx=upsample, fy=upsample,
                       interpolation=cv2.INTER_NEAREST) > 0
        truth.append(dict(
            x0_um=x * um_per_px, y0_um=y0 * um_per_px, w_um=img.shape[1] * um_per_px, h_um=img.shape[0] * um_per_px,
            matrix_area_um2=float(m.sum()) * um_per_px ** 2, n_intact=t["n_intact"], n_pyknotic=t["n_pyknotic"],
            n_fragment=t["n_fragment"], n_pale=t["n_pale"], n_inside=t["n_inside"],
            n_whole=t["n_intact"] + t["n_pyknotic"] + t["n_pale"],
            pyk_idx_pct=100 * t["n_pyknotic"] / (t["n_intact"] + t["n_pyknotic"] + t["n_pale"]),
            infiltration_pct=100 * t["n_inside"] / (t["n_intact"] + t["n_pyknotic"] + t["n_pale"]),
            nuclei_um=[(k, (x + xx * upsample) * um_per_px, (y0 + yy * upsample) * um_per_px) for k, yy, xx in t["nuclei"]],
            matrix_full=(m, x, y0)))
        x += img.shape[1] + gap_px
    return canvas, dict(sections=truth, um_per_px=um_per_px, upsample=upsample)
