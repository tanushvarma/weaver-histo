"""Trainable models (scikit-learn random forests).

Two small models that learn from *your* labels, so the tool adapts to your
stain, scanner and tissue instead of relying on fixed thresholds:

* PixelClassifier  - labels every pixel of a low-resolution section image as
                     e.g. background / host_tissue / matrix (the hydrogel).
                     You teach it by marking a few example spots per class.
* StateClassifier  - labels every nucleus (intact / pyknotic / pale / fragment,
                     or classes you define) from its size, darkness, shape and
                     texture. You teach it by labelling example nuclei.

Both keep their training examples inside the saved model file, so teaching more
later *adds* to what they already know instead of starting over.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import color

from . import stains

MODEL_VERSION = 1
DEFAULT_PIXEL_CLASSES = ("background", "host_tissue", "matrix")
DEFAULT_STATES = ("intact", "pyknotic", "pale", "fragment")


def _gauss(a: np.ndarray, sigma: float) -> np.ndarray:
    return ndi.gaussian_filter(a, sigma, output=np.float32, mode="reflect")


def suppress_nuclei(rgb: np.ndarray, um_per_px: float, stain: str = "he") -> np.ndarray:
    """Paint nuclei over with the colour of their surroundings (normalised convolution)."""
    # Nuclei sit in hydrogel and host tissue alike; left in, they make hydrogel pixels look like host
    # tissue. The classifier should judge the material between the cells.
    from skimage import filters, morphology

    sig, _, _, floor = stains.separate(rgb, stain)
    sm = ndi.gaussian_filter(sig, max(0.7, 1.0 / um_per_px))
    tissue = color.rgb2gray(rgb) < 0.9
    v = sm[tissue]
    if v.size < 100:
        return rgb
    try:
        thr = max(float(filters.threshold_otsu(v)) * 0.8, floor)
    except ValueError:
        thr = floor
    nuc = ndi.binary_dilation(sm > thr, structure=morphology.disk(1))
    if not nuc.any() or nuc.mean() > 0.5:
        return rgb
    w = (~nuc).astype(np.float32)
    sg = max(1.5, 5.0 / um_per_px)
    den = np.maximum(_gauss(w, sg), 1e-3)
    out = rgb.astype(np.float32).copy()
    for c in range(3):
        fill = _gauss(out[..., c] * w, sg) / den
        out[..., c] = np.where(nuc, fill, out[..., c])
    return np.clip(out, 0, 255).astype(rgb.dtype) if rgb.dtype == np.uint8 else out


def pixel_features(
    rgb: np.ndarray, um_per_px: float, stain: str = "he", sigmas_um: Sequence[float] = (1.0, 2.5, 6.0)
) -> np.ndarray:
    """Multi-scale colour and texture features as float32 (H, W, F)."""
    # Scales are in microns, so a model trained at one um/px behaves the same on any image read at
    # that um/px. Channels: CIE Lab L, a, b and the two stain concentrations.
    rgb = suppress_nuclei(rgb, um_per_px, stain)
    lab = color.rgb2lab(rgb).astype(np.float32)
    sig, second, _, _ = stains.separate(rgb, stain)
    chans = [lab[..., 0], lab[..., 1], lab[..., 2], sig.astype(np.float32), second.astype(np.float32)]
    sg = [max(0.7, s / um_per_px) for s in sigmas_um]
    feats = []
    for c in chans:
        for s in sg:
            feats.append(_gauss(c, s))
    for c in (chans[0], chans[3]):
        for s in sg[:2]:
            feats.append(ndi.gaussian_gradient_magnitude(c, s, output=np.float32))
    for s in sg[:2]:
        feats.append(ndi.gaussian_laplace(chans[0], s, output=np.float32) * s * s)
        m1, m2 = _gauss(chans[0], s), _gauss(chans[0] ** 2, s)
        feats.append(np.sqrt(np.maximum(m2 - m1 * m1, 0)))
    return np.stack(feats, axis=-1)


def annotations_from_points(shape, points, radius_px: int = 6) -> np.ndarray:
    """Label map from clicks. points = [(x, y, class_number), ...] with class_number >= 1."""
    lab = np.zeros(shape[:2], np.uint8)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    for x, y, k in points:
        lab[(yy - y) ** 2 + (xx - x) ** 2 <= radius_px**2] = k
    return lab


class PixelClassifier:
    """Random forest on multi-scale pixel features."""

    def __init__(
        self,
        classes: Sequence[str] = DEFAULT_PIXEL_CLASSES,
        um_per_px: float = 2.0,
        stain: str = "he",
        sigmas_um: Sequence[float] = (1.0, 2.5, 6.0),
    ):
        self.classes = list(classes)
        self.um_per_px = float(um_per_px)
        self.stain = stain
        self.sigmas_um = tuple(sigmas_um)
        self.model = None
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.oob_accuracy_: float | None = None

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def class_number(self, name: str) -> int | None:
        return self.classes.index(name) + 1 if name in self.classes else None

    def _sample(self, rgb, labels, max_per_class, rng):
        F = pixel_features(rgb, self.um_per_px, self.stain, self.sigmas_um)
        Xs, ys = [], []
        for k in range(1, len(self.classes) + 1):
            idx = np.flatnonzero(labels.ravel() == k)
            if idx.size == 0:
                continue
            if idx.size > max_per_class:
                idx = rng.choice(idx, max_per_class, replace=False)
            Xs.append(F.reshape(-1, F.shape[-1])[idx])
            ys.append(np.full(idx.size, k, np.int16))
        return (np.concatenate(Xs), np.concatenate(ys)) if Xs else (None, None)

    def fit(
        self, samples, max_per_class: int = 20000, keep_previous: bool = True, seed: int = 0, n_estimators: int = 120
    ) -> PixelClassifier:
        """samples: list of (rgb, labels) where labels is 0 (unmarked) or 1..K (class number),
        and rgb was read at self.um_per_px. Adds to earlier training when keep_previous."""
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.default_rng(seed)
        parts_X, parts_y = [], []
        if keep_previous and self.X_train is not None:
            parts_X.append(self.X_train)
            parts_y.append(self.y_train)
        for rgb, lab in samples:
            if lab.shape[:2] != rgb.shape[:2]:
                raise ValueError("Label map and image sizes differ.")
            X, y = self._sample(rgb, lab, max_per_class, rng)
            if X is not None:
                parts_X.append(X)
                parts_y.append(y)
        if not parts_X:
            raise ValueError("No labelled pixels to learn from.")
        X, y = np.concatenate(parts_X), np.concatenate(parts_y)
        present = np.unique(y)
        if present.size < 2:
            raise ValueError("Mark examples of at least two different classes.")
        # cap each class so old and new examples stay balanced and the file stays small
        keep = []
        for k in present:
            idx = np.flatnonzero(y == k)
            keep.append(rng.choice(idx, min(idx.size, 3 * max_per_class), replace=False))
        keep = np.concatenate(keep)
        self.X_train, self.y_train = X[keep], y[keep]
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=14,
            min_samples_leaf=3,
            n_jobs=-1,
            class_weight="balanced_subsample",
            oob_score=True,
            random_state=seed,
        )
        self.model.fit(self.X_train, self.y_train)
        self.oob_accuracy_ = float(self.model.oob_score_)
        return self

    def predict(self, rgb: np.ndarray, stride: int | None = None) -> np.ndarray:
        """Class number (1..K) for every pixel. stride>1 predicts on a grid and fills in."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        h, w = rgb.shape[:2]
        if stride is None:
            stride = 2 if h * w > 1.5e6 else 1
        F = pixel_features(rgb, self.um_per_px, self.stain, self.sigmas_um)
        sub = F[::stride, ::stride]
        pred = self.model.predict(sub.reshape(-1, sub.shape[-1]).astype(np.float32))
        pred = pred.reshape(sub.shape[:2]).astype(np.uint8)
        if stride > 1:
            pred = np.repeat(np.repeat(pred, stride, axis=0), stride, axis=1)[:h, :w]
        return pred

    def save(self, path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            dict(
                kind="pixel",
                version=MODEL_VERSION,
                classes=self.classes,
                um_per_px=self.um_per_px,
                stain=self.stain,
                sigmas_um=self.sigmas_um,
                model=self.model,
                X_train=self.X_train,
                y_train=self.y_train,
                oob=self.oob_accuracy_,
            ),
            path,
            compress=3,
        )

    @classmethod
    def load(cls, path) -> PixelClassifier:
        import joblib

        d = joblib.load(path)
        if d.get("kind") != "pixel":
            raise ValueError("Not a pixel-classifier model file.")
        m = cls(d["classes"], d["um_per_px"], d["stain"], d["sigmas_um"])
        m.model, m.X_train, m.y_train, m.oob_accuracy_ = d["model"], d["X_train"], d["y_train"], d.get("oob")
        return m


STATE_FEATURES = [
    "rel_area",
    "rel_dark",
    "solidity",
    "eccentricity",
    "circularity",
    "texture",
    "core_frac",
    "peak_ratio",
]


def state_features(cells: pd.DataFrame) -> pd.DataFrame:
    """Size/density/shape/texture descriptors, all unit-free (relative to the typical nucleus)."""
    c = cells
    eps = 1e-6
    # works on the internal pixel table and on the exported micron table alike (only a ratio is used)
    core = c["core_area_px"] if "core_area_px" in c else c["core_area_um2"]
    area = c["area_px"] if "area_px" in c else c["area_um2"]
    return pd.DataFrame(
        {
            "rel_area": c["rel_area"].to_numpy(float),
            "rel_dark": c["rel_dark"].to_numpy(float),
            "solidity": c["solidity"].to_numpy(float),
            "eccentricity": c["eccentricity"].to_numpy(float),
            "circularity": c["circularity"].to_numpy(float),
            "texture": c["std_signal"].to_numpy(float) / np.maximum(c["mean_signal"].to_numpy(float), eps),
            "core_frac": core.to_numpy(float) / np.maximum(area.to_numpy(float), eps),
            "peak_ratio": c["max_signal"].to_numpy(float) / np.maximum(c["core_mean_signal"].to_numpy(float), eps),
        }
    )[STATE_FEATURES]


class StateClassifier:
    """Random forest that learns nuclear states from labelled examples."""

    def __init__(self):
        self.model = None
        self.train_X: pd.DataFrame | None = None
        self.train_y: np.ndarray | None = None
        self.cv_accuracy_: float | None = None

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    @property
    def classes(self) -> list[str]:
        return list(self.model.classes_) if self.model is not None else []

    def fit(self, cells: pd.DataFrame, labels, keep_previous: bool = True, seed: int = 0) -> StateClassifier:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        X = state_features(cells)
        y = np.asarray(labels, dtype=object)
        ok = pd.notna(y) & (y != "")
        X, y = X[ok].reset_index(drop=True), y[ok]
        if keep_previous and self.train_X is not None:
            X = pd.concat([self.train_X, X], ignore_index=True)
            y = np.concatenate([self.train_y, y])
        classes, counts = np.unique(y, return_counts=True)
        if classes.size < 2:
            raise ValueError("Label examples of at least two different states.")
        if len(y) < 10:
            raise ValueError("Label at least ~10 nuclei (more is better).")
        self.train_X, self.train_y = X, y
        self.model = RandomForestClassifier(
            n_estimators=250, min_samples_leaf=2, n_jobs=-1, class_weight="balanced", random_state=seed
        )
        folds = int(min(5, counts.min()))
        if folds >= 2:
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            self.cv_accuracy_ = float(cross_val_score(self.model, X, y, cv=cv).mean())
        else:
            self.cv_accuracy_ = None
        self.model.fit(X, y)
        return self

    def predict(self, cells: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Returns (state array, confidence array)."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        X = state_features(cells)
        proba = self.model.predict_proba(X)
        idx = proba.argmax(axis=1)
        return self.model.classes_[idx].astype(object), proba.max(axis=1)

    def save(self, path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            dict(
                kind="state",
                version=MODEL_VERSION,
                model=self.model,
                train_X=self.train_X,
                train_y=self.train_y,
                cv=self.cv_accuracy_,
            ),
            path,
            compress=3,
        )

    @classmethod
    def load(cls, path) -> StateClassifier:
        import joblib

        d = joblib.load(path)
        if d.get("kind") != "state":
            raise ValueError("Not a nuclear-state model file.")
        m = cls()
        m.model, m.train_X, m.train_y, m.cv_accuracy_ = d["model"], d["train_X"], d["train_y"], d.get("cv")
        return m
