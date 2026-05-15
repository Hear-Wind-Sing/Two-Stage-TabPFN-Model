```python
import os
import re
import glob
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

import matplotlib as mpl
import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import permutation_importance

import logging
logging.getLogger("analytics").setLevel(logging.CRITICAL)
logging.getLogger("segment").setLevel(logging.CRITICAL)
os.environ.setdefault("SEGMENT_WRITE_KEY", "")
os.environ.setdefault("ANALYTICS_WRITE_KEY", "")
os.environ.setdefault("TABPFN_DISABLE_ANALYTICS", "1")

DATA_PATH = r"E:\edgebrowser\数据提取\TabPFN\实验水体.xlsx"
MODEL_DIR = r"E:\edgebrowser\数据提取\TabPFN\tabpfn_2stage_saved1"

PLOT_DIR = os.path.join(MODEL_DIR, "interpretability_plots_600dpi_fast_final")
DATA_DIR = os.path.join(MODEL_DIR, "interpretability_data_fast_final")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

DO_PERM = True
DO_ALE  = True

RANDOM_STATE = 42

TOP_N_FEATURES = 5
MAX_SAMPLES_FOR_CURVES = 500
GRID_POINTS = 15
ALE_BINS = 10

PERM_SCORING = "r2"
PERM_N_REPEATS = 5
MAX_SAMPLES_FOR_PERM = 600

ONLY_TARGETS = None

FONT_FAMILY = "Times New Roman"
FS_BASE  = 12
FS_TITLE = 14
FS_LABEL = 12
FS_TICK  = 11

mpl.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": FS_BASE,
    "axes.titlesize": FS_TITLE,
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_TICK,
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",
})

LABEL_TEMPLATES = {
    "PERM": {"title": "{target} | Permutation importance", "xlabel": "Permutation importance ({perm_scoring})", "ylabel": ""},
    "ALE":  {"title": "{target} | ALE | {feature}",        "xlabel": "{feature}", "ylabel": "ALE"},
}

def get_labels(plot_type: str, target: str, feature: str | None = None):
    tpl = LABEL_TEMPLATES.get(plot_type, {})
    perm_scoring = globals().get("PERM_SCORING", "r2")
    def fmt(s: str):
        return (s or "").format(
            target=target,
            feature=feature if feature is not None else "",
            perm_scoring=perm_scoring
        )
    return fmt(tpl.get("title", "")), fmt(tpl.get("xlabel", "")), fmt(tpl.get("ylabel", ""))

def _sanitize(name: str) -> str:
    name = str(name)
    return re.sub(r"[\\/:*?\"<>|]+", "_", name).strip()

def _subsample_idx(n, max_n, seed=42):
    if max_n is None or n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(np.arange(n), size=max_n, replace=False)

def _grid_from_quantiles(x: np.ndarray, n_points: int):
    qs = np.linspace(0.02, 0.98, n_points)
    grid = np.quantile(x, qs)
    grid = np.unique(grid)
    return grid

def _prepare_X_df(df: pd.DataFrame, feature_cols, fill_values: dict):
    X_df = df[feature_cols].copy()
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    if fill_values is None:
        fill_values = X_df.median(numeric_only=True).to_dict()
    X_df = X_df.fillna(fill_values)
    return X_df

def _prepare_y(df: pd.DataFrame, target_col: str):
    y = pd.to_numeric(df[target_col], errors="coerce")
    y = y.replace([np.inf, -np.inf], np.nan)
    return y

def _plot_save(figpath, dpi=600):
    plt.tight_layout()
    plt.savefig(figpath, dpi=dpi, bbox_inches="tight")
    plt.close()

def _two_stage_predict(bundle, X_np: np.ndarray) -> np.ndarray:
    clf = bundle.get("clf", None)
    reg = bundle.get("reg", None)
    thr = float(bundle.get("threshold", 0.5))

    if reg is None:
        return np.zeros(X_np.shape[0], dtype=float)

    if clf is not None and hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_np)[:, 1]
        is_nz = (proba >= thr)
    elif clf is not None and hasattr(clf, "predict"):
        is_nz = (clf.predict(X_np) == 1)
    else:
        is_nz = np.ones(X_np.shape[0], dtype=bool)

    yhat = np.zeros(X_np.shape[0], dtype=float)
    if np.any(is_nz):
        yhat[is_nz] = np.asarray(reg.predict(X_np[is_nz]), dtype=float).ravel()
    return yhat

class TwoStageEstimatorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, bundle, feature_cols):
        self.bundle = bundle
        self.feature_cols = list(feature_cols)

    def fit(self, X, y=None):
        self.n_features_in_ = len(self.feature_cols)
        self.feature_names_in_ = np.array(self.feature_cols, dtype=object)
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X_np = X[self.feature_cols].values.astype(float, copy=False)
        else:
            X_np = np.asarray(X, dtype=float)
        return _two_stage_predict(self.bundle, X_np)

def compute_perm(bundle, X_df, y_s, target):
    feature_cols = bundle["feature_cols"]
    y = y_s.values.astype(float)
    mask = ~np.isnan(y)

    if mask.sum() < 20:
        var = X_df[feature_cols].var(axis=0).sort_values(ascending=False)
        top_feats = list(var.index[:TOP_N_FEATURES])
        perm_df = pd.DataFrame({"feature": feature_cols, "importance_mean": np.nan, "importance_std": np.nan})
        return perm_df, top_feats

    X_pi = X_df.loc[mask, feature_cols].copy()
    y_pi = y[mask]

    sel = _subsample_idx(len(X_pi), MAX_SAMPLES_FOR_PERM, seed=RANDOM_STATE)
    X_pi = X_pi.iloc[sel].copy()
    y_pi = y_pi[sel]

    est = TwoStageEstimatorWrapper(bundle, feature_cols).fit(X_pi, y_pi)

    res = permutation_importance(
        est, X_pi, y_pi,
        scoring=PERM_SCORING,
        n_repeats=PERM_N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    perm_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": res.importances_mean,
        "importance_std": res.importances_std
    }).sort_values("importance_mean", ascending=False)

    top_feats = perm_df["feature"].head(TOP_N_FEATURES).tolist()

    figpath = os.path.join(PLOT_DIR, f"{_sanitize(target)}__perm_importance_top{TOP_N_FEATURES}.png")
    plt.figure(figsize=(8, 5))
    show = perm_df.head(TOP_N_FEATURES).iloc[::-1]
    plt.barh(show["feature"], show["importance_mean"], xerr=show["importance_std"])
    title, xlabel, ylabel = get_labels("PERM", target, None)
    plt.xlabel(xlabel)
    plt.title(title + f" (Top {TOP_N_FEATURES})")
    _plot_save(figpath, dpi=600)

    return perm_df, top_feats

def compute_ale(bundle, X_np, feat_idx, target, feat_name, bins=10):
    x = X_np[:, feat_idx].astype(float)

    qs = np.linspace(0, 1, bins + 1)
    edges = np.quantile(x, qs)
    edges = np.unique(edges)
    if len(edges) < 3:
        return pd.DataFrame(columns=["target", "feature", "x_mid", "ale"])

    bin_id = np.digitize(x, edges[1:-1], right=True)
    K = len(edges) - 1

    deltas = np.zeros(K, dtype=float)
    counts = np.zeros(K, dtype=int)

    for k in range(K):
        idx = np.where(bin_id == k)[0]
        if idx.size == 0:
            continue
        low, high = edges[k], edges[k + 1]

        X_low = X_np[idx].copy()
        X_high = X_np[idx].copy()
        X_low[:, feat_idx] = low
        X_high[:, feat_idx] = high

        y_high = _two_stage_predict(bundle, X_high)
        y_low = _two_stage_predict(bundle, X_low)
        deltas[k] = float(np.mean(y_high - y_low))
        counts[k] = int(idx.size)

    ale = np.cumsum(deltas)
    x_mid = 0.5 * (edges[:-1] + edges[1:])

    w = counts / max(counts.sum(), 1)
    ale_centered = ale - np.sum(ale * w)

    ale_df = pd.DataFrame({
        "target": target,
        "feature": feat_name,
        "x_mid": x_mid,
        "ale": ale_centered
    })

    figpath = os.path.join(PLOT_DIR, f"{_sanitize(target)}__ALE__{_sanitize(feat_name)}.png")
    plt.figure(figsize=(7.2, 4.5))
    plt.plot(x_mid, ale_centered, marker="o", linewidth=1.2, markersize=3)

    title, xlabel, ylabel = get_labels("ALE", target, feat_name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    _plot_save(figpath, dpi=600)
    return ale_df

def main():
    df = pd.read_excel(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    joblibs = sorted(glob.glob(os.path.join(MODEL_DIR, "model_*.joblib")))
    if not joblibs:
        raise FileNotFoundError(f"在 {MODEL_DIR} 没找到 model_*.joblib")

    print(f"[Info] 模型数: {len(joblibs)}")
    print(f"[Info] 图输出: {PLOT_DIR}")
    print(f"[Info] 数据输出: {DATA_DIR}")

    for mp in joblibs:
        bundle = joblib.load(mp)
        target = bundle.get("target", None)
        if target is None:
            base = os.path.basename(mp)
            target = re.sub(r"^model_|\.joblib$", "", base)

        if ONLY_TARGETS is not None and target not in ONLY_TARGETS:
            continue

        feature_cols = bundle["feature_cols"]
        fill_values = bundle.get("fill_values", None)

        if target not in df.columns:
            print(f"[Skip] {target}: 数据缺少目标列")
            continue
        miss = [c for c in feature_cols if c not in df.columns]
        if miss:
            print(f"[Skip] {target}: 数据缺少特征列（前10个）{miss[:10]}")
            continue

        print(f"\n========== Target: {target} ==========", flush=True)

        X_df_all = _prepare_X_df(df, feature_cols, fill_values)
        y_s_all = _prepare_y(df, target)

        sel_curve = _subsample_idx(len(X_df_all), MAX_SAMPLES_FOR_CURVES, seed=RANDOM_STATE)
        X_df = X_df_all.iloc[sel_curve].copy()
        y_s = y_s_all.iloc[sel_curve].copy()
        X_np = X_df[feature_cols].values.astype(float, copy=False)

        if DO_PERM:
            print("  -> computing permutation importance ...", flush=True)
            perm_df, top_feats = compute_perm(bundle, X_df, y_s, target)
        else:
            var = X_df[feature_cols].var(axis=0).sort_values(ascending=False)
            top_feats = list(var.index[:TOP_N_FEATURES])
            perm_df = pd.DataFrame()

        print(f"  -> Top features: {top_feats}", flush=True)

        all_ale = []

        for i, feat in enumerate(top_feats, 1):
            feat_idx = feature_cols.index(feat)
            print(f"  -> [{i}/{len(top_feats)}] feature={feat}", flush=True)

            if DO_ALE:
                all_ale.append(compute_ale(bundle, X_np, feat_idx, target, feat, bins=ALE_BINS))

        out_xlsx = os.path.join(DATA_DIR, f"interpretability_{_sanitize(target)}.xlsx")
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            meta = pd.DataFrame({
                "key": [
                    "target", "model_path", "n_rows_curve",
                    "TOP_N_FEATURES", "MAX_SAMPLES_FOR_CURVES",
                    "GRID_POINTS", "ALE_BINS",
                    "DO_PERM", "DO_ALE",
                    "PERM_N_REPEATS", "MAX_SAMPLES_FOR_PERM", "PERM_SCORING",
                    "threshold", "zero_eps",
                    "FONT_FAMILY", "FS_BASE", "FS_TITLE", "FS_LABEL", "FS_TICK"
                ],
                "value": [
                    target, mp, len(X_df),
                    TOP_N_FEATURES, MAX_SAMPLES_FOR_CURVES,
                    GRID_POINTS, ALE_BINS,
                    DO_PERM, DO_ALE,
                    PERM_N_REPEATS, MAX_SAMPLES_FOR_PERM, PERM_SCORING,
                    bundle.get("threshold", None), bundle.get("zero_eps", None),
                    FONT_FAMILY, FS_BASE, FS_TITLE, FS_LABEL, FS_TICK
                ]
            })
            meta.to_excel(w, sheet_name="meta", index=False)

            if DO_PERM and not perm_df.empty:
                perm_df.to_excel(w, sheet_name="perm", index=False)
            if len(all_ale) > 0:
                pd.concat(all_ale, ignore_index=True).to_excel(w, sheet_name="ale", index=False)

        plt.close("all")
        print(f"[OK] {target}: 图已保存到 {PLOT_DIR}；数据已保存到 {out_xlsx}", flush=True)

    print("\n[Done] 全部完成。")

if __name__ == "__main__":
    main()
```