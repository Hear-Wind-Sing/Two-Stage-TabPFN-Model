import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import inspect
import matplotlib as mpl
import matplotlib.pyplot as plt

DATA_PATH = r"E:\edgebrowser\数据提取\TabPFN\实验水体.xlsx"

SAVED_DIR = r"E:\edgebrowser\数据提取\TabPFN\tabpfn_2stage_saved1"

FIG_DIR = os.path.join(SAVED_DIR, "uncertainty_figs_95PI_fixed_size0428")
os.makedirs(FIG_DIR, exist_ok=True)

X_COLS_SLICE = (1, 15)
Y_COLS_SLICE = (15, 25)

DPI = 300

SINGLE_FIGSIZE = (10.5, 6.0)

AX_POS_SINGLE = [0.13, 0.16, 0.82, 0.72]

TITLE_SIZE = 20
LABEL_SIZE = 20
TICK_SIZE = 18
LEGEND_SIZE = 18

PI_ALPHA = 0.18
SHOW_BOUND_LINES = True
BOUND_LW = 1.1

TRUE_LW = 2.0
PRED_LW = 2.0
PRED_LS = "--"

XLAB = "Sample Index (sorted by y_true)"
YLAB = "Target Value"

CLIP_LOWER_TO_ZERO = True
ADD_CLASSIFIER_UNCERT = True

RESID_BINS = 30
RESID_Q = 0.95

PI_COLOR = "tab:blue"
PI_FILL_COLOR = "tab:blue"

SPINE_LW = 1.8
TICK_LW = 1.6

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams["axes.labelweight"] = "bold"
mpl.rcParams["axes.titleweight"] = "bold"
mpl.rcParams["axes.unicode_minus"] = False

def safe_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"[\\/:*?\"<>|]", "_", s)
    s = s.replace("\n", "_").replace("\r", "_")
    return s.strip()

def prepare_xy_for_one_target(df, X_cols, y_col):
    X_df = df[X_cols].copy()
    y_s = df[y_col].copy()

    data = pd.concat([X_df, y_s.rename("y")], axis=1)
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(axis=0)

    row_ids = data.index.values
    X = data[X_cols].values.astype(float)
    y = data["y"].values.astype(float)
    return X, y, row_ids

def predict_proba_1(clf, X):
    if clf is None:
        return np.ones(X.shape[0], dtype=float)

    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X)[:, 1]
        return np.asarray(p, dtype=float)

    pred = clf.predict(X)
    return np.asarray(pred, dtype=float)

def reg_predict_mean_std(reg, X):
    if reg is None:
        m = np.zeros(X.shape[0], dtype=float)
        return m, None, False

    try:
        sig = inspect.signature(reg.predict)
        if "return_std" in sig.parameters:
            out = reg.predict(X, return_std=True)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                mean, std = out
                return np.asarray(mean, float), np.asarray(std, float), True
    except Exception:
        pass

    try:
        sig = inspect.signature(reg.predict)
        if "return_cov" in sig.parameters:
            out = reg.predict(X, return_cov=True)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                mean, cov = out
                cov = np.asarray(cov, float)
                if cov.ndim == 2:
                    std = np.sqrt(np.clip(np.diag(cov), 0, None))
                    return np.asarray(mean, float), np.asarray(std, float), True
    except Exception:
        pass

    mean = reg.predict(X)
    return np.asarray(mean, float), None, False

def build_adaptive_halfwidth_from_train(y_tr, yhat_tr, bins=30, q=0.95):
    yhat_tr = np.asarray(yhat_tr, float).reshape(-1)
    y_tr = np.asarray(y_tr, float).reshape(-1)
    abs_res = np.abs(y_tr - yhat_tr)

    if len(y_tr) < max(30, bins * 3):
        hw = np.quantile(abs_res, q)
        return lambda yhat: np.full_like(np.asarray(yhat, float), hw, dtype=float)

    order = np.argsort(yhat_tr)
    yhat_s = yhat_tr[order]
    abs_s = abs_res[order]

    edges = np.quantile(yhat_s, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)

    if len(edges) < 5:
        hw = np.quantile(abs_s, q)
        return lambda yhat: np.full_like(np.asarray(yhat, float), hw, dtype=float)

    centers = []
    hw_list = []

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (yhat_s >= lo) & (yhat_s <= hi if i == len(edges) - 2 else yhat_s < hi)

        if mask.sum() < 5:
            continue

        centers.append((lo + hi) / 2)
        hw_list.append(float(np.quantile(abs_s[mask], q)))

    if len(hw_list) < 3:
        hw = np.quantile(abs_s, q)
        return lambda yhat: np.full_like(np.asarray(yhat, float), hw, dtype=float)

    centers = np.asarray(centers, float)
    hw_list = np.asarray(hw_list, float)

    def halfwidth(yhat):
        yhat = np.asarray(yhat, float)
        return np.interp(yhat, centers, hw_list, left=hw_list[0], right=hw_list[-1])

    return halfwidth

def two_stage_mean_pi(bundle, X_all, y_all, tr_idx, te_idx):
    thr = float(bundle.get("threshold", 0.5))
    clf = bundle.get("clf", None)
    reg = bundle.get("reg", None)

    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_te, y_te = X_all[te_idx], y_all[te_idx]

    p = predict_proba_1(clf, X_te)
    is_nz = (p >= thr)

    mu_all, std_all, has_std = reg_predict_mean_std(reg, X_te)

    pred_mean = np.zeros_like(mu_all, dtype=float)
    pred_mean[is_nz] = mu_all[is_nz]

    if has_std:
        var = np.zeros_like(std_all, dtype=float)
        var[is_nz] = np.square(std_all[is_nz])

        if ADD_CLASSIFIER_UNCERT:
            var = var + (p * (1.0 - p)) * np.square(mu_all)

        z = 1.96
        half = z * np.sqrt(np.clip(var, 0, None))

        lower = pred_mean - half
        upper = pred_mean + half

    else:
        p_tr = predict_proba_1(clf, X_tr)
        is_nz_tr = (p_tr >= thr)

        mu_tr, _, _ = reg_predict_mean_std(reg, X_tr)

        yhat_tr = np.zeros_like(mu_tr, dtype=float)
        yhat_tr[is_nz_tr] = mu_tr[is_nz_tr]

        halfwidth_fn = build_adaptive_halfwidth_from_train(
            y_tr, yhat_tr,
            bins=RESID_BINS,
            q=RESID_Q
        )

        half = halfwidth_fn(pred_mean)
        lower = pred_mean - half
        upper = pred_mean + half

    if CLIP_LOWER_TO_ZERO:
        lower = np.maximum(lower, 0.0)

    return y_te, pred_mean, lower, upper

def set_bold_ticks(ax):
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontweight("bold")

def plot_one(y_true, y_pred, y_lo, y_hi, out_png, target_name, panel_label=None):
    y_true = np.asarray(y_true, float).reshape(-1)
    y_pred = np.asarray(y_pred, float).reshape(-1)
    y_lo = np.asarray(y_lo, float).reshape(-1)
    y_hi = np.asarray(y_hi, float).reshape(-1)

    order = np.argsort(y_true)

    y_true = y_true[order]
    y_pred = y_pred[order]
    y_lo = y_lo[order]
    y_hi = y_hi[order]

    x = np.arange(len(y_true))

    fig = plt.figure(figsize=SINGLE_FIGSIZE, dpi=DPI)
    ax = fig.add_axes(AX_POS_SINGLE)

    ax.fill_between(
        x, y_lo, y_hi,
        color=PI_FILL_COLOR,
        alpha=PI_ALPHA,
        label="95% PI"
    )

    if SHOW_BOUND_LINES:
        ax.plot(x, y_lo, color=PI_COLOR, linewidth=BOUND_LW)
        ax.plot(x, y_hi, color=PI_COLOR, linewidth=BOUND_LW)

    ax.plot(x, y_true, linewidth=TRUE_LW, label="True")
    ax.plot(x, y_pred, linewidth=PRED_LW, linestyle=PRED_LS, label="Predicted Mean")

    ax.set_title(
        f"Target: {target_name}",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=16
    )

    ax.set_xlabel(XLAB, fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel(YLAB, fontsize=LABEL_SIZE, fontweight="bold")

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_SIZE,
        width=TICK_LW,
        length=7,
        direction="out"
    )

    set_bold_ticks(ax)

    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)
        spine.set_color("black")

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=LEGEND_SIZE
    )

    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontname("Times New Roman")
            text.set_fontweight("bold")

    if panel_label is not None:
        ax.text(
            -0.12, 1.08, f"({panel_label})",
            transform=ax.transAxes,
            fontsize=TITLE_SIZE,
            fontweight="bold",
            ha="left",
            va="center"
        )

    fig.savefig(out_png, dpi=DPI, format="png")
    plt.close(fig)

df = pd.read_excel(DATA_PATH)

X_cols = df.columns[X_COLS_SLICE[0]:X_COLS_SLICE[1]].tolist()
y_cols = df.columns[Y_COLS_SLICE[0]:Y_COLS_SLICE[1]].tolist()

letters = "abcdefghijklmnopqrstuvwxyz"

print("DATA:", DATA_PATH)
print("SAVED_DIR:", SAVED_DIR)
print("FIG_DIR:", FIG_DIR)
print("Targets:", y_cols)

for i, y_col in enumerate(y_cols):
    model_path = os.path.join(SAVED_DIR, f"model_{y_col}.joblib")
    pred_path = os.path.join(SAVED_DIR, f"pred_{y_col}.npz")

    if not (os.path.exists(model_path) and os.path.exists(pred_path)):
        print(
            f"[SKIP] {y_col} 缺文件：",
            ("no model" if not os.path.exists(model_path) else ""),
            ("no pred npz" if not os.path.exists(pred_path) else "")
        )
        continue

    bundle = joblib.load(model_path)
    npz = np.load(pred_path, allow_pickle=True)

    tr_idx = npz["tr_idx"].astype(int)
    te_idx = npz["te_idx"].astype(int)
    row_ids_saved = npz["row_ids"].astype(int)

    X_all, y_all, row_ids_now = prepare_xy_for_one_target(df, X_cols, y_col)

    if len(row_ids_now) == len(row_ids_saved) and np.all(row_ids_now == row_ids_saved):
        pass
    else:
        pos = {rid: j for j, rid in enumerate(row_ids_now)}
        mapped = []
        ok = True

        for rid in row_ids_saved:
            if rid not in pos:
                ok = False
                break
            mapped.append(pos[rid])

        if not ok:
            print(f"[SKIP] {y_col} 无法对齐 row_ids（清洗结果与保存不一致）")
            continue

        mapped = np.asarray(mapped, int)
        X_all = X_all[mapped]
        y_all = y_all[mapped]
        row_ids_now = row_ids_now[mapped]

    y_te, pred_mean, lower, upper = two_stage_mean_pi(
        bundle, X_all, y_all, tr_idx, te_idx
    )

    out_png = os.path.join(
        FIG_DIR,
        f"uncertainty_{safe_name(y_col)}_95PI_fixed_size.png"
    )

    panel = letters[i % len(letters)]

    plot_one(
        y_true=y_te,
        y_pred=pred_mean,
        y_lo=lower,
        y_hi=upper,
        out_png=out_png,
        target_name=y_col,
        panel_label=panel
    )

    reg = bundle.get("reg", None)
    has_std = False

    if reg is not None:
        try:
            sig = inspect.signature(reg.predict)
            has_std = ("return_std" in sig.parameters) or ("return_cov" in sig.parameters)
        except Exception:
            has_std = False

    print(
        f"[OK] {y_col} -> {out_png} | PI来源："
        f"{'TabPFN return_std/cov' if has_std else 'Residual-adaptive fallback'}"
    )

print("\n=== DONE ===")
print("All figures saved in:", FIG_DIR)