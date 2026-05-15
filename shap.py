# -*- coding: utf-8 -*-
import os
import re
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["axes.unicode_minus"] = False

DATA_PATH = r"E:\edgebrowser\数据提取\TabPFN\实验水体.xlsx"
MODEL_DIR = r"E:\edgebrowser\数据提取\TabPFN\tabpfn_2stage_saved1"

SHAP_DIR = os.path.join(MODEL_DIR, "shap_outputs_TEST_ypos_only_fast")
os.makedirs(SHAP_DIR, exist_ok=True)

ONLY_TARGETS = None
RANDOM_STATE = 42
Y_ZERO_TOL = 0.0

N_EXPLAIN_MAX = 150 
N_BINS_POS = 8 

BACKGROUND_SIZE = 64 
MAX_EVALS = 128 

MAKE_PLOTS = True
MAX_DISPLAY = 20
DPI = 600

SAVE_WIDE_CSV = True

def prepare_xy_for_one_target(df, X_cols, y_col):
    X_df = df[X_cols].copy()
    y_s  = df[y_col].copy()

    data = pd.concat([X_df, y_s.rename("y")], axis=1)
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(axis=0)

    row_ids = data.index.values
    X = data[X_cols].values
    y = data["y"].values.astype(float)
    return X, y, row_ids

def sample_explain_from_test_ypos(y_test, n_explain_max=150, n_bins=8, seed=42, y_zero_tol=0.0):
    rng = np.random.default_rng(seed)
    y_test = np.asarray(y_test).reshape(-1)

    idx_pos = np.where(y_test > y_zero_tol)[0]
    if len(idx_pos) == 0:
        return np.array([], dtype=int), {"n_pos_available": 0, "n_explain": 0}

    n_total = min(n_explain_max, len(idx_pos))

    y_pos = y_test[idx_pos]
    if np.unique(y_pos).size < 2 or len(idx_pos) < n_bins:
        pick = rng.choice(idx_pos, size=n_total, replace=False)
        rng.shuffle(pick)
        return pick, {"mode": "random", "n_pos_available": int(len(idx_pos)), "n_explain": int(len(pick))}

    s = pd.Series(y_pos, index=idx_pos)
    bins = pd.qcut(s, q=n_bins, duplicates="drop")
    groups = s.groupby(bins)

    n_bins_eff = len(groups)
    per_bin = max(1, n_total // n_bins_eff)

    pick_list = []
    for _, g in groups:
        cand = g.index.to_numpy()
        k = min(per_bin, len(cand))
        if k > 0:
            pick_list.append(rng.choice(cand, size=k, replace=False))
    pick = np.concatenate(pick_list) if len(pick_list) else np.array([], dtype=int)

    if len(pick) < n_total:
        remain = np.setdiff1d(idx_pos, pick, assume_unique=False)
        need = n_total - len(pick)
        if len(remain) > 0:
            add = rng.choice(remain, size=min(need, len(remain)), replace=False)
            pick = np.concatenate([pick, add])

    if len(pick) > n_total:
        pick = rng.choice(pick, size=n_total, replace=False)

    rng.shuffle(pick)
    return pick, {
        "mode": "qcut",
        "n_pos_available": int(len(idx_pos)),
        "n_explain": int(len(pick)),
        "n_bins": int(n_bins),
    }

def make_predict_fn_from_bundle(bundle):
    clf = bundle["clf"]
    reg = bundle["reg"]
    thr = float(bundle["threshold"])

    def predict_fn(X_input):
        X_np = np.asarray(X_input)
        n = X_np.shape[0]
        pred = np.zeros(n, dtype=float)
        if reg is None:
            return pred

        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_np)[:, 1]
            is_nz = (proba >= thr)
        else:
            is_nz = (clf.predict(X_np) == 1)

        if np.any(is_nz):
            pred[is_nz] = reg.predict(X_np[is_nz])
        return pred

    return predict_fn

def plot_bar_importance(shap_values, feature_names, out_png, title):
    imp = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(imp)[::-1]
    names = [feature_names[i] for i in order]
    vals = imp[order]

    plt.figure(figsize=(7.2, max(4.0, 0.28 * len(names))))
    plt.barh(names[::-1], vals[::-1])
    plt.xlabel("mean(|SHAP value|)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=DPI, format="png")
    plt.close()

def main():
    df = pd.read_excel(DATA_PATH)

    model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("model_") and f.endswith(".joblib")]
    if not model_files:
        raise FileNotFoundError(f"在 {MODEL_DIR} 未找到 model_*.joblib")

    targets = []
    for f in model_files:
        t = re.sub(r"^model_", "", f)
        t = re.sub(r"\.joblib$", "", t)
        targets.append((t, os.path.join(MODEL_DIR, f)))

    if ONLY_TARGETS is not None:
        targets = [x for x in targets if x[0] in set(ONLY_TARGETS)]

    print(f"[Info] Found {len(targets)} model(s). SHAP will use TEST set only (y>0 only).")

    for target, model_path in targets:
        pred_npz = os.path.join(MODEL_DIR, f"pred_{target}.npz")
        if not os.path.exists(pred_npz):
            print(f"[SKIP] pred file not found: {pred_npz}")
            continue

        print(f"\n========== Target: {target} ==========")
        bundle = joblib.load(model_path)
        X_cols = bundle["feature_cols"]
        y_col = bundle["target"]

        X_clean, y_clean, row_ids = prepare_xy_for_one_target(df, X_cols, y_col)
        if len(y_clean) < 20:
            print("  [SKIP] clean 样本太少")
            continue

        z = np.load(pred_npz, allow_pickle=True)
        te_idx = z["te_idx"].astype(int)

        X_test = X_clean[te_idx]
        y_test = y_clean[te_idx]
        row_test = row_ids[te_idx]

        pick_local, meta_pick = sample_explain_from_test_ypos(
            y_test,
            n_explain_max=N_EXPLAIN_MAX,
            n_bins=N_BINS_POS,
            seed=RANDOM_STATE,
            y_zero_tol=Y_ZERO_TOL
        )
        if len(pick_local) == 0:
            print("  [SKIP] test 集没有 y>0 样本，无法做 y>0 机制 SHAP")
            continue

        X_explain = X_test[pick_local]
        y_explain = y_test[pick_local]
        row_explain = row_test[pick_local]

        print(f"  test_n={len(y_test)} | test_y>0={meta_pick['n_pos_available']} | explain_n={len(pick_local)}")

        out_dir = os.path.join(SHAP_DIR, target)
        os.makedirs(out_dir, exist_ok=True)

        rng = np.random.default_rng(RANDOM_STATE)
        bg_n = min(BACKGROUND_SIZE, len(X_explain))
        bg_local = rng.choice(np.arange(len(X_explain)), size=bg_n, replace=False)
        X_bg = X_explain[bg_local]

        predict_fn = make_predict_fn_from_bundle(bundle)

        explainer = shap.PermutationExplainer(predict_fn, X_bg)
        exp = explainer(X_explain, max_evals=MAX_EVALS)

        shap_values = np.asarray(exp.values)
        base_values = np.asarray(exp.base_values)

        npz_path = os.path.join(out_dir, f"shap_TEST_ypos_{target}.npz")
        np.savez_compressed(
            npz_path,
            shap_values=shap_values,
            base_values=base_values,
            X_explain=X_explain,
            y_explain=y_explain,
            row_ids_explain=row_explain,
            feature_names=np.asarray(X_cols, dtype=object),
            test_te_idx=te_idx,
            explain_pick_local=pick_local
        )

        X_df = pd.DataFrame(X_explain, columns=X_cols)
        X_df.insert(0, "row_id_in_original_excel", row_explain)
        X_df.insert(1, "y_true", y_explain)

        shap_df = pd.DataFrame(shap_values, columns=[f"shap__{c}" for c in X_cols])
        wide = pd.concat([X_df, shap_df], axis=1)
        wide.to_csv(os.path.join(out_dir, f"shap_TEST_ypos_{target}_wide.csv"), index=False, encoding="utf-8-sig")

        meta = {
            "target": target,
            "policy": "TEST only, y>0 only",
            "y_zero_tol": float(Y_ZERO_TOL),
            "N_EXPLAIN_MAX": int(N_EXPLAIN_MAX),
            "N_BINS_POS": int(N_BINS_POS),
            "BACKGROUND_SIZE": int(bg_n),
            "MAX_EVALS": int(MAX_EVALS),
            "explainer": "PermutationExplainer",
            "pick_meta": meta_pick,
            "bundle_threshold": float(bundle.get("threshold", np.nan)),
            "zero_eps_train": float(bundle.get("zero_eps", np.nan)),
        }
        with open(os.path.join(out_dir, f"meta_TEST_ypos_{target}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if MAKE_PLOTS:
            X_explain_df = pd.DataFrame(X_explain, columns=X_cols)

            plt.figure()
            shap.summary_plot(shap_values, X_explain_df, show=False, max_display=MAX_DISPLAY)
            plt.title(f"SHAP Beeswarm (TEST, y>0) - {target}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"beeswarm_TEST_ypos_{target}.png"), dpi=DPI, format="png")
            plt.close()

            plot_bar_importance(
                shap_values, X_cols,
                out_png=os.path.join(out_dir, f"importance_bar_TEST_ypos_{target}.png"),
                title=f"SHAP Importance (mean|SHAP|, TEST y>0) - {target}"
            )

        print(f"  [OK] Saved: {npz_path}")

    print("\n=== DONE (FAST TEST y>0 SHAP) ===")
    print("SHAP outputs:", SHAP_DIR)

if __name__ == "__main__":
    main()
