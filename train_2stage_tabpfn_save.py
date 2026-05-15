import os
import warnings
warnings.filterwarnings("ignore")

import json
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
import optuna
from optuna.samplers import TPESampler

import joblib
import inspect

from tabpfn import TabPFNRegressor
try:
    from tabpfn import TabPFNClassifier
    HAS_TPFN_CLF = True
except Exception:
    HAS_TPFN_CLF = False

DATA_PATH = r"E:\edgebrowser\数据提取\TabPFN\实验水体.xlsx"
OUT_PATH  = r"E:\edgebrowser\数据提取\TabPFN\实验水体模型评估TabPFN.txt"

OUT_DIR = os.path.join(os.path.dirname(OUT_PATH), "tabpfn_2stage_saved1")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_TXT  = os.path.join(OUT_DIR, "summary_2stage_TabPFN.txt")
OUT_XLSX = os.path.join(OUT_DIR, "summary_2stage_TabPFN.xlsx")

RANDOM_STATE = 42
TRAIN_FRAC_EACH_GROUP = 0.8
ZERO_EPS = 0.0

N_TRIALS = 30
N_SPLITS = 5

N_EST_CHOICES = [2, 4, 8, 16, 32]
THR_MIN, THR_MAX = 0.30, 0.70

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

device_str = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device_str}")
if not HAS_TPFN_CLF:
    print("[WARN] 未成功导入 TabPFNClassifier：两段式中的“0/非0判别”将自动回退到 LogisticRegression（仍可跑通）。")

from sklearn.linear_model import LogisticRegression


def evaluate_regression(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae


def _build_kwargs(model_cls, n_estimators, device_str):
    sig = inspect.signature(model_cls.__init__)
    allowed = set(sig.parameters.keys())

    kwargs = {}
    if "device" in allowed:
        kwargs["device"] = device_str

    if "n_estimators" in allowed:
        kwargs["n_estimators"] = n_estimators
    elif "N_ensemble_configurations" in allowed:
        kwargs["N_ensemble_configurations"] = n_estimators
    elif "n_ensemble_configurations" in allowed:
        kwargs["n_ensemble_configurations"] = n_estimators

    if "seed" in allowed:
        kwargs["seed"] = RANDOM_STATE
    if "random_state" in allowed:
        kwargs["random_state"] = RANDOM_STATE

    return kwargs


def build_reg(n_estimators):
    kwargs = _build_kwargs(TabPFNRegressor, n_estimators, device_str)
    return TabPFNRegressor(**kwargs)


def build_clf(n_estimators):
    if HAS_TPFN_CLF:
        kwargs = _build_kwargs(TabPFNClassifier, n_estimators, device_str)
        return TabPFNClassifier(**kwargs)
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs"
    )


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


def stratified_split_zero_nonzero(y, train_frac=0.8, seed=42, eps=0.0):
    rng = np.random.RandomState(seed)
    is_zero = (np.abs(y) <= eps)

    idx0 = np.where(is_zero)[0]
    idx1 = np.where(~is_zero)[0]

    if len(idx0) == 0 or len(idx1) == 0:
        idx = np.arange(len(y))
        rng.shuffle(idx)
        n_tr = int(np.floor(train_frac * len(idx)))
        return idx[:n_tr], idx[n_tr:]

    rng.shuffle(idx0); rng.shuffle(idx1)
    n0_tr = int(np.floor(train_frac * len(idx0)))
    n1_tr = int(np.floor(train_frac * len(idx1)))

    tr = np.concatenate([idx0[:n0_tr], idx1[:n1_tr]])
    te = np.concatenate([idx0[n0_tr:], idx1[n1_tr:]])
    rng.shuffle(tr); rng.shuffle(te)
    return tr, te


def two_stage_fit_predict(
    X_train, y_train, X_pred,
    n_est_clf, n_est_reg, threshold,
    eps=0.0
):
    z_train = (np.abs(y_train) > eps).astype(int)

    clf = build_clf(n_est_clf)
    clf.fit(X_train, z_train)

    nz_mask = (z_train == 1)
    if nz_mask.sum() < 5:
        pred = np.zeros(X_pred.shape[0], dtype=float)
        return pred, clf, None

    reg = build_reg(n_est_reg)
    reg.fit(X_train[nz_mask], y_train[nz_mask])

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_pred)[:, 1]
        is_nz = (proba >= threshold)
    else:
        is_nz = (clf.predict(X_pred) == 1)

    pred = np.zeros(X_pred.shape[0], dtype=float)
    if np.any(is_nz):
        pred[is_nz] = reg.predict(X_pred[is_nz])
    return pred, clf, reg


def make_objective_cv(X_train, y_train, eps=0.0):
    z = (np.abs(y_train) > eps).astype(int)
    c0 = int((z == 0).sum())
    c1 = int((z == 1).sum())

    use_strat = (c0 >= 2 and c1 >= 2)
    if use_strat:
        n_splits_eff = min(N_SPLITS, min(c0, c1))
        n_splits_eff = max(n_splits_eff, 2)
        splitter = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=RANDOM_STATE)
        folds = [(tr, va) for tr, va in splitter.split(X_train, z)]
    else:
        n_splits_eff = min(N_SPLITS, max(2, len(y_train)//10))
        splitter = KFold(n_splits=n_splits_eff, shuffle=True, random_state=RANDOM_STATE)
        folds = [(tr, va) for tr, va in splitter.split(X_train)]

    def objective(trial: optuna.trial.Trial):
        n_est_reg = trial.suggest_categorical("n_estimators_reg", N_EST_CHOICES)
        threshold = trial.suggest_float("threshold", THR_MIN, THR_MAX)

        if HAS_TPFN_CLF:
            n_est_clf = trial.suggest_categorical("n_estimators_clf", N_EST_CHOICES)
        else:
            n_est_clf = 0

        r2_list = []
        for tr, va in folds:
            X_tr, y_tr = X_train[tr], y_train[tr]
            X_va, y_va = X_train[va], y_train[va]

            pred_va, _, _ = two_stage_fit_predict(
                X_tr, y_tr, X_va,
                n_est_clf=n_est_clf,
                n_est_reg=n_est_reg,
                threshold=threshold,
                eps=eps
            )
            r2, _, _ = evaluate_regression(y_va, pred_va)
            r2_list.append(r2)

        return float(np.mean(r2_list))

    return objective, use_strat, (c0, c1)


df = pd.read_excel(DATA_PATH)

X_cols = df.columns[1:15].tolist()
y_cols = df.columns[15:25].tolist()

all_results = []

for y_col in y_cols:
    print(f"\n========== Target: {y_col} ==========")

    X, y, row_ids = prepare_xy_for_one_target(df, X_cols, y_col)
    n = len(y)
    if n < 30:
        print("  [SKIP] 可用样本太少")
        continue

    tr_idx, te_idx = stratified_split_zero_nonzero(y, TRAIN_FRAC_EACH_GROUP, RANDOM_STATE, ZERO_EPS)
    X_train, y_train = X[tr_idx], y[tr_idx]
    X_test,  y_test  = X[te_idx], y[te_idx]

    X_train_df = pd.DataFrame(X_train, columns=X_cols)
    fill_values = X_train_df.median(numeric_only=True).to_dict()

    z_tr = (np.abs(y_train) > ZERO_EPS).astype(int)
    print(f"  N_total={n} | N_train={len(y_train)} | N_test={len(y_test)} | train(0/non0)={(z_tr==0).sum()}/{(z_tr==1).sum()}")

    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=f"{y_col}_2stage")
    objective, use_strat, (c0, c1) = make_objective_cv(X_train, y_train, ZERO_EPS)

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_params = study.best_params
    best_cv_r2 = float(study.best_value)

    n_est_reg = int(best_params.get("n_estimators_reg", 8))
    thr = float(best_params.get("threshold", 0.5))
    if HAS_TPFN_CLF:
        n_est_clf = int(best_params.get("n_estimators_clf", 8))
    else:
        n_est_clf = 0

    print(f"  Best CV R2={best_cv_r2:.4f} | best={best_params} | stratCV={use_strat} (train class 0/non0={c0}/{c1})")

    yhat_train, clf, reg = two_stage_fit_predict(
        X_train, y_train, X_train,
        n_est_clf=n_est_clf, n_est_reg=n_est_reg, threshold=thr, eps=ZERO_EPS
    )
    yhat_test, _, _ = two_stage_fit_predict(
        X_train, y_train, X_test,
        n_est_clf=n_est_clf, n_est_reg=n_est_reg, threshold=thr, eps=ZERO_EPS
    )
    r2_tr, rmse_tr, mae_tr = evaluate_regression(y_train, yhat_train)
    r2_te, rmse_te, mae_te = evaluate_regression(y_test,  yhat_test)
    print(f"  Train: R2={r2_tr:.4f} RMSE={rmse_tr:.4f} MAE={mae_tr:.4f}")
    print(f"  Test : R2={r2_te:.4f} RMSE={rmse_te:.4f} MAE={mae_te:.4f}")

    yhat_all, _, _ = two_stage_fit_predict(
        X_train, y_train, X,
        n_est_clf=n_est_clf, n_est_reg=n_est_reg, threshold=thr, eps=ZERO_EPS
    )

    model_path = os.path.join(OUT_DIR, f"model_{y_col}.joblib")
    bundle = {
        "target": y_col,
        "feature_cols": X_cols,
        "zero_eps": ZERO_EPS,
        "threshold": thr,
        "n_estimators_clf": n_est_clf,
        "n_estimators_reg": n_est_reg,
        "clf": clf,
        "reg": reg,
        "fill_values": fill_values,
        "device": device_str,
        "meta": {
            "random_state": RANDOM_STATE,
            "train_frac_each_group": TRAIN_FRAC_EACH_GROUP,
            "n_trials": N_TRIALS,
            "n_splits": N_SPLITS,
            "best_cv_r2": best_cv_r2,
            "train_metrics": {"R2": r2_tr, "RMSE": rmse_tr, "MAE": mae_tr},
            "test_metrics":  {"R2": r2_te, "RMSE": rmse_te, "MAE": mae_te},
            "n_total_used": int(len(y)),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        }
    }
    joblib.dump(bundle, model_path, compress=3)

    pred_npz_path = os.path.join(OUT_DIR, f"pred_{y_col}.npz")
    np.savez_compressed(
        pred_npz_path,
        row_ids=row_ids,
        tr_idx=tr_idx, te_idx=te_idx,
        y_all=y, yhat_all=yhat_all,
        y_train=y_train, yhat_train=yhat_train,
        y_test=y_test,   yhat_test=yhat_test
    )

    pred_xlsx_path = os.path.join(OUT_DIR, f"pred_{y_col}.xlsx")
    pred_df = pd.DataFrame({
        "row_id_in_original_excel": row_ids,
        "y_true": y,
        "y_pred": yhat_all
    })
    pred_df.to_excel(pred_xlsx_path, index=False)

    all_results.append({
        "Target": y_col,
        "Model": "TabPFN_TwoStage",
        "Best_CV_R2": best_cv_r2,
        "threshold": thr,
        "n_estimators_reg": n_est_reg,
        "n_estimators_clf": n_est_clf if HAS_TPFN_CLF else "LogReg_fallback",
        "R2_train": r2_tr, "RMSE_train": rmse_tr, "MAE_train": mae_tr,
        "R2_test": r2_te,  "RMSE_test": rmse_te,  "MAE_test": mae_te,
        "model_path": model_path,
        "pred_npz": pred_npz_path,
        "pred_xlsx": pred_xlsx_path,
    })

res_df = pd.DataFrame(all_results)
res_df.to_excel(OUT_XLSX, index=False)

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write(f"Saved dir: {OUT_DIR}\n")
    f.write(f"DATA_PATH: {DATA_PATH}\n\n")
    for _, row in res_df.iterrows():
        f.write(
            f"Target={row['Target']}\tCV_R2={row['Best_CV_R2']:.4f}\t"
            f"thr={row['threshold']:.3f}\treg_ens={row['n_estimators_reg']}\tclf={row['n_estimators_clf']}\t"
            f"Train_R2={row['R2_train']:.4f}\tTest_R2={row['R2_test']:.4f}\t"
            f"model={row['model_path']}\tpred={row['pred_npz']}\n"
        )

manifest_path = os.path.join(OUT_DIR, "manifest.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump({"out_dir": OUT_DIR, "summary_xlsx": OUT_XLSX, "summary_txt": OUT_TXT,
               "models": res_df.to_dict(orient="records")}, f, ensure_ascii=False, indent=2)

print("\n=== DONE ===")
print("Saved folder:", OUT_DIR)
print("Summary XLSX:", OUT_XLSX)
print("Summary TXT :", OUT_TXT)
print("Manifest   :", manifest_path)
