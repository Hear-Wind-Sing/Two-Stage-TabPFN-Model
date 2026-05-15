# -*- coding: utf-8 -*-
import os
import glob
import json
import re
import warnings
warnings.filterwarnings("ignore")

os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

DATA_PATH = r"E:\edgebrowser\数据提取\TabPFN\实验水体.xlsx"
OUT_DIR   = r"E:\edgebrowser\数据提取\TabPFN\tabpfn_2stage_saved1"

TARGETS = None

STRICT_REQUIRE_SPLIT = True

SHEET_NAME = 0

ANA_DIR = os.path.join(OUT_DIR, "analysis_reports_AD_centroid_p95_mean2sd")
os.makedirs(ANA_DIR, exist_ok=True)

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["mathtext.fontset"] = "stix"

DPI = 800

COLOR_P95 = "#F68B33"
COLOR_MU2SD = "#5C9BCB"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def safe_filename(s: str, max_len: int = 160) -> str:
    s = str(s)
    s = re.sub(r'[\\/:*?"<>|\r\n]+', "_", s)
    s = s.strip(" .")
    if len(s) > max_len:
        s = s[:max_len]
    return s

def savefig(path: str, dpi: int = DPI):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")

def prepare_xy_like_training(df, feature_cols, y_col):
    X_df = df[feature_cols].copy()
    y_s  = df[y_col].copy()

    data = pd.concat([X_df, y_s.rename("y")], axis=1)
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(axis=0)

    row_ids = data.index.values
    X = data[feature_cols].values.astype(float)
    y = data["y"].values.astype(float)
    return X, y, row_ids, data

def load_split_indices(out_dir, target, n_samples):
    pred_npz = os.path.join(out_dir, f"pred_{target}.npz")
    if os.path.exists(pred_npz):
        npz = np.load(pred_npz, allow_pickle=True)
        tr_idx = npz["tr_idx"].astype(int)
        te_idx = npz["te_idx"].astype(int)
        return tr_idx, te_idx, pred_npz

    if STRICT_REQUIRE_SPLIT:
        raise FileNotFoundError(
            f"未找到划分文件：{pred_npz}\n"
            f"建议使用训练阶段保存的 pred_{target}.npz，以保证 AD 分析与测试集划分一致。"
        )
    else:
        print(f"[WARN] 未找到 {pred_npz}，将全体样本同时作为 train/test（不建议论文正式使用）")
        all_idx = np.arange(n_samples, dtype=int)
        return all_idx, all_idx, None

def calc_centroid_ad(X_train, X_test):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xte_s = scaler.transform(X_test)

    centroid = Xtr_s.mean(axis=0)

    d_train = np.linalg.norm(Xtr_s - centroid, axis=1)
    d_test  = np.linalg.norm(Xte_s - centroid, axis=1)

    mu = float(np.mean(d_train))
    sd = float(np.std(d_train, ddof=0))

    thr_p95 = float(np.quantile(d_train, 0.95))
    thr_mu2sd = float(mu + 2.0 * sd)

    in_ad_p95 = (d_test <= thr_p95).astype(int)
    in_ad_mu2sd = (d_test <= thr_mu2sd).astype(int)

    return {
        "scaler": scaler,
        "centroid": centroid,
        "d_train": d_train,
        "d_test": d_test,
        "mean_d_train": mu,
        "sd_d_train": sd,
        "thr_p95": thr_p95,
        "thr_mu2sd": thr_mu2sd,
        "in_ad_p95": in_ad_p95,
        "in_ad_mu2sd": in_ad_mu2sd,
    }

def main():
    if DATA_PATH.lower().endswith(".csv"):
        df_raw = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    else:
        df_raw = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

    model_paths = sorted(glob.glob(os.path.join(OUT_DIR, "model_*.joblib")))
    if TARGETS is not None:
        picked = []
        for mp in model_paths:
            tgt = os.path.basename(mp).replace("model_", "").replace(".joblib", "")
            if tgt in TARGETS:
                picked.append(mp)
        model_paths = picked

    if not model_paths:
        raise RuntimeError(f"未找到模型文件：{os.path.join(OUT_DIR, 'model_*.joblib')}")

    print(f"[INFO] 共发现 {len(model_paths)} 个 target 模型")

    all_summary = []
    all_chart_rows = []

    for model_path in model_paths:
        bundle = joblib.load(model_path)
        target = bundle["target"]
        feature_cols = bundle["feature_cols"]

        print(f"\n========== Target: {target} ==========")

        tdir = os.path.join(ANA_DIR, safe_filename(target))
        os.makedirs(tdir, exist_ok=True)

        X, y, row_ids, cleaned_data = prepare_xy_like_training(df_raw, feature_cols, target)

        tr_idx, te_idx, pred_npz_path = load_split_indices(OUT_DIR, target, len(y))

        X_train, y_train = X[tr_idx], y[tr_idx]
        X_test,  y_test  = X[te_idx], y[te_idx]
        row_train = row_ids[tr_idx]
        row_test  = row_ids[te_idx]

        ad_res = calc_centroid_ad(X_train, X_test)

        d_train = ad_res["d_train"]
        d_test = ad_res["d_test"]
        mean_d_train = ad_res["mean_d_train"]
        sd_d_train = ad_res["sd_d_train"]
        thr_p95 = ad_res["thr_p95"]
        thr_mu2sd = ad_res["thr_mu2sd"]
        in_ad_p95 = ad_res["in_ad_p95"]
        in_ad_mu2sd = ad_res["in_ad_mu2sd"]

        in_rate_p95 = float(np.mean(in_ad_p95))
        in_rate_mu2sd = float(np.mean(in_ad_mu2sd))

        print(f"[AD] train mean distance     = {mean_d_train:.6f}")
        print(f"[AD] train std distance      = {sd_d_train:.6f}")
        print(f"[AD] 95th percentile         = {thr_p95:.6f}")
        print(f"[AD] mean + 2SD              = {thr_mu2sd:.6f}")
        print(f"[AD] test in-AD rate (p95)   = {in_rate_p95*100:.2f}%")
        print(f"[AD] test in-AD rate (mu+2sd)= {in_rate_mu2sd*100:.2f}%")

        test_ad_df = pd.DataFrame({
            "row_id_in_cleaned_data": te_idx,
            "row_id_in_original_file_0_based": row_test,
            "excel_row_number_1_based_header_included": row_test + 2,
            "y_true": y_test,
            "distance_to_centroid": d_test,
            "threshold_95th_percentile": thr_p95,
            "threshold_mean_plus_2sd": thr_mu2sd,
            "in_AD_95th_percentile": in_ad_p95,
            "in_AD_mean_plus_2sd": in_ad_mu2sd,
        })

        test_feature_df = pd.DataFrame(X_test, columns=feature_cols)
        test_ad_detail_df = pd.concat([test_ad_df, test_feature_df], axis=1)

        train_ref_df = pd.DataFrame({
            "row_id_in_cleaned_data": tr_idx,
            "row_id_in_original_file_0_based": row_train,
            "excel_row_number_1_based_header_included": row_train + 2,
            "y_train": y_train,
            "distance_to_centroid": d_train,
        })

        chart_df = pd.DataFrame({
            "target": [target, target],
            "threshold_label": ["95th percentile", "mean+2SD"],
            "in_AD_rate": [in_rate_p95, in_rate_mu2sd],
            "in_AD_percent": [in_rate_p95 * 100.0, in_rate_mu2sd * 100.0],
        })

        summary_payload = {
            "target": target,
            "n_total_after_clean": int(len(y)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_features": int(len(feature_cols)),
            "split_npz_path": pred_npz_path if pred_npz_path is not None else "NOT_FOUND",

            "train_distance_mean": float(np.mean(d_train)),
            "train_distance_std": float(np.std(d_train, ddof=0)),
            "train_distance_median": float(np.median(d_train)),
            "train_distance_min": float(np.min(d_train)),
            "train_distance_max": float(np.max(d_train)),

            "AD_threshold_95th_percentile": float(thr_p95),
            "AD_threshold_mean_plus_2sd": float(thr_mu2sd),

            "test_in_AD_n_95th_percentile": int(np.sum(in_ad_p95)),
            "test_out_AD_n_95th_percentile": int(np.sum(1 - in_ad_p95)),
            "test_in_AD_rate_95th_percentile": float(in_rate_p95),
            "test_in_AD_percent_95th_percentile": float(in_rate_p95 * 100.0),

            "test_in_AD_n_mean_plus_2sd": int(np.sum(in_ad_mu2sd)),
            "test_out_AD_n_mean_plus_2sd": int(np.sum(1 - in_ad_mu2sd)),
            "test_in_AD_rate_mean_plus_2sd": float(in_rate_mu2sd),
            "test_in_AD_percent_mean_plus_2sd": float(in_rate_mu2sd * 100.0),
        }
        summary_df = pd.DataFrame([summary_payload])

        plt.figure(figsize=(7.2, 5.4))
        plt.hist(d_train, bins=30, alpha=0.75, label="Train", edgecolor="black")
        plt.hist(d_test, bins=30, alpha=0.55, label="Test", edgecolor="black")
        plt.axvline(thr_p95, color=COLOR_P95, linestyle="--", linewidth=2, label="95th percentile")
        plt.axvline(thr_mu2sd, color=COLOR_MU2SD, linestyle="--", linewidth=2, label="mean+2SD")
        plt.xlabel("Euclidean distance to training centroid")
        plt.ylabel("Count")
        plt.title(f"{target}")
        plt.legend(frameon=False)
        savefig(os.path.join(tdir, f"{safe_filename(target)}_AD_distance_distribution.png"))

        plt.figure(figsize=(8.0, 5.2))
        x_order = np.arange(1, len(d_test) + 1)
        plt.scatter(x_order, d_test, s=18)
        plt.axhline(thr_p95, color=COLOR_P95, linestyle="--", linewidth=2, label="95th percentile")
        plt.axhline(thr_mu2sd, color=COLOR_MU2SD, linestyle="--", linewidth=2, label="mean+2SD")
        plt.xlabel("Test compound index")
        plt.ylabel("Distance to training centroid")
        plt.title(f"{target}")
        plt.legend(frameon=False)
        savefig(os.path.join(tdir, f"{safe_filename(target)}_AD_test_distance_scatter.png"))

        xlsx_out = os.path.join(tdir, f"{safe_filename(target)}_AD_analysis.xlsx")
        with pd.ExcelWriter(xlsx_out, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            chart_df.to_excel(writer, sheet_name="chart_data", index=False)
            test_ad_detail_df.to_excel(writer, sheet_name="test_AD_detail", index=False)
            train_ref_df.to_excel(writer, sheet_name="train_distance_ref", index=False)
            pd.DataFrame({"feature_cols": feature_cols}).to_excel(
                writer, sheet_name="feature_list", index=False
            )

        json_out = os.path.join(tdir, f"{safe_filename(target)}_AD_summary.json")
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, ensure_ascii=False, indent=2)

        all_summary.append(summary_payload)
        all_chart_rows.append(chart_df)

        print(f"[OK] {target} -> 输出：{tdir}")

    all_summary_df = pd.DataFrame(all_summary)

    if len(all_chart_rows) > 0:
        all_chart_df = pd.concat(all_chart_rows, axis=0, ignore_index=True)
    else:
        all_chart_df = pd.DataFrame(columns=["target", "threshold_label", "in_AD_rate", "in_AD_percent"])

    all_summary_xlsx = os.path.join(ANA_DIR, "ALL_targets_AD_summary.xlsx")
    with pd.ExcelWriter(all_summary_xlsx, engine="openpyxl") as writer:
        all_summary_df.to_excel(writer, sheet_name="summary", index=False)
        all_chart_df.to_excel(writer, sheet_name="chart_data", index=False)

    all_summary_df.to_csv(
        os.path.join(ANA_DIR, "ALL_targets_AD_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    all_chart_df.to_csv(
        os.path.join(ANA_DIR, "ALL_targets_AD_chart_data.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    if len(all_summary_df) > 0:
        plot_df = all_summary_df.copy()
        plot_df = plot_df.sort_values("target").reset_index(drop=True)

        targets = plot_df["target"].tolist()
        y_p95 = plot_df["test_in_AD_percent_95th_percentile"].values.astype(float)
        y_mu2sd = plot_df["test_in_AD_percent_mean_plus_2sd"].values.astype(float)

        x = np.arange(len(targets))
        width = 0.30

        fig_width = max(7.5, len(targets) * 1.15)
        plt.figure(figsize=(fig_width, 5.5))

        plt.bar(
            x - width / 2,
            y_p95,
            width=width,
            color=COLOR_P95,
            edgecolor="black",
            linewidth=0.35,
            label=r"95$^{\mathrm{th}}$ percentile"
        )
        plt.bar(
            x + width / 2,
            y_mu2sd,
            width=width,
            color=COLOR_MU2SD,
            edgecolor="black",
            linewidth=0.35,
            label="mean+2SD"
        )

        plt.ylabel("Compounds in the AD (%)", fontsize=17)
        plt.xticks(x, targets, rotation=90, fontsize=11)
        plt.yticks(fontsize=12)

        ymax = float(np.nanmax([np.nanmax(y_p95), np.nanmax(y_mu2sd)]))
        plt.ylim(0, max(105, ymax + 3))

        plt.legend(title="Threshold:", frameon=False, fontsize=12, title_fontsize=13)

        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)

        savefig(os.path.join(ANA_DIR, "ALL_targets_AD_barplot.png"))

    print("\n=== DONE ===")
    print("Output folder:", ANA_DIR)

if __name__ == "__main__":
    main()