#!/usr/bin/env python3
"""
PLS-only screening: supervised feature ranking with GroupKFold CV and VIP.

Usage examples
--------------
# 1) Using a single merged CSV (recommended)
python pls_runner.py --wide_csv ./data/data_wide.csv --out_dir ./reports_pls --max_components 10 --top_k 25

# 2) Using a YAML config that lists per-table CSVs (optional; see below)
python pls_runner.py --yaml ./configs/pls_data.yaml --out_dir ./reports_pls

Notes
-----
- Expects targets named 'estrogen' and 'lh'.
- Uses GroupKFold by 'id' to select number of PLS components.
- Outputs VIP ranking, component scores, basic visuals.
"""
import os
import json
import argparse
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# ---------------------------
# I/O helpers
# ---------------------------
def load_wide_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Wide CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"id", "day_in_study", "estrogen", "lh"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Wide CSV must contain columns: {missing}")
    return df

def load_from_yaml(yaml_path: str) -> pd.DataFrame:
    """
    Minimal multi-file loader:
    YAML structure:
      hormones: ./data/hormones_and_selfreport.csv
      files:
        - path: ./data/wrist_temperature.csv
          groupby_mean: [temperature_diff_from_baseline]
        - path: ./data/resting_heart_rate.csv
          rename: {value: rhr}
        - path: ./data/heart_rate_variability_details.csv
          groupby_median: [rmssd]
        - path: ./data/glucose.csv
          cgm: true
      id_col: id
      day_col: day_in_study
    """
    import yaml
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    id_col = cfg.get("id_col", "id")
    day_col = cfg.get("day_col", "day_in_study")
    # targets
    h_path = cfg.get("hormones")
    if not h_path or not os.path.exists(h_path):
        raise FileNotFoundError("YAML must set 'hormones' pointing to hormones_and_selfreport.csv")
    H = pd.read_csv(h_path)
    if not {"estrogen", "lh", id_col, day_col}.issubset(H.columns):
        raise ValueError("Hormones file must include columns: id, day_in_study, estrogen, lh")
    H = H[[id_col, day_col, "estrogen", "lh"]]

    chunks = []
    for spec in cfg.get("files", []):
        p = spec["path"]
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        # groupby means
        if "groupby_mean" in spec:
            cols = [c for c in spec["groupby_mean"] if c in df.columns]
            if cols:
                g = df.groupby([id_col, day_col], as_index=False)[cols].mean()
                chunks.append(g)
        # groupby medians
        if "groupby_median" in spec:
            cols = [c for c in spec["groupby_median"] if c in df.columns]
            if cols:
                g = df.groupby([id_col, day_col], as_index=False)[cols].median()
                chunks.append(g)
        # simple rename passthrough (already day-level)
        if "rename" in spec:
            rename_map = {k: v for k, v in spec["rename"].items() if k in df.columns}
            if rename_map:
                sub = df[[id_col, day_col] + list(rename_map.keys())].rename(columns=rename_map)
                chunks.append(sub)
        # CGM convenience
        if spec.get("cgm", False):
            if "glucose_value" in df.columns:
                g = df[[id_col, day_col, "glucose_value"]].groupby([id_col, day_col]).agg(
                    cgm_median=("glucose_value", "median"),
                    cgm_mean=("glucose_value", "mean"),
                    cgm_sd=("glucose_value", "std"),
                ).reset_index()
                g["cgm_cv"] = g["cgm_sd"] / g["cgm_mean"]
                chunks.append(g)

    # merge all sources
    if chunks:
        X = chunks[0]
        for c in chunks[1:]:
            X = X.merge(c, on=[id_col, day_col], how="outer")
    else:
        X = pd.DataFrame(columns=[id_col, day_col])

    XY = H.merge(X, on=[id_col, day_col], how="left")
    return XY

# ---------------------------
# Core PLS utilities
# ---------------------------
def choose_n_components(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, max_comp: int = 10) -> Tuple[int, Dict[int, float]]:
    """
    GroupKFold CV to pick #components by minimizing RMSE over both targets jointly.
    """
    n_samples, n_features = X.shape
    max_k = int(min(max_comp, n_samples - 1, n_features))
    if max_k < 1:
        raise ValueError("Not enough samples/features for PLS.")
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    results: Dict[int, float] = {}
    for k in range(1, max_k + 1):
        rmses = []
        for tr, te in cv.split(X, Y, groups):
            pls = PLSRegression(n_components=k, scale=False)
            pls.fit(X[tr], Y[tr])
            Yp = pls.predict(X[te])
            rmse = float(np.sqrt(mean_squared_error(Y[te], Yp)))
            rmses.append(rmse)
        results[k] = float(np.mean(rmses))
    best_k = int(min(results, key=results.get))
    return best_k, results

def fit_pls(X: np.ndarray, Y: np.ndarray, n_components: int) -> PLSRegression:
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X, Y)
    return pls

def vip_scores(pls: PLSRegression, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Multi-response VIP (aggregate per-component contribution to Y).
    VIP_j = sqrt( p * sum_a (w_{j,a}^2 * SSY_a) / sum_a SSY_a )
    where SSY_a ~ sum of squared contribution of component a to Y.
    """
    T = pls.x_scores_
    W = pls.x_weights_
    Q = pls.y_loadings_
    p = X.shape[1]

    # explained SS of Y for each component
    SSY = []
    for a in range(T.shape[1]):
        ta = T[:, [a]]
        qa = Q[[a], :]
        Yhat_a = ta @ qa
        SSY.append(np.sum(Yhat_a ** 2))
    SSY = np.array(SSY)
    denom = SSY.sum() if SSY.sum() > 0 else 1.0

    vip = np.zeros(p)
    for j in range(p):
        w2 = W[j, :] ** 2
        vip[j] = np.sqrt(p * np.sum(w2 * SSY) / denom)
    return vip

def explained_variance(pls: PLSRegression, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cumulative R^2 for X and Y across components.
    """
    T = pls.x_scores_
    P = pls.x_loadings_
    r2x, r2y = [], []
    for k in range(1, T.shape[1] + 1):
        # X reconstruction
        Xhat = T[:, :k] @ P[:, :k].T
        sse_x = np.sum((X - Xhat) ** 2)
        sst_x = np.sum((X - X.mean(axis=0)) ** 2)
        r2x.append(1 - sse_x / sst_x if sst_x > 0 else 0)

        # Y prediction
        pls_k = PLSRegression(n_components=k, scale=False)
        pls_k.fit(X, Y)
        Yhat = pls_k.predict(X)
        sse_y = np.sum((Y - Yhat) ** 2)
        sst_y = np.sum((Y - Y.mean(axis=0)) ** 2)
        r2y.append(1 - sse_y / sst_y if sst_y > 0 else 0)
    return np.array(r2x), np.array(r2y)

# ---------------------------
# Plot helpers
# ---------------------------
def save_vip_bar(vip_series: pd.Series, out_png: str, top_k: int = 25):
    top = vip_series.head(top_k)[::-1]
    plt.figure(figsize=(8, max(4, 0.32 * len(top))))
    sns.barplot(x=top.values, y=top.index, color="#4e79a7")
    plt.xlabel("VIP score"); plt.ylabel("Feature"); plt.title(f"Top {len(top)} VIP features")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def save_score_scatter(scores: np.ndarray, y_cont: np.ndarray, label: str, out_png: str):
    if scores.shape[1] < 2:
        return
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(scores[:, 0], scores[:, 1], c=y_cont, cmap="viridis", s=12, alpha=0.8)
    plt.colorbar(sc, label=label)
    plt.axhline(0, color="grey", lw=0.6); plt.axvline(0, color="grey", lw=0.6)
    plt.xlabel("t1"); plt.ylabel("t2"); plt.title(f"PLS scores (colored by {label})")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def save_biplot(scores: np.ndarray, loadings: np.ndarray, feature_names: List[str], out_png: str, top: int = 15):
    if scores.shape[1] < 2 or loadings.shape[1] < 2:
        return
    norms = np.linalg.norm(loadings[:, :2], axis=1)
    idx = np.argsort(norms)[-top:]
    plt.figure(figsize=(6, 6))
    plt.scatter(scores[:, 0], scores[:, 1], s=10, alpha=0.3, color="#999999")
    scale_x, scale_y = np.std(scores[:, 0]), np.std(scores[:, 1])
    for i in idx:
        x, y = loadings[i, 0] * scale_x, loadings[i, 1] * scale_y
        plt.arrow(0, 0, x, y, color="#d62728", head_width=0.02 * max(np.abs(scores[:, :2])), length_includes_head=True)
        plt.text(x * 1.05, y * 1.05, feature_names[i], fontsize=8, color="#d62728")
    plt.axhline(0, color="grey", lw=0.6); plt.axvline(0, color="grey", lw=0.6)
    plt.xlabel("t1"); plt.ylabel("t2"); plt.title("PLS biplot (top loadings)")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def save_corr_heatmap(df_top: pd.DataFrame, out_png: str):
    plt.figure(figsize=(max(6, 0.45 * df_top.shape[1]), 5))
    sns.heatmap(df_top.corr(method="spearman"), cmap="coolwarm", center=0, annot=False)
    plt.title("Spearman correlations (top features & targets)")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def save_cum_variance_plot(r2x: np.ndarray, r2y: np.ndarray, out_png: str):
    k = np.arange(1, len(r2x) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(k, r2x, "-o", label="Cumulative R²X")
    plt.plot(k, r2y, "-o", label="Cumulative R²Y")
    plt.xlabel("n_components"); plt.ylabel("explained variance (R²)")
    plt.title("PLS cumulative explained variance")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide_csv", type=str, help="Path to merged wide CSV with id, day_in_study, estrogen, lh + features")
    ap.add_argument("--yaml", type=str, help="Optional YAML describing multiple CSVs to join (see docstring).")
    ap.add_argument("--out_dir", type=str, default="./reports_pls")
    ap.add_argument("--max_components", type=int, default=10)
    ap.add_argument("--top_k", type=int, default=25)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fig_dir = os.path.join(args.out_dir, "figs"); os.makedirs(fig_dir, exist_ok=True)

    # Load data
    if args.wide_csv:
        df = load_wide_csv(args.wide_csv)
    elif args.yaml:
        df = load_from_yaml(args.yaml)
    else:
        raise ValueError("Provide either --wide_csv or --yaml")

    # Keep rows with both targets
    df = df.dropna(subset=["estrogen", "lh"]).copy()

    # Split targets and features
    id_col = "id"; day_col = "day_in_study"
    Y = df[["estrogen", "lh"]].values
    feature_cols = [c for c in df.columns if c not in [id_col, day_col, "estrogen", "lh"]]
    X_df = df[feature_cols]

    # Drop columns with >50% missingness; median-impute remaining; standardize
    keep_cols = [c for c in X_df.columns if X_df[c].notna().mean() >= 0.5]
    X_df = X_df[keep_cols]
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    X = X_df.values.astype(float)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    Xs = x_scaler.fit_transform(X)
    Ys = y_scaler.fit_transform(Y)

    # Groups for CV (by person)
    groups = df[id_col].values

    # Pick number of components by GroupKFold CV
    best_k, cv_rmse = choose_n_components(Xs, Ys, groups, max_comp=args.max_components)

    # Fit final PLS and compute VIP
    pls = fit_pls(Xs, Ys, best_k)
    vip = vip_scores(pls, Xs, Ys)
    vip_s = pd.Series(vip, index=X_df.columns).sort_values(ascending=False)
    vip_s.to_csv(os.path.join(args.out_dir, "pls_vip_scores.csv"))

    # Scores & explained variance
    scores = pls.x_scores_
    r2x, r2y = explained_variance(pls, Xs, Ys)

    # Save component scores with id/day
    scores_df = pd.DataFrame(scores, columns=[f"t{i+1}" for i in range(scores.shape[1])])
    scores_df.insert(0, day_col, df[day_col].values)
    scores_df.insert(0, id_col, df[id_col].values)
    scores_df.to_csv(os.path.join(args.out_dir, "pls_component_scores.csv"), index=False)

    # Visuals
    save_vip_bar(vip_s, os.path.join(fig_dir, "vip_topK_bar.png"), top_k=args.top_k)
    save_score_scatter(scores, Ys[:, 0], "estrogen (scaled)", os.path.join(fig_dir, "scores_t1_t2_estrogen.png"))
    save_score_scatter(scores, Ys[:, 1], "lh (scaled)", os.path.join(fig_dir, "scores_t1_t2_lh.png"))
    save_biplot(scores, pls.x_loadings_, X_df.columns.tolist(), os.path.join(fig_dir, "biplot_t1_t2.png"), top=min(args.top_k, 20))

    # Heatmap of top‑K VIP vs targets
    top_feats = vip_s.head(args.top_k).index.tolist()
    heat_df = pd.concat([pd.DataFrame(Y, columns=["estrogen", "lh"]), X_df[top_feats].reset_index(drop=True)], axis=1)
    save_corr_heatmap(heat_df, os.path.join(fig_dir, "heatmap_topK_vs_targets.png"))

    # Explained variance plot & summary
    save_cum_variance_plot(r2x, r2y, os.path.join(fig_dir, "cum_explained_variance.png"))
    summary = {
        "best_n_components": int(best_k),
        "cv_rmse_by_k": {str(k): float(v) for k, v in cv_rmse.items()},
        "cum_R2X": [float(x) for x in r2x],
        "cum_R2Y": [float(x) for x in r2y],
        "n_samples": int(Xs.shape[0]),
        "n_features_after_filter": int(Xs.shape[1])
    }
    with open(os.path.join(args.out_dir, "pls_cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[PLS] Done. Components={best_k}. Reports in {args.out_dir}")

if __name__ == "__main__":
    main()
