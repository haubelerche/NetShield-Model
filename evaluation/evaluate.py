#!/usr/bin/env python
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import (
    classification_report,
    roc_curve, precision_recall_curve,
    average_precision_score, roc_auc_score
)
from tensorflow import keras


# 1) Đánh giá LGBM
def eval_lgbm(data_dir: Path, art_dir: Path, rep_dir: Path):
    Xte = np.load(data_dir / "X_test.npy")
    yte = np.load(data_dir / "y_test.npy")

    feat_path = data_dir / "feature_names.json"
    X_infer = Xte
    if feat_path.exists():
        names = json.loads(feat_path.read_text(encoding="utf-8"))
        X_infer = pd.DataFrame(Xte, columns=names)

    # Tải model (ưu tiên lgbm_model.pkl, fallback lgbm.pkl)
    lgbm_path = art_dir / "lgbm_model.pkl"
    lgbm = load(lgbm_path)

    # Threshold
    best_th = 0.5
    th_path = art_dir / "lgbm_threshold.json"
    if th_path.exists():
        try:
            data = json.loads(th_path.read_text(encoding="utf-8"))
            best_th = float(data.get("best_threshold", 0.5))
        except Exception:
            pass

    proba = lgbm.predict_proba(X_infer)[:, 1]
    pred = (proba >= best_th).astype(int)

    # Report
    report_txt = classification_report(yte, pred, digits=4, target_names=["benign", "attack"])
    print(f"LightGBM classification report (threshold {best_th:.4f}):")
    print(report_txt)
    (rep_dir / "classification_lgbm.txt").write_text(report_txt, encoding="utf-8")

    # Curves & metrics
    fpr, tpr, _ = roc_curve(yte, proba)
    roc_auc = roc_auc_score(yte, proba)
    prec, rec, _ = precision_recall_curve(yte, proba)
    ap = average_precision_score(yte, proba)
    print(f"ROC-AUC: {roc_auc:.4f} | AP: {ap:.4f}")

    # Save plots -> reports/
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC — LightGBM"); plt.legend()
    plt.tight_layout(); plt.savefig(rep_dir / "roc_lgbm.png", dpi=150); plt.close()

    plt.figure()
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR — LightGBM"); plt.legend()
    plt.tight_layout(); plt.savefig(rep_dir / "pr_lgbm.png", dpi=150); plt.close()

    return proba, pred, best_th


# 2) Đánh giá AE (Keras)
def eval_autoencoder(data_dir: Path, art_dir: Path, rep_dir: Path):
    Xte = np.load(data_dir / "X_test.npy")
    yte = np.load(data_dir / "y_test.npy")

    ae_h5 = art_dir / "ae_model.h5"
    thr_pkl = art_dir / "ae_threshold.pkl"
    if not ae_h5.exists() or not thr_pkl.exists():
        print("[AE] Skipped: missing 'ae_model.h5' or 'ae_threshold.pkl'.")
        return None, None, None

    model = keras.models.load_model(str(ae_h5), compile=False)
    rec = model.predict(Xte, verbose=0)
    errs = ((rec - Xte) ** 2).mean(axis=1)

    thr = float(load(thr_pkl))
    y_pred_ae = (errs > thr).astype(int)

    report_txt = classification_report(yte, y_pred_ae, digits=4, target_names=["benign", "attack"])
    print("\nAutoencoder (threshold-based) report:")
    print(report_txt)
    (rep_dir / "classification_ae.txt").write_text(report_txt, encoding="utf-8")

    # ---- Plot: zoom quanh ngưỡng dựa trên benign để tránh bị “đè” về 0 ----
    benign_errs = errs[yte == 0]
    attack_errs = errs[yte == 1]
    # upper = max(p99.5 của benign, 3*threshold) để nhìn rõ vùng quyết định
    upper = float(max(np.percentile(benign_errs, 99.5), thr * 3.0))
    lower = 0.0

    plt.figure()
    bins = np.linspace(lower, upper, 120)
    plt.hist(np.clip(benign_errs, lower, upper), bins=bins, alpha=0.6, density=True, label="benign")
    plt.hist(np.clip(attack_errs, lower, upper), bins=bins, alpha=0.6, density=True, label="attack")
    plt.axvline(thr, color="k", linestyle="--", linewidth=1.5, label=f"threshold={thr:.4f}")
    plt.xlabel("Reconstruction MSE"); plt.ylabel("Density")
    plt.title("AE Reconstruction Error (zoomed near threshold)")
    plt.legend()
    plt.tight_layout(); plt.savefig(rep_dir / "ae_error_dist.png", dpi=150); plt.close()

    return errs, y_pred_ae, thr


# 3) Fusion OR (tùy chọn)
def eval_fusion(yte, lgbm_pred, ae_pred, rep_dir: Path):
    if lgbm_pred is None or ae_pred is None:
        return
    y_pred_or = ((lgbm_pred == 1) | (ae_pred == 1)).astype(int)
    rep = classification_report(yte, y_pred_or, digits=4, target_names=["benign", "attack"])
    print("\nFusion OR (LGBM OR AE) report:")
    print(rep)
    (rep_dir / "classification_fusion_or.txt").write_text(rep, encoding="utf-8")


def main(args):
    data_dir = Path(args.data)
    art_dir = Path(args.artifacts)
    rep_dir = Path(args.report)
    rep_dir.mkdir(parents=True, exist_ok=True)

    lgbm_proba, lgbm_pred, _ = eval_lgbm(data_dir, art_dir, rep_dir)
    errs, ae_pred, _ = eval_autoencoder(data_dir, art_dir, rep_dir)

    yte = np.load(data_dir / "y_test.npy")
    eval_fusion(yte, lgbm_pred, ae_pred, rep_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--report", default="reports")
    main(p.parse_args())
