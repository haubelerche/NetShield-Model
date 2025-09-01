#!/usr/bin/env python
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score, average_precision_score,precision_recall_curve)

def main(args):
    data_dir = Path(args.data)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(data_dir / "X_train.npy")
    y = np.load(data_dir / "y_train.npy")
    Xte = np.load(data_dir / "X_test.npy")
    yte = np.load(data_dir / "y_test.npy")
    feat_path = data_dir / "feature_names.json"
    if feat_path.exists():
        names = json.loads(feat_path.read_text(encoding="utf-8"))
        X   = pd.DataFrame(X, columns=names)
        Xte = pd.DataFrame(Xte, columns=names)

    #split validation từ train
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf = LGBMClassifier(objective="binary",n_estimators=5000,learning_rate=0.03,num_leaves=127,min_child_samples=20,subsample=0.9,
        subsample_freq=1,colsample_bytree=0.9,random_state=42,n_jobs=-1,class_weight="balanced",)

    clf.fit(Xtr, ytr,eval_set=[(Xval, yval)],eval_metric="auc",callbacks=[early_stopping(stopping_rounds=200, verbose=False), log_evaluation(0)],)

    # Lưu model
    dump(clf, out_dir / "lgbm_model.pkl")

    # ---- chọn threshold theo F1 trên validation ----
    val_proba = clf.predict_proba(Xval)[:, 1]
    prec, rec, th = precision_recall_curve(yval, val_proba)
    f1 = 2 * prec[1:] * rec[1:] / (prec[1:] + rec[1:] + 1e-12)
    best_idx = int(f1.argmax())
    best_th = float(th[best_idx])

    th_info = {
        "best_threshold": best_th,
        "val_best_f1": float(f1[best_idx]),
        "val_ap": float(average_precision_score(yval, val_proba)),
        "val_roc_auc": float(roc_auc_score(yval, val_proba)),
        "best_iteration": int(getattr(clf, "best_iteration_", getattr(clf, "best_iteration", 0)) or 0),
        "note": "Threshold chosen on validation to maximize F1 (no test leakage)."
    }
    (out_dir / "lgbm_threshold.json").write_text(json.dumps(th_info, indent=2), encoding="utf-8")

    # ---- đánh giá nhanh trên test ----
    te_proba = clf.predict_proba(Xte)[:, 1]
    te_auc = roc_auc_score(yte, te_proba)
    te_ap  = average_precision_score(yte, te_proba)
    te_pred = (te_proba >= best_th).astype(int)

    print(f"[TEST] ROC-AUC: {te_auc:.4f} | AP: {te_ap:.4f}")
    print("[TEST] Classification @best_th (from val):")
    print(classification_report(yte, te_pred, target_names=["benign", "attack"]))

    # Lưu số liệu gọn để README dùng
    (out_dir / "metrics_lgbm.json").write_text(
        json.dumps({"test_auc": float(te_auc), "test_ap": float(te_ap), "threshold": best_th}, indent=2),
        encoding="utf-8"
    )

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
