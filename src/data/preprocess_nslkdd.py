#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import OneHotEncoder, StandardScaler


attributes = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
    "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
    "num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"
]


cat_cols = ["protocol_type","service","flag"]
kdd_cols = attributes + ["label", "difficulty"]

def read_kdd(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path,names=kdd_cols, header=None,sep=",",encoding="utf-8",engine="python"  )
    # normalize
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    # drop difficulty
    if "difficulty" in df.columns:
        df = df.drop(columns=["difficulty"])
    # sanity: required categorical columns present
    missing = set(cat_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing categorical columns: {missing}")
    return df

def binarize_label(series: pd.Series) -> np.ndarray:
    # normal->0 (benign), others-> 1(attack)
    return (series != "normal").astype(int).values

def build_encoders():
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    scaler = StandardScaler()
    return ohe, scaler

def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame, cast_float32: bool = True):
    y_train = binarize_label(train_df["label"])
    y_test  = binarize_label(test_df["label"])

    X_train_raw = train_df.drop(columns=["label"]).copy()
    X_test_raw  = test_df.drop(columns=["label"]).copy()

    for c in cat_cols:
        X_train_raw[c] = X_train_raw[c].astype(str)
        X_test_raw[c]  = X_test_raw[c].astype(str)

    num_cols = [c for c in X_train_raw.columns if c not in cat_cols]

    X_train_num = X_train_raw[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_test_num  = X_test_raw[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    ohe, scaler = build_encoders()

    X_train_cat_ohe = ohe.fit_transform(X_train_raw[cat_cols])
    X_test_cat_ohe  = ohe.transform(X_test_raw[cat_cols])

    X_train_num_sc = scaler.fit_transform(X_train_num)
    X_test_num_sc  = scaler.transform(X_test_num)

    X_train = np.hstack([X_train_num_sc, X_train_cat_ohe])
    X_test  = np.hstack([X_test_num_sc, X_test_cat_ohe])

    if cast_float32:
        X_train = X_train.astype(np.float32, copy=False)
        X_test  = X_test.astype(np.float32, copy=False)

    feature_names = num_cols + list(ohe.get_feature_names_out(cat_cols))
    return X_train, y_train, X_test, y_test, ohe, scaler, feature_names

def save_outputs(out_dir: Path,
                 X_train, y_train, X_test, y_test,
                 ohe, scaler, feature_names):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_train.npy", X_train, allow_pickle=False)
    np.save(out_dir / "y_train.npy", y_train, allow_pickle=False)
    np.save(out_dir / "X_test.npy",  X_test,  allow_pickle=False)
    np.save(out_dir / "y_test.npy",  y_test,  allow_pickle=False)
    dump(ohe, out_dir / "ohe.pkl")
    dump(scaler, out_dir / "scaler.pkl")
    (out_dir / "feature_names.json").write_text(
        json.dumps(feature_names, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to KDDTrain+.txt")
    ap.add_argument("--test",  required=True, help="Path to KDDTest+.txt")
    ap.add_argument("--out",   required=True, help="Output directory for processed arrays & transformers")
    ap.add_argument("--no-float32", action="store_true", help="Disable casting arrays to float32")
    args = ap.parse_args()

    train_df = read_kdd(Path(args.train))
    test_df  = read_kdd(Path(args.test))

    missing_in_test = set(train_df.columns) - set(test_df.columns)
    if missing_in_test:
        raise ValueError(f"Test thiếu cột: {missing_in_test}")

    X_train, y_train, X_test, y_test, ohe, scaler, feature_names = preprocess(
        train_df, test_df, cast_float32=(not args.no_float32)
    )
    save_outputs(Path(args.out), X_train, y_train, X_test, y_test, ohe, scaler, feature_names)

    print(f"Saved to: {args.out}")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train attack ratio: {y_train.mean():.4f}, y_test attack ratio: {y_test.mean():.4f}")
    print(f"Feature dim: {X_train.shape[1]} (names saved to feature_names.json)")

if __name__ == "__main__":
    main()





