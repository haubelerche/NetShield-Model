#!/usr/bin/env python
import argparse
from pathlib import Path
import numpy as np
from tensorflow import keras
from joblib import dump

def build_ae(input_dim: int) -> keras.Model:
    inp = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(256, activation="relu")(inp)
    x = keras.layers.Dense(64, activation="relu")(x)
    z = keras.layers.Dense(16, activation="relu")(x)
    x = keras.layers.Dense(64, activation="relu")(z)
    x = keras.layers.Dense(256, activation="relu")(x)
    out = keras.layers.Dense(input_dim, activation="linear")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model

def main(args):
    data_dir = Path(args.data)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")


#train ae tren normal pattern
    Xb = X_train[y_train == 0]
    model = build_ae(X_train.shape[1])
    model.fit(Xb, Xb, epochs=20, batch_size=256, shuffle=True, verbose=1)
    model.save(out_dir / "ae_model.h5")

#tính threshold
    rec_train = model.predict(Xb, verbose=0)
    errs = ((rec_train - Xb) ** 2).mean(axis=1)
    threshold = float(np.percentile(errs, 99.5))
    dump(threshold, out_dir / "ae_threshold.pkl")

    print("Saved:", out_dir / "ae_model.h5", "and", out_dir / "ae_threshold.pkl")
    print("AE threshold is:", threshold)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
