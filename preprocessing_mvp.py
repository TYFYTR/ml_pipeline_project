# preprocessing_mvp.py
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import KFold
import joblib


def find_csv(user_path: str | None) -> Path:
    if user_path:
        p = Path(user_path).expanduser().resolve()
        if p.exists():
            return p
        print(f"[error] CSV not found at: {p}", file=sys.stderr)
        sys.exit(1)
    here = Path(__file__).resolve().parent
    for name in ["Housing.csv", "Houseing.csv", "housing.csv", "houseing.csv"]:
        p = here / name
        if p.exists():
            return p
    print("[error] Could not find Housing.csv next to this script.", file=sys.stderr)
    sys.exit(1)


def detect_target(df: pd.DataFrame) -> str:
    for c in ["SalePrice", "saleprice", "Saleprice", "price", "Price"]:
        if c in df.columns:
            return c
    print("[error] Target not found. Expected 'SalePrice' or 'price'.", file=sys.stderr)
    sys.exit(1)


def make_onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >=1.2
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)         # older sklearn


def main():
    parser = argparse.ArgumentParser(description="Minimal preprocessing for Housing.csv")
    parser.add_argument("--csv", type=str, default=None, help="Path to Housing.csv (optional)")
    parser.add_argument("--outdir", type=str, default="artifacts", help="Output directory")
    args = parser.parse_args()

    csv_path = find_csv(args.csv)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load
    df = pd.read_csv(csv_path)
    print(f"Loaded: {csv_path.name}  shape={df.shape}")

    # 2) Inspect
    print("\nDtypes:")
    print(df.dtypes)
    missing_pct = df.isna().mean().sort_values(ascending=False) * 100.0
    print("\nMissingness (% of rows) by column:")
    print(missing_pct.to_string())

    # 3) Separate target
    target_col = detect_target(df)
    y = df[target_col].astype(float)
    X = df.drop(columns=[target_col])
    print(f"\nTarget detected: {target_col}")
    print(f"Feature matrix shape before transform: {X.shape}")

    # 4) Transform target
    y_log = np.log1p(y.values)
    print("Applied log1p to target.")

    # 5) Plan splits (not executed here)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    print("KFold ready: n_splits=5, shuffle=True, random_state=42")

    # Phase 2 — minimal cleaning
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    for c in X.columns:
        if c not in numeric_cols and c not in categorical_cols:
            categorical_cols.append(c)

    print("\nColumn groups:")
    print(f"  numeric ({len(numeric_cols)}): {numeric_cols}")
    print(f"  categorical ({len(categorical_cols)}): {categorical_cols}")

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    X_processed = preprocessor.fit_transform(X)
    print(f"\nTransformed feature matrix shape: {X_processed.shape}")

    # Save artifacts
    joblib.dump(preprocessor, outdir / "preprocessor.joblib")
    np.save(outdir / "X_processed.npy", X_processed)
    np.save(outdir / "y_log.npy", y_log)

    meta = {
        "csv": str(csv_path),
        "rows": int(df.shape[0]),
        "raw_feature_cols": X.columns.tolist(),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "target": target_col,
        "X_processed_shape": list(X_processed.shape),
        "y_log_shape": list(y_log.shape),
        "kfold": {"n_splits": 5, "shuffle": True, "random_state": 42},
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved artifacts to: {outdir}")
    print("Done.")


if __name__ == "__main__":
    main()
