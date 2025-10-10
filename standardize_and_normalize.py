#!/usr/bin/env python3
"""
Standardize PPP & POSS_PCT features + SEASON, set WIN_PCT as label,
and run season-out cross-validated baselines.

Outputs:
  - standardized_all_data.csv   (one-hot SEASON + standardized numeric features; includes WIN_PCT)
  - season_out_cv_report.txt    (metrics by season-out fold + overall)
  - artifacts/
      preprocessor.joblib
      elasticnet_cv.joblib
      (optional) xgb.joblib

Usage:
  python standardize_and_model.py --csv features_with_ids_with_wins.csv
"""

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone
from sklearn.utils import check_random_state
import numpy as np
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

import joblib

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Basic sanity
    needed = {"SEASON","TEAM_ID","TEAM_NAME","WIN_PCT"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    # Keep only one row per team-season (should already be). Drop duplicates defensively.
    df = df.drop_duplicates(subset=["SEASON","TEAM_ID"]).reset_index(drop=True)
    return df

def select_feature_columns(df: pd.DataFrame):
    num_cols = sorted([c for c in df.columns if c.startswith("PPP_") or c.startswith("POSS_PCT_")])
    cat_cols = ["SEASON"]
    y_col = "WIN_PCT"
    if not num_cols:
        raise ValueError("No feature columns found with prefixes 'PPP_' or 'POSS_PCT_'")
    if y_col not in df.columns:
        raise ValueError("WIN_PCT not found—run load_wins() first to add wins/win_pct columns.")
    return num_cols, cat_cols, y_col

def build_preprocessor(numeric_cols, categorical_cols):
    numeric = Pipeline(steps=[("scaler", StandardScaler(with_mean=True, with_std=True))])
    categorical = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_cols),
            ("cat", categorical, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0
    )
    return pre

def get_feature_names(preprocessor: ColumnTransformer) -> list:
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            # numeric cols pass through scaler → same names
            if isinstance(cols, (list, tuple, np.ndarray)):
                names.extend(list(cols))
            else:
                names.append(cols)
        elif name == "cat":
            # onehot
            ohe = trans.named_steps["onehot"]
            cats = ohe.categories_
            colnames = []
            # cols is ["SEASON"]
            for base, categories in zip(cols, cats):
                colnames.extend([f"{base}__{c}" for c in categories])
            names.extend(colnames)
    return names

def export_standardized_all_data(df: pd.DataFrame, pre: ColumnTransformer, numeric_cols, categorical_cols, y_col, out_csv: str):
    X = df[numeric_cols + categorical_cols].copy()
    y = df[y_col].astype(float).values
    Xt = pre.fit_transform(X)
    feat_names = get_feature_names(pre)
    std_df = pd.DataFrame(Xt, columns=feat_names, index=df.index)
    std_df[y_col] = y
    # Include keys for reference
    std_df.insert(0, "TEAM_ID", df["TEAM_ID"].values)
    std_df.insert(1, "TEAM_NAME", df["TEAM_NAME"].values)
    std_df.insert(2, "SEASON_ORIG", df["SEASON"].values)  # original season label
    std_df.to_csv(out_csv, index=False)
    return std_df

def season_out_folds(df: pd.DataFrame) -> list:
    """Yield (train_idx, test_idx, season) for leave-one-season-out CV."""
    seasons = sorted(df["SEASON"].unique().tolist())
    for s in seasons:
        test_idx = df.index[df["SEASON"] == s].to_numpy()
        train_idx = df.index[df["SEASON"] != s].to_numpy()
        yield train_idx, test_idx, s

def evaluate_model(df: pd.DataFrame, numeric_cols, categorical_cols, y_col, random_state=123):
    rng = check_random_state(random_state)

    pre = build_preprocessor(numeric_cols, categorical_cols)

    # Base models
    enet = ElasticNetCV(
        l1_ratio=[0.05, 0.2, 0.5, 0.8, 0.95],
        alphas=None,             # will generate path
        cv=5,                    # inner CV on training fold
        max_iter=20000,
        n_jobs=None,
        random_state=random_state
    )

    if XGB_AVAILABLE:
        xgb = XGBRegressor(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=random_state
        )
    else:
        xgb = None

    rows = []
    y_true_all = []
    y_pred_all_en = []
    y_pred_all_xgb = [] if xgb else None

    for train_idx, test_idx, season in season_out_folds(df):
        X_train = df.loc[train_idx, numeric_cols + categorical_cols]
        y_train = df.loc[train_idx, y_col].astype(float).values
        X_test  = df.loc[test_idx,  numeric_cols + categorical_cols]
        y_test  = df.loc[test_idx,  y_col].astype(float).values

        # Fresh preprocessor per fold (avoid leakage)
        from sklearn.base import clone as _clone
        pre_fold = _clone(pre)
        Xtr = pre_fold.fit_transform(X_train)
        Xte = pre_fold.transform(X_test)

        # ElasticNet
        from sklearn.base import clone as _clone2
        en = _clone2(enet)
        en.fit(Xtr, y_train)
        pred_en = np.clip(en.predict(Xte), 0.0, 1.0)  # WIN_PCT is [0,1]
        mae_en = mean_absolute_error(y_test, pred_en)
        rmse_en = mean_squared_error(y_test, pred_en)
        r2_en = r2_score(y_test, pred_en)

        # XGB (optional)
        if xgb:
            from sklearn.base import clone as _clone3
            xg = _clone3(xgb)
            xg.fit(Xtr, y_train)
            pred_xg = np.clip(xg.predict(Xte), 0.0, 1.0)
            mae_xg = mean_absolute_error(y_test, pred_xg)
            rmse_xg = mean_squared_error(y_test, pred_xg)
            r2_xg = r2_score(y_test, pred_xg)
        else:
            mae_xg = rmse_xg = r2_xg = np.nan

        rows.append({
            "SEASON_OUT": season,
            "MAE_EN": mae_en, "RMSE_EN": rmse_en, "R2_EN": r2_en,
            "MAE_XGB": mae_xg, "RMSE_XGB": rmse_xg, "R2_XGB": r2_xg,
            "N_TEST": len(test_idx)
        })

        # Collect for overall
        y_true_all.append(y_test)
        y_pred_all_en.append(pred_en)
        if xgb:
            y_pred_all_xgb.append(pred_xg)

    rep = pd.DataFrame(rows).sort_values("SEASON_OUT").reset_index(drop=True)

    # Overall metrics
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all_en = np.concatenate(y_pred_all_en, axis=0)
    overall = {
        "OVERALL_MAE_EN": mean_absolute_error(y_true_all, y_pred_all_en),
        "OVERALL_RMSE_EN": mean_squared_error(y_true_all, y_pred_all_en),
        "OVERALL_R2_EN": r2_score(y_true_all, y_pred_all_en),
    }
    if xgb:
        y_pred_all_xgb = np.concatenate(y_pred_all_xgb, axis=0)
        overall.update({
            "OVERALL_MAE_XGB": mean_absolute_error(y_true_all, y_pred_all_xgb),
            "OVERALL_RMSE_XGB": mean_squared_error(y_true_all, y_pred_all_xgb),
            "OVERALL_R2_XGB": r2_score(y_true_all, y_pred_all_xgb),
        })

    return rep, overall, pre, enet, (xgb if XGB_AVAILABLE else None)

def save_report(rep: pd.DataFrame, overall: dict, out_path: Path):
    lines = []
    lines.append("Season-out cross-validation report\n")
    lines.append(rep.to_string(index=False))
    lines.append("\nOverall metrics:")
    for k, v in overall.items():
        lines.append(f"{k}: {v:.6f}")
    out_path.write_text("\n".join(lines))
    print(f"[REPORT] Wrote {out_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="features_with_ids_with_wins.csv",
                        help="Path to features_with_ids_with_wins.csv (must include WIN_PCT).")
    parser.add_argument("--standardized_out", type=str, default="standardized_all_data.csv",
                        help="Where to write the fully standardized dataset CSV.")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts",
                        help="Directory to save preprocessor/model artifacts.")
    args = parser.parse_args()

    df = load_data(args.csv)
    numeric_cols, categorical_cols, y_col = select_feature_columns(df)

    # Build preprocessor and export fully standardized dataset (fit on ALL data)
    pre_all = build_preprocessor(numeric_cols, categorical_cols)
    std_df = export_standardized_all_data(
        df, pre_all, numeric_cols, categorical_cols, y_col, args.standardized_out
    )
    print(f"[STD] Wrote standardized features to {args.standardized_out} (rows={len(std_df)})")

    # Season-out CV with fresh preprocessors per fold (no leakage)
    rep, overall, pre_cv, enet_cv, xgb_cv = evaluate_model(df, numeric_cols, categorical_cols, y_col)

    # Save artifacts
    art_dir = Path(args.artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    import joblib as _joblib
    _joblib.dump(pre_all, art_dir / "preprocessor.joblib")
    _joblib.dump(enet_cv, art_dir / "elasticnet_cv.joblib")
    if xgb_cv is not None:
        _joblib.dump(xgb_cv, art_dir / "xgb.joblib")

    # Report
    save_report(rep, overall, Path("season_out_cv_report.txt"))

    # Print a short TL;DR
    print("\n[TL;DR] Season-out CV (ElasticNet): "
          f"MAE={overall['OVERALL_MAE_EN']:.4f}, RMSE={overall['OVERALL_RMSE_EN']:.4f}, "
          f"R2={overall['OVERALL_R2_EN']:.4f}")
    if 'OVERALL_MAE_XGB' in overall:
        print("[TL;DR] Season-out CV (XGB): "
              f"MAE={overall['OVERALL_MAE_XGB']:.4f}, RMSE={overall['OVERALL_RMSE_XGB']:.4f}, "
              f"R2={overall['OVERALL_R2_XGB']:.4f}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
