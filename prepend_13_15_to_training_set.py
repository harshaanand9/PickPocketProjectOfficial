# scripts/prepend_13_15_to_training_set.py
# Prepend games_13-15_sorted.csv to training_set.csv, keeping the training_set schema.

import pandas as pd
import numpy as np

GAMES_FILE   = "games_13-15_sorted.csv"   # finalized earlier segment
TRAIN_FILE   = "training_set.csv"         # your existing dataset
OUT_FILE     = "training_set_complete.csv"  # output (change to TRAIN_FILE to overwrite)

def main():
    # Read both as strings to avoid dtype surprises; preserve exact text formatting (e.g., 'MM/DD/YYYY')
    df_train = pd.read_csv(TRAIN_FILE, dtype=str, low_memory=False)
    df_old   = pd.read_csv(GAMES_FILE, dtype=str, low_memory=False)

    # Normalize column set to match training_set schema
    train_cols = list(df_train.columns)

    # Add any missing columns to df_old (filled with NaN)
    for c in train_cols:
        if c not in df_old.columns:
            df_old[c] = np.nan

    # Drop any extra columns from df_old that aren’t in training_set
    extras = [c for c in df_old.columns if c not in train_cols]
    if extras:
        df_old = df_old.drop(columns=extras)

    # Reorder to exact training_set column order
    df_old = df_old[train_cols]

    # Optional: sanity-check required key columns exist
    required = ["date", "home_team", "away_team", "season"]
    missing_required = [c for c in required if c not in df_old.columns]
    if missing_required:
        raise ValueError(f"Missing required columns in prepended file: {missing_required}")

    # Concatenate with old rows FIRST (i.e., prepend), then training rows
    combined = pd.concat([df_old, df_train], ignore_index=True)

    # Optional: if you want strictly chronological order across the whole file,
    # uncomment the sort below (it keeps stable order for same dates):
    # dt = pd.to_datetime(combined["date"], format="%m/%d/%Y", errors="coerce")
    # combined = (combined.assign(__dt=dt)
    #                     .sort_values(["__dt", "home_team", "away_team"], kind="mergesort")
    #                     .drop(columns="__dt")
    #                     .reset_index(drop=True))

    combined.to_csv(OUT_FILE, index=False)
    print(f"Prepended {len(df_old)} rows to {len(df_train)} rows → wrote {OUT_FILE} ({len(combined)} total).")

if __name__ == "__main__":
    main()
