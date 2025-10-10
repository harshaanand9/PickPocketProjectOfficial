# scripts/reorder_games_by_date.py
import pandas as pd
import numpy as np

INFILE  = "games_13-15.csv"
OUTFILE = "games_13-15_sorted.csv"

def main():
    # Read as strings to avoid dtype surprises
    df = pd.read_csv(INFILE, dtype=str)

    # Parse dates robustly (data is MM/DD/YYYY from your builder)
    # If any row is malformed, coerce to NaT; we'll push NaT to the end
    dt = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df["__date_dt"] = dt

    # Sort by date, then a deterministic tiebreaker (home/away) to keep order stable
    df_sorted = (
        df.sort_values(["__date_dt", "home_team", "away_team"], kind="mergesort")
          .drop(columns="__date_dt")
          .reset_index(drop=True)
    )

    # Optional: put any NaT rows (if they exist) at the end explicitly
    if df_sorted["date"].isna().any():
        good = df_sorted[df_sorted["date"].notna()]
        bad  = df_sorted[df_sorted["date"].isna()]
        df_sorted = pd.concat([good, bad], ignore_index=True)

    df_sorted.to_csv(OUTFILE, index=False)
    print(f"Sorted {len(df_sorted)} rows by date -> {OUTFILE}")

if __name__ == "__main__":
    main()
