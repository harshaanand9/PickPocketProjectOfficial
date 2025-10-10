import pandas as pd
import numpy as np

def check_na_values():
    """
    Check for NA/null values in training_set.csv and show exact line numbers
    """
    
    print("🔍 Checking NA Values in Training Set")
    print("=" * 50)
    
    # Load the training set
    try:
        df = pd.read_csv('training_set.csv')
        print(f"✅ Loaded training set: {len(df):,} rows, {len(df.columns)} columns")
    except FileNotFoundError:
        print("❌ Error: training_set.csv not found")
        return
    
    print(f"\n📊 Dataset shape: {df.shape}")
    print(f"📊 Columns: {list(df.columns)}")
    
    # Check for NA values in each column
    print(f"\n🔍 NA VALUE SUMMARY:")
    print("-" * 60)
    
    total_na = 0
    for col in df.columns:
        na_count = df[col].isna().sum()
        total_na += na_count
        if na_count > 0:
            print(f"❌ {col}: {na_count:,} NA values")
        else:
            print(f"✅ {col}: No NA values")
    
    print(f"\n📊 Total NA values across all columns: {total_na:,}")
    
    if total_na == 0:
        print(f"\n🎉 PERFECT! No NA values found in the dataset!")
        return df
    
    # Find rows with ANY NA values
    rows_with_na = df[df.isna().any(axis=1)]
    print(f"\n📍 ROWS WITH NA VALUES:")
    print(f"   Total rows with NA: {len(rows_with_na):,}")
    print("-" * 60)
    
    # Show detailed breakdown by column
    for col in df.columns:
        if df[col].isna().sum() > 0:
            na_rows = df[df[col].isna()]
            print(f"\n❌ Column '{col}' - {len(na_rows)} NA values:")
            print(f"   Line numbers (0-indexed): {list(na_rows.index)}")
            print(f"   Line numbers (1-indexed): {[i+1 for i in na_rows.index]}")
            
            # Show first 10 rows with NA in this column
            if len(na_rows) <= 10:
                print(f"   All rows with NA in '{col}':")
                display_cols = ['date', 'home_team', 'away_team', col, 'season']
                print(na_rows[display_cols].to_string())
            else:
                print(f"   First 10 rows with NA in '{col}':")
                display_cols = ['date', 'home_team', 'away_team', col, 'season']
                print(na_rows[display_cols].head(10).to_string())
                print(f"   ... and {len(na_rows) - 10} more rows")
    
    # Summary by season
    if len(rows_with_na) > 0:
        print(f"\n📅 NA VALUES BY SEASON:")
        print("-" * 30)
        for season in sorted(df['season'].unique()):
            season_data = df[df['season'] == season]
            season_na = season_data[season_data.isna().any(axis=1)]
            if len(season_na) > 0:
                print(f"   {season}: {len(season_na):,} rows with NA values")
        
        # Show specific missing data patterns
        print(f"\n🔍 MISSING DATA PATTERNS:")
        print("-" * 40)
        
        # Check for missing scores
        missing_scores = df[(df['home_score'].isna()) | (df['away_score'].isna())]
        if len(missing_scores) > 0:
            print(f"❌ Games missing scores: {len(missing_scores)}")
            by_season = missing_scores['season'].value_counts().sort_index()
            for season, count in by_season.items():
                print(f"   {season}: {count} games")
        
        # Check for missing odds
        missing_odds = df[(df['home_money_line'].isna()) | (df['away_money_line'].isna()) | 
                         (df['home_spread'].isna()) | (df['away_spread'].isna())]
        if len(missing_odds) > 0:
            print(f"❌ Games missing odds: {len(missing_odds)}")
            by_season = missing_odds['season'].value_counts().sort_index()
            for season, count in by_season.items():
                print(f"   {season}: {count} games")
    
    # Data completeness percentage
    total_cells = df.shape[0] * df.shape[1]
    filled_cells = total_cells - total_na
    completeness = (filled_cells / total_cells) * 100
    
    print(f"\n📊 DATA COMPLETENESS:")
    print(f"   Total cells: {total_cells:,}")
    print(f"   Filled cells: {filled_cells:,}")
    print(f"   NA cells: {total_na:,}")
    print(f"   Completeness: {completeness:.2f}%")
    
    return df

def main():
    """Main function to run the NA check"""
    try:
        df = check_na_values()
        
        if df is not None:
            print(f"\n✅ NA value analysis complete!")
            
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()