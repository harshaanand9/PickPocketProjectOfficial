import pandas as pd
import numpy as np
from datetime import datetime
import os

def validate_nba_dataset():
    """
    Comprehensive validation of the completed NBA dataset against the original Kaggle data
    Identifies missing games, extra games, and data quality issues
    """
    
    print("🔍 NBA Dataset Validation and Quality Check")
    print("=" * 60)
    
    try:
        # Load datasets
        print("📁 Loading datasets...")
        
        # Load completed dataset
        completed_df = pd.read_csv('nba_complete_dataset_20152025.csv')
        print(f"✅ Completed dataset: {len(completed_df):,} games")
        
        # Load original Kaggle data
        kaggle_df = pd.read_csv('oddsData.csv')
        print(f"✅ Original Kaggle data: {len(kaggle_df):,} rows")
        
        # Load Draft Kings data for reference
        draft_kings_df = pd.read_csv('draft_kings_nba_odds.csv')
        print(f"✅ Draft Kings data: {len(draft_kings_df):,} games")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find required CSV files.")
        print("Make sure these files are in the same directory:")
        print("- nba_complete_dataset_20152025.csv")
        print("- oddsData.csv") 
        print("- draft_kings_nba_odds.csv")
        return
    
    # Filter Kaggle data for our target seasons (2016-2023)
    print(f"\n🔍 Filtering Kaggle data for seasons 2016-2023...")
    kaggle_filtered = kaggle_df[
        (kaggle_df['season'] >= 2016) & 
        (kaggle_df['season'] <= 2023)
    ].copy()
    print(f"✅ Filtered Kaggle data: {len(kaggle_filtered):,} rows")
    
    # Create the same team mapping used in processing
    team_mapping = {
        'Atlanta': 'Atlanta Hawks',
        'Boston': 'Boston Celtics',
        'Brooklyn': 'Brooklyn Nets',
        'Charlotte': 'Charlotte Hornets',
        'Chicago': 'Chicago Bulls',
        'Cleveland': 'Cleveland Cavaliers',
        'Dallas': 'Dallas Mavericks',
        'Denver': 'Denver Nuggets',
        'Detroit': 'Detroit Pistons',
        'Golden State': 'Golden State Warriors',
        'Houston': 'Houston Rockets',
        'Indiana': 'Indiana Pacers',
        'LA Clippers': 'Los Angeles Clippers',
        'LA Lakers': 'Los Angeles Lakers',
        'Memphis': 'Memphis Grizzlies',
        'Miami': 'Miami Heat',
        'Milwaukee': 'Milwaukee Bucks',
        'Minnesota': 'Minnesota Timberwolves',
        'New Orleans': 'New Orleans Pelicans',
        'New York': 'New York Knicks',
        'Oklahoma City': 'Oklahoma City Thunder',
        'Orlando': 'Orlando Magic',
        'Philadelphia': 'Philadelphia 76ers',
        'Phoenix': 'Phoenix Suns',
        'Portland': 'Portland Trail Blazers',
        'Sacramento': 'Sacramento Kings',
        'San Antonio': 'San Antonio Spurs',
        'Toronto': 'Toronto Raptors',
        'Utah': 'Utah Jazz',
        'Washington': 'Washington Wizards',
        'New Jersey': 'Brooklyn Nets',
        'Seattle': 'Oklahoma City Thunder',
        'Charlotte Bobcats': 'Charlotte Hornets',
        'New Orleans Hornets': 'New Orleans Pelicans'
    }
    
    def standardize_team_name(name):
        return team_mapping.get(name, name)
    
    # Process Kaggle data to create expected game list
    print(f"\n🎯 Creating expected games list from Kaggle data...")
    
    kaggle_filtered['team_std'] = kaggle_filtered['team'].apply(standardize_team_name)
    kaggle_filtered['opponent_std'] = kaggle_filtered['opponent'].apply(standardize_team_name)
    kaggle_filtered['is_home'] = kaggle_filtered['home/visitor'] == 'vs'
    
    # Create game keys for Kaggle data
    kaggle_filtered['game_key'] = kaggle_filtered.apply(
        lambda row: f"{row['date']}_{'_'.join(sorted([row['team_std'], row['opponent_std']]))}", 
        axis=1
    )
    
    # Convert Kaggle to expected games format
    expected_games = []
    
    for game_key, group in kaggle_filtered.groupby('game_key'):
        if len(group) == 2:  # Valid game with both teams
            home_row = group[group['is_home'] == True]
            away_row = group[group['is_home'] == False]
            
            if len(home_row) == 1 and len(away_row) == 1:
                home_row = home_row.iloc[0]
                away_row = away_row.iloc[0]
                
                # Convert season format
                season_year = int(home_row['season'])
                season_string = f"{season_year - 1}-{str(season_year)[-2:]}"
                
                expected_games.append({
                    'date': home_row['date'],
                    'home_team': home_row['team_std'],
                    'away_team': away_row['team_std'],
                    'home_score': int(home_row['score']) if pd.notna(home_row['score']) else None,
                    'away_score': int(away_row['score']) if pd.notna(away_row['score']) else None,
                    'season': season_string,
                    'game_key': game_key
                })
    
    expected_df = pd.DataFrame(expected_games)
    print(f"✅ Expected games from Kaggle: {len(expected_df):,}")
    
    # Create game keys for completed dataset
    print(f"\n🔍 Analyzing completed dataset...")
    completed_df['game_key'] = completed_df.apply(
        lambda row: f"{row['date']}_{'_'.join(sorted([row['home_team'], row['away_team']]))}", 
        axis=1
    )
    
    # Convert dates to consistent format for comparison
    completed_df['date'] = pd.to_datetime(completed_df['date']).dt.strftime('%Y-%m-%d')
    expected_df['date'] = pd.to_datetime(expected_df['date']).dt.strftime('%Y-%m-%d')
    
    # Update game keys with standardized dates
    completed_df['game_key'] = completed_df.apply(
        lambda row: f"{row['date']}_{'_'.join(sorted([row['home_team'], row['away_team']]))}", 
        axis=1
    )
    expected_df['game_key'] = expected_df.apply(
        lambda row: f"{row['date']}_{'_'.join(sorted([row['home_team'], row['away_team']]))}", 
        axis=1
    )
    
    # Find missing games (in expected but not in completed)
    expected_keys = set(expected_df['game_key'])
    completed_keys = set(completed_df['game_key'])
    
    missing_keys = expected_keys - completed_keys
    extra_keys = completed_keys - expected_keys
    
    print(f"\n📊 VALIDATION RESULTS:")
    print(f"Expected games (from Kaggle): {len(expected_df):,}")
    print(f"Completed dataset games: {len(completed_df):,}")
    print(f"Missing games: {len(missing_keys):,}")
    print(f"Extra games: {len(extra_keys):,}")
    
    # Analyze missing games
    if missing_keys:
        print(f"\n❌ MISSING GAMES ({len(missing_keys):,}):")
        missing_games = expected_df[expected_df['game_key'].isin(missing_keys)].copy()
        missing_games = missing_games.sort_values(['season', 'date']).reset_index(drop=True)
        
        # Show missing games by season
        missing_by_season = missing_games['season'].value_counts().sort_index()
        print("Missing games by season:")
        for season, count in missing_by_season.items():
            print(f"  {season}: {count} games")
        
        # Show first 10 missing games
        print(f"\nFirst 10 missing games:")
        print(missing_games[['date', 'home_team', 'away_team', 'season']].head(10).to_string(index=False))
        
        # Save missing games to CSV
        missing_games.to_csv('missing_games_analysis.csv', index=False)
        print(f"\n💾 Full missing games list saved as: missing_games_analysis.csv")
    
    # Analyze extra games
    if extra_keys:
        print(f"\n⚠️ EXTRA GAMES ({len(extra_keys):,}):")
        extra_games = completed_df[completed_df['game_key'].isin(extra_keys)].copy()
        extra_games = extra_games.sort_values(['season', 'date']).reset_index(drop=True)
        
        # Show extra games by season
        extra_by_season = extra_games['season'].value_counts().sort_index()
        print("Extra games by season:")
        for season, count in extra_by_season.items():
            print(f"  {season}: {count} games")
        
        # Show first 10 extra games
        print(f"\nFirst 10 extra games:")
        print(extra_games[['date', 'home_team', 'away_team', 'season']].head(10).to_string(index=False))
        
        # Save extra games to CSV
        extra_games.to_csv('extra_games_analysis.csv', index=False)
        print(f"\n💾 Full extra games list saved as: extra_games_analysis.csv")
    
    # Check for data quality issues
    print(f"\n🔍 DATA QUALITY ANALYSIS:")
    
    # Check for games with missing scores that should have them
    games_with_kaggle_match = completed_df[completed_df['game_key'].isin(expected_keys)]
    missing_scores = games_with_kaggle_match[
        (games_with_kaggle_match['home_score'].isna()) | 
        (games_with_kaggle_match['away_score'].isna())
    ]
    
    print(f"Games missing scores (that should have them): {len(missing_scores):,}")
    
    # Check for duplicate games
    duplicates = completed_df[completed_df['game_key'].duplicated(keep=False)]
    print(f"Duplicate games in completed dataset: {len(duplicates):,}")
    
    if len(duplicates) > 0:
        print("Duplicate games:")
        print(duplicates[['date', 'home_team', 'away_team', 'season']].to_string(index=False))
    
    # Season completeness analysis
    print(f"\n📅 SEASON COMPLETENESS ANALYSIS:")
    
    completed_by_season = completed_df['season'].value_counts().sort_index()
    expected_by_season = expected_df['season'].value_counts().sort_index()
    
    print(f"{'Season':<10} {'Expected':<10} {'Completed':<10} {'Missing':<10} {'Extra':<10} {'%Complete':<10}")
    print("-" * 70)
    
    all_seasons = sorted(set(list(completed_by_season.index) + list(expected_by_season.index)))
    
    for season in all_seasons:
        expected_count = expected_by_season.get(season, 0)
        completed_count = completed_by_season.get(season, 0)
        
        # Count missing and extra for this season
        season_missing = len([k for k in missing_keys if season in expected_df[expected_df['game_key'] == k]['season'].values]) if missing_keys else 0
        season_extra = len([k for k in extra_keys if season in completed_df[completed_df['game_key'] == k]['season'].values]) if extra_keys else 0
        
        if expected_count > 0:
            pct_complete = (completed_count / expected_count) * 100
        else:
            pct_complete = 100 if completed_count == 0 else float('inf')
        
        print(f"{season:<10} {expected_count:<10} {completed_count:<10} {season_missing:<10} {season_extra:<10} {pct_complete:<10.1f}%")
    
    # Summary and recommendations
    print(f"\n✅ SUMMARY AND RECOMMENDATIONS:")
    
    if len(missing_keys) == 0 and len(extra_keys) == 0:
        print("🎉 PERFECT MATCH! Your dataset contains exactly the expected games from Kaggle data.")
    else:
        if len(missing_keys) > 0:
            print(f"⚠️  {len(missing_keys):,} games are missing from your dataset")
            print("   - Check missing_games_analysis.csv for details")
            print("   - These might be games where team name matching failed")
            print("   - Or games that had incomplete data in the original Kaggle dataset")
        
        if len(extra_keys) > 0:
            print(f"ℹ️  {len(extra_keys):,} extra games in your dataset")
            print("   - These likely come from your Draft Kings data")
            print("   - Check extra_games_analysis.csv for details")
            print("   - This is normal if Draft Kings has games not in Kaggle data")
    
    coverage_pct = (len(expected_keys & completed_keys) / len(expected_keys)) * 100 if expected_keys else 0
    print(f"\n📊 Overall Kaggle Data Coverage: {coverage_pct:.1f}%")
    print(f"📊 Total Unique Games: {len(completed_keys):,}")

def main():
    """Main function to run the validation"""
    try:
        validate_nba_dataset()
        
    except Exception as e:
        print(f"❌ An error occurred during validation: {str(e)}")
        print("Please check your CSV files and try again.")

if __name__ == "__main__":
    main()