import pandas as pd
import numpy as np
from datetime import datetime
import os

def process_nba_data():
    """
    Complete NBA Data Processing Pipeline
    Merges Draft Kings odds data with Kaggle NBA data (2015-2025 seasons)
    """
    
    print("🏀 Starting NBA Data Processing Pipeline...")
    print("=" * 50)
    
    # Step 1: Read the CSV files
    print("📁 Step 1: Reading CSV files...")
    try:
        # Read Draft Kings data
        draft_kings_df = pd.read_csv('draft_kings_nba_odds.csv')
        print(f"✅ Loaded Draft Kings data: {len(draft_kings_df)} rows")
        
        # Read Kaggle odds data  
        kaggle_odds_df = pd.read_csv('oddsData.csv')
        print(f"✅ Loaded Kaggle odds data: {len(kaggle_odds_df)} rows")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find CSV file. Make sure both files are in the same directory as this script.")
        print(f"Expected files: 'draft_kings_nba_odds.csv' and 'oddsData.csv'")
        return None
    
    # Step 2: Filter Kaggle data for seasons 2016-2023 (representing 2015-16 through 2022-23)
    print("\n🔍 Step 2: Filtering for seasons 2016-2023...")
    filtered_kaggle = kaggle_odds_df[
        (kaggle_odds_df['season'] >= 2016) & 
        (kaggle_odds_df['season'] <= 2023)
    ].copy()
    
    print(f"✅ Filtered to {len(filtered_kaggle)} rows for seasons 2016-2023")
    available_seasons = sorted(filtered_kaggle['season'].unique())
    print(f"Available seasons: {available_seasons}")
    
    # Step 3: Create team name standardization mapping
    print("\n🏀 Step 3: Creating team name standardization...")
    
    # Comprehensive mapping from Kaggle format (city names) to full team names
    team_mapping = {
        # Current teams
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
        
        # Historical team names that might appear in older data
        'New Jersey': 'Brooklyn Nets',  # Nets moved from NJ to Brooklyn
        'Seattle': 'Oklahoma City Thunder',  # SuperSonics became Thunder
        'Charlotte Bobcats': 'Charlotte Hornets',  # Bobcats became Hornets again
        'New Orleans Hornets': 'New Orleans Pelicans',  # Hornets became Pelicans
        
        # Alternative formatting that might appear
        'Los Angeles Lakers': 'Los Angeles Lakers',
        'Los Angeles Clippers': 'Los Angeles Clippers',
        'Golden State Warriors': 'Golden State Warriors',
        'San Antonio Spurs': 'San Antonio Spurs',
        'New Orleans Pelicans': 'New Orleans Pelicans',
        'New York Knicks': 'New York Knicks',
        'Oklahoma City Thunder': 'Oklahoma City Thunder'
    }
    
    def standardize_team_name(name):
        return team_mapping.get(name, name)
    
    # Apply team name standardization
    filtered_kaggle['team'] = filtered_kaggle['team'].apply(standardize_team_name)
    filtered_kaggle['opponent'] = filtered_kaggle['opponent'].apply(standardize_team_name)
    
    # Step 4: Convert Kaggle data from 2-rows-per-game to 1-row-per-game format
    print("\n🎯 Step 4: Converting to game-based format...")
    
    # Create game groups
    filtered_kaggle['is_home'] = filtered_kaggle['home/visitor'] == 'vs'
    filtered_kaggle['game_key'] = filtered_kaggle.apply(
        lambda row: f"{row['date']}_{'_'.join(sorted([row['team'], row['opponent']]))}", 
        axis=1
    )
    
    # Group by game and convert to single row per game
    games_list = []
    
    for game_key, group in filtered_kaggle.groupby('game_key'):
        if len(group) == 2:  # Should have exactly 2 rows per game
            # Find home and away teams
            home_row = group[group['is_home'] == True]
            away_row = group[group['is_home'] == False]
            
            if len(home_row) == 1 and len(away_row) == 1:
                home_row = home_row.iloc[0]
                away_row = away_row.iloc[0]
                
                # Convert season format (2016 -> "2015-16")
                season_year = int(home_row['season'])
                season_string = f"{season_year - 1}-{str(season_year)[-2:]}"
                
                game_record = {
                    'date': home_row['date'],
                    'home_team': home_row['team'],
                    'away_team': away_row['team'],
                    'home_money_line': float(home_row['moneyLine']) if pd.notna(home_row['moneyLine']) else None,
                    'home_spread': float(home_row['spread']) if pd.notna(home_row['spread']) else None,
                    'away_money_line': float(away_row['moneyLine']) if pd.notna(away_row['moneyLine']) else None,
                    'away_spread': float(away_row['spread']) if pd.notna(away_row['spread']) else None,
                    'home_score': int(home_row['score']) if pd.notna(home_row['score']) else None,
                    'away_score': int(away_row['score']) if pd.notna(away_row['score']) else None,
                    'season': season_string
                }
                
                games_list.append(game_record)
    
    kaggle_games_df = pd.DataFrame(games_list)
    print(f"✅ Created {len(kaggle_games_df)} games from Kaggle data")
    
    # Step 5: Process Draft Kings data (remove unwanted columns and add scores)
    print("\n🔗 Step 5: Processing Draft Kings data...")
    
    # Remove unwanted columns and prepare the data
    draft_kings_processed = draft_kings_df.drop(['api_call_date', 'game_id'], axis=1).copy()
    
    # Add score columns initialized as None
    draft_kings_processed['home_score'] = None
    draft_kings_processed['away_score'] = None
    
    # Try to match games with Kaggle data to get scores
    for idx, dk_game in draft_kings_processed.iterrows():
        # Find matching game in Kaggle data
        matching_games = kaggle_games_df[
            (kaggle_games_df['date'] == dk_game['date']) &
            (kaggle_games_df['home_team'] == dk_game['home_team']) &
            (kaggle_games_df['away_team'] == dk_game['away_team'])
        ]
        
        if len(matching_games) > 0:
            match = matching_games.iloc[0]
            draft_kings_processed.at[idx, 'home_score'] = match['home_score']
            draft_kings_processed.at[idx, 'away_score'] = match['away_score']
    
    print(f"✅ Processed Draft Kings data: {len(draft_kings_processed)} games")
    
    # Step 6: Combine datasets, avoiding duplicates
    print("\n🎯 Step 6: Combining datasets and removing duplicates...")
    
    # Start with Kaggle games (these have scores)
    combined_df = kaggle_games_df.copy()
    
    # Add Draft Kings games that don't already exist
    for idx, dk_game in draft_kings_processed.iterrows():
        # Check if this game already exists in combined data
        existing_game = combined_df[
            (combined_df['date'] == dk_game['date']) &
            (combined_df['home_team'] == dk_game['home_team']) &
            (combined_df['away_team'] == dk_game['away_team'])
        ]
        
        if len(existing_game) == 0:
            # Add this game as it's not a duplicate
            combined_df = pd.concat([combined_df, pd.DataFrame([dk_game])], ignore_index=True)
    
    # Step 7: Sort by date and finalize
    print("\n📊 Step 7: Finalizing dataset...")
    
    # Convert date to datetime for sorting
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df = combined_df.sort_values('date').reset_index(drop=True)
    
    # Convert date back to string format
    combined_df['date'] = combined_df['date'].dt.strftime('%Y-%m-%d')
    
    # Ensure column order
    column_order = [
        'date', 'home_team', 'away_team', 'home_money_line', 'home_spread', 
        'away_money_line', 'away_spread', 'home_score', 'away_score', 'season'
    ]
    
    final_df = combined_df[column_order].copy()
    
    # Step 8: Generate statistics
    print("\n📈 Step 8: Generating statistics...")
    
    total_games = len(final_df)
    games_with_scores = len(final_df[final_df['home_score'].notna()])
    games_with_odds = len(final_df[final_df['home_money_line'].notna()])
    
    # Season breakdown
    season_counts = final_df['season'].value_counts().sort_index()
    
    print(f"\n🎉 Processing Complete!")
    print("=" * 50)
    print(f"📊 FINAL DATASET STATISTICS:")
    print(f"   Total Games: {total_games:,}")
    print(f"   Games with Scores: {games_with_scores:,}")
    print(f"   Games with Odds: {games_with_odds:,}")
    print(f"   Date Range: {final_df['date'].min()} to {final_df['date'].max()}")
    
    print(f"\n📅 GAMES BY SEASON:")
    for season, count in season_counts.items():
        print(f"   {season}: {count:,} games")
    
    # Step 9: Save to CSV
    output_filename = 'nba_complete_dataset_2015-2025.csv'
    final_df.to_csv(output_filename, index=False)
    print(f"\n💾 Dataset saved as: {output_filename}")
    
    # Display sample data
    print(f"\n📋 SAMPLE DATA (First 10 games):")
    print(final_df.head(10).to_string(index=False))
    
    return final_df

def main():
    """Main function to run the NBA data processing"""
    try:
        # Check if required files exist
        required_files = ['draft_kings_nba_odds.csv', 'oddsData.csv']
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print("❌ Missing required files:")
            for file in missing_files:
                print(f"   - {file}")
            print("\nPlease ensure both CSV files are in the same directory as this script.")
            return
        
        # Process the data
        result_df = process_nba_data()
        
        if result_df is not None:
            print(f"\n✅ SUCCESS! Your complete NBA dataset has been created.")
            print(f"📁 File saved as: nba_complete_dataset_2015-2025.csv")
            print(f"📊 Total records: {len(result_df):,}")
            
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        print("Please check your CSV files and try again.")

if __name__ == "__main__":
    main()