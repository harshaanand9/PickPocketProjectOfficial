import pandas as pd
import numpy as np
from datetime import datetime
import os

def fill_2024_25_scores():
    """
    Fill missing scores for 2024-25 season using TeamStatistics.csv
    Team that appears first is home team, second is away team
    """
    
    print("🏀 NBA 2024-25 Season Score Filler")
    print("=" * 50)
    
    # Load the current dataset
    try:
        df = pd.read_csv('nba_complete_dataset_20152025_final.csv')
        print(f"✅ Loaded current dataset: {len(df):,} games")
    except FileNotFoundError:
        print("❌ Error: nba_complete_dataset_20152025_final.csv not found")
        return None
    
    # Load TeamStatistics dataset
    try:
        team_stats = pd.read_csv('TeamStatistics.csv')
        print(f"✅ Loaded TeamStatistics dataset: {len(team_stats):,} rows")
    except FileNotFoundError:
        print("❌ Error: TeamStatistics.csv not found")
        return None
    
    print(f"\n📊 TeamStatistics columns: {list(team_stats.columns)}")
    print(f"\n📋 Sample TeamStatistics data:")
    print(team_stats[['gameDate', 'teamCity', 'teamName', 'home', 'teamScore', 'opponentScore']].head(6).to_string())
    
    # Check available date range in TeamStatistics
    team_stats['gameDate'] = pd.to_datetime(team_stats['gameDate'])
    date_range = f"{team_stats['gameDate'].min().strftime('%Y-%m-%d')} to {team_stats['gameDate'].max().strftime('%Y-%m-%d')}"
    print(f"\n📅 TeamStatistics date range: {date_range}")
    
    # Filter for 2024-25 season (starts around October 2024)
    season_2024_25 = team_stats[
        (team_stats['gameDate'] >= '2024-10-01') & 
        (team_stats['gameDate'] <= '2025-09-30')
    ].copy()
    
    print(f"✅ 2024-25 season data: {len(season_2024_25):,} rows")
    
    # Create comprehensive team name mapping
    team_mapping = {
        # Handle city + name combinations
        'Atlanta Hawks': 'Atlanta Hawks',
        'Boston Celtics': 'Boston Celtics',
        'Brooklyn Nets': 'Brooklyn Nets',
        'Charlotte Hornets': 'Charlotte Hornets',
        'Chicago Bulls': 'Chicago Bulls',
        'Cleveland Cavaliers': 'Cleveland Cavaliers',
        'Dallas Mavericks': 'Dallas Mavericks',
        'Denver Nuggets': 'Denver Nuggets',
        'Detroit Pistons': 'Detroit Pistons',
        'Golden State Warriors': 'Golden State Warriors',
        'Houston Rockets': 'Houston Rockets',
        'Indiana Pacers': 'Indiana Pacers',
        'Los Angeles Clippers': 'Los Angeles Clippers',
        'Los Angeles Lakers': 'Los Angeles Lakers',
        'Memphis Grizzlies': 'Memphis Grizzlies',
        'Miami Heat': 'Miami Heat',
        'Milwaukee Bucks': 'Milwaukee Bucks',
        'Minnesota Timberwolves': 'Minnesota Timberwolves',
        'New Orleans Pelicans': 'New Orleans Pelicans',
        'New York Knicks': 'New York Knicks',
        'Oklahoma City Thunder': 'Oklahoma City Thunder',
        'Orlando Magic': 'Orlando Magic',
        'Philadelphia 76ers': 'Philadelphia 76ers',
        'Phoenix Suns': 'Phoenix Suns',
        'Portland Trail Blazers': 'Portland Trail Blazers',
        'Sacramento Kings': 'Sacramento Kings',
        'San Antonio Spurs': 'San Antonio Spurs',
        'Toronto Raptors': 'Toronto Raptors',
        'Utah Jazz': 'Utah Jazz',
        'Washington Wizards': 'Washington Wizards'
    }
    
    def create_full_team_name(city, name):
        """Create full team name from city and team name"""
        if pd.isna(city) or pd.isna(name):
            return None
        
        full_name = f"{str(city).strip()} {str(name).strip()}"
        
        # Handle special cases
        if 'LA' in city and 'Lakers' in name:
            return 'Los Angeles Lakers'
        elif 'LA' in city and 'Clippers' in name:
            return 'Los Angeles Clippers'
        elif city == 'Los Angeles' and 'Lakers' in name:
            return 'Los Angeles Lakers'
        elif city == 'Los Angeles' and 'Clippers' in name:
            return 'Los Angeles Clippers'
        
        # Return standardized name or the constructed name
        return team_mapping.get(full_name, full_name)
    
    # Create full team names
    season_2024_25['full_team_name'] = season_2024_25.apply(
        lambda row: create_full_team_name(row['teamCity'], row['teamName']), axis=1
    )
    season_2024_25['full_opponent_name'] = season_2024_25.apply(
        lambda row: create_full_team_name(row['opponentTeamCity'], row['opponentTeamName']), axis=1
    )
    
    # Convert dates to consistent format
    season_2024_25['date_str'] = season_2024_25['gameDate'].dt.strftime('%Y-%m-%d')
    
    # Identify games missing scores in 2024-25 season
    missing_scores_2024_25 = df[
        (df['season'] == '2024-25') &
        ((df['home_score'].isna()) | (df['away_score'].isna()))
    ].copy()
    
    print(f"\n🔍 2024-25 season analysis:")
    total_2024_25 = len(df[df['season'] == '2024-25'])
    print(f"   Total 2024-25 games in dataset: {total_2024_25}")
    print(f"   Games missing scores: {len(missing_scores_2024_25)}")
    
    # Convert dates to consistent format for matching
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    # Show unique team names for debugging
    print(f"\n🔍 Team name analysis:")
    stats_teams = set(season_2024_25['full_team_name'].dropna().unique())
    dataset_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
    
    print(f"   TeamStats teams (2024-25): {len(stats_teams)}")
    print(f"   Dataset teams: {len(dataset_teams)}")
    
    unmatched_stats = stats_teams - dataset_teams
    unmatched_dataset = dataset_teams - stats_teams
    
    if unmatched_stats:
        print(f"   Unmatched TeamStats teams: {unmatched_stats}")
    if unmatched_dataset:
        print(f"   Unmatched dataset teams: {unmatched_dataset}")
    
    # Fill scores by matching games - ONLY target rows with empty scores
    print(f"\n🔄 Filling 2024-25 missing scores...")
    
    scores_filled = 0
    
    for idx, game in missing_scores_2024_25.iterrows():
        # Skip if both scores are already filled
        if pd.notna(game['home_score']) and pd.notna(game['away_score']):
            continue
        
        # Find matching games in TeamStatistics
        # Look for games where this team combination played on this date
        game_matches = season_2024_25[
            (season_2024_25['date_str'] == game['date']) &
            (
                # Home team is first team in TeamStatistics (home=1)
                ((season_2024_25['full_team_name'] == game['home_team']) & 
                 (season_2024_25['full_opponent_name'] == game['away_team']) &
                 (season_2024_25['home'] == 1)) |
                # Or away team is first team in TeamStatistics (home=0)  
                ((season_2024_25['full_team_name'] == game['away_team']) & 
                 (season_2024_25['full_opponent_name'] == game['home_team']) &
                 (season_2024_25['home'] == 0))
            )
        ]
        
        if len(game_matches) > 0:
            # Use the first match
            match = game_matches.iloc[0]
            
            # Determine home and away scores based on which team matched
            if match['full_team_name'] == game['home_team'] and match['home'] == 1:
                # Matched team is home team
                home_score = match['teamScore']
                away_score = match['opponentScore']
            elif match['full_team_name'] == game['away_team'] and match['home'] == 0:
                # Matched team is away team  
                home_score = match['opponentScore']
                away_score = match['teamScore']
            else:
                continue  # Skip if logic doesn't match
            
            # Only update scores that are actually missing
            home_updated = False
            away_updated = False
            
            if pd.isna(df.at[idx, 'home_score']) and pd.notna(home_score):
                df.at[idx, 'home_score'] = int(home_score)
                home_updated = True
                
            if pd.isna(df.at[idx, 'away_score']) and pd.notna(away_score):
                df.at[idx, 'away_score'] = int(away_score)
                away_updated = True
            
            if home_updated or away_updated:
                scores_filled += 1
                if scores_filled <= 15:  # Show first 15 matches
                    status = []
                    if home_updated: status.append("home")
                    if away_updated: status.append("away")
                    print(f"   ✅ {game['away_team']} @ {game['home_team']} ({game['date']}) -> {int(away_score)}-{int(home_score)} (filled {'/'.join(status)})")
    
    print(f"\n📊 2024-25 scores filled: {scores_filled} games")
    
    # Save updated dataset (OVERWRITE original file)
    output_filename = 'nba_complete_dataset_20152025_final.csv'
    df.to_csv(output_filename, index=False)
    print(f"\n💾 Updated dataset saved to: {output_filename}")
    
    # Summary
    print(f"\n✅ SUMMARY:")
    print(f"   2024-25 scores filled: {scores_filled} games")
    print(f"   Total games in dataset: {len(df):,}")
    print(f"   Output file: {output_filename}")
    
    # Check remaining missing scores in 2024-25
    remaining_missing_2024_25 = df[
        (df['season'] == '2024-25') & 
        ((df['home_score'].isna()) | (df['away_score'].isna()))
    ]
    
    if len(remaining_missing_2024_25) > 0:
        print(f"\n⚠️  Still missing 2024-25 scores for {len(remaining_missing_2024_25)} games:")
        print(remaining_missing_2024_25[['date', 'home_team', 'away_team']].head(10).to_string(index=False))
    else:
        print(f"\n🎉 All 2024-25 scores now complete!")
    
    return df

def main():
    """Main function"""
    try:
        # Check if required files exist
        required_files = [
            'nba_complete_dataset_20152025_final.csv',
            'TeamStatistics.csv'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print("❌ Missing required files:")
            for file in missing_files:
                print(f"   - {file}")
            return
        
        # Process the data
        updated_df = fill_2024_25_scores()
        
        if updated_df is not None:
            print(f"\n✅ SUCCESS! 2024-25 score filling complete.")
            
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()