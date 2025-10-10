import pandas as pd
import numpy as np
from datetime import datetime
import os

def fill_missing_scores_from_team_traditional():
    """
    Fill missing scores for 2019-20, 2022-23, 2023-24, and 2024-25 seasons
    using the team_traditional.csv dataset
    """
    
    print("🏀 NBA Score Filler - Using Team Traditional Dataset")
    print("=" * 60)
    
    # Load the current dataset
    try:
        df = pd.read_csv('nba_complete_dataset_20152025_final.csv')
        print(f"✅ Loaded current dataset: {len(df):,} games")
    except FileNotFoundError:
        print("❌ Error: nba_complete_dataset_20152025_final.csv not found")
        return None
    
    # Load team traditional dataset
    try:
        team_stats = pd.read_csv('team_traditional.csv')
        print(f"✅ Loaded team traditional dataset: {len(team_stats):,} rows")
    except FileNotFoundError:
        print("❌ Error: team_traditional.csv not found")
        return None
    
    print(f"\n📊 Team traditional dataset columns: {list(team_stats.columns)}")
    print(f"\n📋 Sample team traditional data:")
    print(team_stats[['date', 'team', 'home', 'away', 'PTS', 'season']].head(6).to_string())
    
    # Check available seasons in team_stats
    available_seasons = sorted(team_stats['season'].unique())
    print(f"\n📅 Available seasons in team_stats: {available_seasons}")
    
    # Create comprehensive team name mapping (FIXED: BKN for Brooklyn Nets)
    team_mapping = {
        # Handle various team name formats - CORRECTED BROOKLYN NETS
        'ATL': 'Atlanta Hawks', 'Atlanta': 'Atlanta Hawks', 'Hawks': 'Atlanta Hawks',
        'BOS': 'Boston Celtics', 'Boston': 'Boston Celtics', 'Celtics': 'Boston Celtics',
        'BKN': 'Brooklyn Nets', 'BRK': 'Brooklyn Nets', 'Brooklyn': 'Brooklyn Nets', 'Nets': 'Brooklyn Nets',  # FIXED
        'CHA': 'Charlotte Hornets', 'Charlotte': 'Charlotte Hornets', 'Hornets': 'Charlotte Hornets',
        'CHI': 'Chicago Bulls', 'Chicago': 'Chicago Bulls', 'Bulls': 'Chicago Bulls',
        'CLE': 'Cleveland Cavaliers', 'Cleveland': 'Cleveland Cavaliers', 'Cavaliers': 'Cleveland Cavaliers',
        'DAL': 'Dallas Mavericks', 'Dallas': 'Dallas Mavericks', 'Mavericks': 'Dallas Mavericks',
        'DEN': 'Denver Nuggets', 'Denver': 'Denver Nuggets', 'Nuggets': 'Denver Nuggets',
        'DET': 'Detroit Pistons', 'Detroit': 'Detroit Pistons', 'Pistons': 'Detroit Pistons',
        'GSW': 'Golden State Warriors', 'Golden State': 'Golden State Warriors', 'Warriors': 'Golden State Warriors',
        'HOU': 'Houston Rockets', 'Houston': 'Houston Rockets', 'Rockets': 'Houston Rockets',
        'IND': 'Indiana Pacers', 'Indiana': 'Indiana Pacers', 'Pacers': 'Indiana Pacers',
        'LAC': 'Los Angeles Clippers', 'LA Clippers': 'Los Angeles Clippers', 'Clippers': 'Los Angeles Clippers',
        'LAL': 'Los Angeles Lakers', 'LA Lakers': 'Los Angeles Lakers', 'Lakers': 'Los Angeles Lakers',
        'MEM': 'Memphis Grizzlies', 'Memphis': 'Memphis Grizzlies', 'Grizzlies': 'Memphis Grizzlies',
        'MIA': 'Miami Heat', 'Miami': 'Miami Heat', 'Heat': 'Miami Heat',
        'MIL': 'Milwaukee Bucks', 'Milwaukee': 'Milwaukee Bucks', 'Bucks': 'Milwaukee Bucks',
        'MIN': 'Minnesota Timberwolves', 'Minnesota': 'Minnesota Timberwolves', 'Timberwolves': 'Minnesota Timberwolves',
        'NOP': 'New Orleans Pelicans', 'New Orleans': 'New Orleans Pelicans', 'Pelicans': 'New Orleans Pelicans',
        'NYK': 'New York Knicks', 'New York': 'New York Knicks', 'Knicks': 'New York Knicks',
        'OKC': 'Oklahoma City Thunder', 'Oklahoma City': 'Oklahoma City Thunder', 'Thunder': 'Oklahoma City Thunder',
        'ORL': 'Orlando Magic', 'Orlando': 'Orlando Magic', 'Magic': 'Orlando Magic',
        'PHI': 'Philadelphia 76ers', 'Philadelphia': 'Philadelphia 76ers', '76ers': 'Philadelphia 76ers',
        'PHX': 'Phoenix Suns', 'Phoenix': 'Phoenix Suns', 'Suns': 'Phoenix Suns',
        'POR': 'Portland Trail Blazers', 'Portland': 'Portland Trail Blazers', 'Trail Blazers': 'Portland Trail Blazers',
        'SAC': 'Sacramento Kings', 'Sacramento': 'Sacramento Kings', 'Kings': 'Sacramento Kings',
        'SAS': 'San Antonio Spurs', 'San Antonio': 'San Antonio Spurs', 'Spurs': 'San Antonio Spurs',
        'TOR': 'Toronto Raptors', 'Toronto': 'Toronto Raptors', 'Raptors': 'Toronto Raptors',
        'UTA': 'Utah Jazz', 'Utah': 'Utah Jazz', 'Jazz': 'Utah Jazz',
        'WAS': 'Washington Wizards', 'Washington': 'Washington Wizards', 'Wizards': 'Washington Wizards',
        
        # Full team names (already standardized)
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
        'Washington Wizards': 'Washington Wizards',
        
        # Historical names
        'New Jersey Nets': 'Brooklyn Nets',
        'Seattle SuperSonics': 'Oklahoma City Thunder',
        'Charlotte Bobcats': 'Charlotte Hornets',
        'New Orleans Hornets': 'New Orleans Pelicans'
    }
    
    def standardize_team_name(name):
        """Standardize team names to match our dataset format"""
        if pd.isna(name):
            return name
        return team_mapping.get(str(name).strip(), str(name).strip())
    
    # Standardize team names in team_stats dataset
    team_stats['team_std'] = team_stats['team'].apply(standardize_team_name)
    team_stats['home_std'] = team_stats['home'].apply(standardize_team_name)
    team_stats['away_std'] = team_stats['away'].apply(standardize_team_name)
    
    # Convert season format in team_stats to match our dataset (2020 -> "2019-20")
    def convert_season_format(season_int):
        """Convert season integer (2020) to string format (2019-20)"""
        if pd.isna(season_int):
            return season_int
        year = int(season_int)
        return f"{year-1}-{str(year)[-2:]}"
    
    team_stats['season_str'] = team_stats['season'].apply(convert_season_format)
    
    # Identify games missing scores in target seasons
    target_seasons = ['2019-20', '2022-23', '2023-24', '2024-25']
    
    missing_scores = df[
        (df['season'].isin(target_seasons)) &
        ((df['home_score'].isna()) | (df['away_score'].isna()))
    ].copy()
    
    print(f"\n🔍 Analysis of missing scores:")
    print(f"   Total games in target seasons: {len(df[df['season'].isin(target_seasons)])}")
    print(f"   Games missing scores: {len(missing_scores)}")
    
    if len(missing_scores) > 0:
        missing_by_season = missing_scores['season'].value_counts().sort_index()
        print(f"\n📅 Missing scores by season:")
        for season, count in missing_by_season.items():
            print(f"   {season}: {count} games")
    
    # Convert dates to consistent format for matching
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    team_stats['date'] = pd.to_datetime(team_stats['date']).dt.strftime('%Y-%m-%d')
    
    # Show unique team names for debugging
    print(f"\n🔍 Team name analysis:")
    team_stats_teams = set(team_stats['team_std'].unique())
    dataset_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
    
    print(f"   Team stats teams: {len(team_stats_teams)}")
    print(f"   Dataset teams: {len(dataset_teams)}")
    unmatched_stats = team_stats_teams - dataset_teams
    unmatched_dataset = dataset_teams - team_stats_teams
    
    if unmatched_stats:
        print(f"   Unmatched team_stats teams: {unmatched_stats}")
    if unmatched_dataset:
        print(f"   Unmatched dataset teams: {unmatched_dataset}")
    
    # Fill scores by matching games - ONLY target rows with empty scores
    print(f"\n🔄 Filling missing scores (targeting only empty values)...")
    
    scores_filled = 0
    
    # Only process games that actually have missing scores
    for idx, game in missing_scores.iterrows():
        # Skip if both scores are already filled
        if pd.notna(game['home_score']) and pd.notna(game['away_score']):
            continue
            
        # Find matching game in team_stats (should have 2 rows per game)
        game_matches = team_stats[
            (team_stats['date'] == game['date']) &
            (team_stats['season_str'] == game['season']) &
            (
                ((team_stats['home_std'] == game['home_team']) & (team_stats['away_std'] == game['away_team'])) |
                ((team_stats['team_std'] == game['home_team']) | (team_stats['team_std'] == game['away_team']))
            )
        ]
        
        if len(game_matches) >= 2:  # Should have at least 2 rows (home and away team stats)
            # Find home team score and away team score
            home_team_row = game_matches[game_matches['team_std'] == game['home_team']]
            away_team_row = game_matches[game_matches['team_std'] == game['away_team']]
            
            if len(home_team_row) > 0 and len(away_team_row) > 0:
                home_score = home_team_row.iloc[0]['PTS']
                away_score = away_team_row.iloc[0]['PTS']
                
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
                    if scores_filled <= 10:  # Show first 10 matches
                        status = []
                        if home_updated: status.append("home")
                        if away_updated: status.append("away")
                        print(f"   ✅ {game['away_team']} @ {game['home_team']} ({game['date']}) -> filled {'/'.join(status)} score(s)")
        
        elif len(game_matches) == 1:  # Sometimes might only find one team's stats
            team_row = game_matches.iloc[0]
            team_score = team_row['PTS']
            
            if team_row['team_std'] == game['home_team'] and pd.isna(df.at[idx, 'home_score']) and pd.notna(team_score):
                df.at[idx, 'home_score'] = int(team_score)
                scores_filled += 0.5  # Partial fill
                if scores_filled <= 10:
                    print(f"   🔄 {game['away_team']} @ {game['home_team']} ({game['date']}) -> filled home score only")
                    
            elif team_row['team_std'] == game['away_team'] and pd.isna(df.at[idx, 'away_score']) and pd.notna(team_score):
                df.at[idx, 'away_score'] = int(team_score)
                scores_filled += 0.5  # Partial fill
                if scores_filled <= 10:
                    print(f"   🔄 {game['away_team']} @ {game['home_team']} ({game['date']}) -> filled away score only")
    
    print(f"\n📊 Scores filled: {int(scores_filled)} games")
    
    # Manual odds entries for July 30th, 2020
    print(f"\n📝 Adding manual odds for July 30th, 2020 games...")
    
    july_30_games = [
        {
            'date': '2020-07-30',
            'home_team': 'New Orleans Pelicans',
            'away_team': 'Utah Jazz',
            'home_spread': -2.0,
            'away_spread': 2.0,
            'home_money_line': -130.0,
            'away_money_line': 110.0
        },
        {
            'date': '2020-07-30', 
            'home_team': 'Los Angeles Lakers',
            'away_team': 'Los Angeles Clippers',
            'home_spread': -5.0,
            'away_spread': 5.0,
            'home_money_line': -200.0,
            'away_money_line': 175.0
        }
    ]
    
    odds_added = 0
    for manual_game in july_30_games:
        # Find matching game in dataset
        matching_games = df[
            (df['date'] == manual_game['date']) &
            (df['home_team'] == manual_game['home_team']) &
            (df['away_team'] == manual_game['away_team'])
        ]
        
        if len(matching_games) > 0:
            idx = matching_games.index[0]
            
            # Update odds if they're missing
            updated = False
            if pd.isna(df.at[idx, 'home_money_line']):
                df.at[idx, 'home_money_line'] = manual_game['home_money_line']
                df.at[idx, 'away_money_line'] = manual_game['away_money_line']
                df.at[idx, 'home_spread'] = manual_game['home_spread']
                df.at[idx, 'away_spread'] = manual_game['away_spread']
                updated = True
            
            if updated:
                odds_added += 1
                print(f"   ✅ Added odds for {manual_game['away_team']} @ {manual_game['home_team']}")
        else:
            print(f"   ⚠️  Could not find game: {manual_game['away_team']} @ {manual_game['home_team']} on {manual_game['date']}")
    
    print(f"\n📊 Manual odds added: {odds_added} games")
    
    # Save updated dataset (OVERWRITE original file)
    output_filename = 'nba_complete_dataset_20152025_final.csv'
    df.to_csv(output_filename, index=False)
    print(f"\n💾 Updated dataset saved to: {output_filename}")
    
    # Summary
    print(f"\n✅ SUMMARY:")
    print(f"   Scores filled: {int(scores_filled)} games")
    print(f"   Manual odds added: {odds_added} games")
    print(f"   Output file: {output_filename}")
    
    # Check remaining missing scores by season
    remaining_missing = df[
        (df['season'].isin(target_seasons)) & 
        ((df['home_score'].isna()) | (df['away_score'].isna()))
    ]
    
    if len(remaining_missing) > 0:
        print(f"\n⚠️  Still missing scores for {len(remaining_missing)} games:")
        remaining_by_season = remaining_missing['season'].value_counts().sort_index()
        for season, count in remaining_by_season.items():
            print(f"   {season}: {count} games")
        
        # Show a few examples
        print(f"\nFirst 5 games still missing scores:")
        print(remaining_missing[['date', 'home_team', 'away_team', 'season']].head().to_string(index=False))
    else:
        print(f"\n🎉 All scores in target seasons now complete!")
    
    return df

def main():
    """Main function"""
    try:
        # Check if required files exist
        required_files = [
            'nba_complete_dataset_20152025_final.csv',
            'team_traditional.csv'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print("❌ Missing required files:")
            for file in missing_files:
                print(f"   - {file}")
            return
        
        # Process the data
        updated_df = fill_missing_scores_from_team_traditional()
        
        if updated_df is not None:
            print(f"\n✅ SUCCESS! Score filling complete.")
            
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()