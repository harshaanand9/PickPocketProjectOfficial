import pandas as pd
import os

def add_july_30_bubble_games():
    """
    Add the two July 30th, 2020 bubble games with scores and odds
    """
    
    print("🏀 Adding July 30th, 2020 Bubble Games")
    print("=" * 50)
    
    # Load the current dataset
    try:
        df = pd.read_csv('nba_complete_dataset_20152025_updated.csv')
        print(f"✅ Loaded current dataset: {len(df):,} games")
    except FileNotFoundError:
        try:
            df = pd.read_csv('nba_complete_dataset_20152025.csv')
            print(f"✅ Loaded original dataset: {len(df):,} games")
        except FileNotFoundError:
            print("❌ Error: No NBA dataset found")
            return None
    
    # Define the two July 30th, 2020 games
    july_30_games = [
        {
            'date': '2020-07-30',
            'home_team': 'New Orleans Pelicans',
            'away_team': 'Utah Jazz',
            'home_money_line': -130.0,
            'home_spread': -2.0,
            'away_money_line': 110.0,
            'away_spread': 2.0,
            'home_score': 104,  # Pelicans score
            'away_score': 106,  # Jazz score (Jazz won)
            'season': '2019-20'
        },
        {
            'date': '2020-07-30',
            'home_team': 'Los Angeles Lakers',
            'away_team': 'Los Angeles Clippers',
            'home_money_line': -200.0,
            'home_spread': -5.0,
            'away_money_line': 175.0,
            'away_spread': 5.0,
            'home_score': 103,  # Lakers score (Lakers won)
            'away_score': 101,  # Clippers score
            'season': '2019-20'
        }
    ]
    
    print(f"\n📝 Adding July 30th, 2020 bubble games...")
    
    games_added = 0
    games_updated = 0
    
    for new_game in july_30_games:
        # Check if this game already exists
        existing_game_mask = (
            (df['date'] == new_game['date']) &
            (df['home_team'] == new_game['home_team']) &
            (df['away_team'] == new_game['away_team'])
        )
        existing_games = df[existing_game_mask]
        
        if len(existing_games) == 0:
            # Game doesn't exist, add it as a new row
            new_row = pd.DataFrame([new_game])
            df = pd.concat([df, new_row], ignore_index=True)
            games_added += 1
            print(f"   ✅ Added: {new_game['away_team']} @ {new_game['home_team']} -> {new_game['away_score']}-{new_game['home_score']}")
            
        else:
            # Game exists, update missing data
            idx = existing_games.index[0]
            updated_fields = []
            
            # Update missing scores
            if pd.isna(df.at[idx, 'home_score']):
                df.at[idx, 'home_score'] = new_game['home_score']
                updated_fields.append('home_score')
            if pd.isna(df.at[idx, 'away_score']):
                df.at[idx, 'away_score'] = new_game['away_score']
                updated_fields.append('away_score')
                
            # Update missing odds
            if pd.isna(df.at[idx, 'home_money_line']):
                df.at[idx, 'home_money_line'] = new_game['home_money_line']
                updated_fields.append('home_money_line')
            if pd.isna(df.at[idx, 'away_money_line']):
                df.at[idx, 'away_money_line'] = new_game['away_money_line']
                updated_fields.append('away_money_line')
            if pd.isna(df.at[idx, 'home_spread']):
                df.at[idx, 'home_spread'] = new_game['home_spread']
                updated_fields.append('home_spread')
            if pd.isna(df.at[idx, 'away_spread']):
                df.at[idx, 'away_spread'] = new_game['away_spread']
                updated_fields.append('away_spread')
            
            if updated_fields:
                games_updated += 1
                print(f"   🔄 Updated: {new_game['away_team']} @ {new_game['home_team']} -> {', '.join(updated_fields)}")
            else:
                print(f"   ℹ️  Complete: {new_game['away_team']} @ {new_game['home_team']} (no updates needed)")
    
    # Sort by date to keep chronological order
    df = df.sort_values('date').reset_index(drop=True)
    
    # Save updated dataset
    output_filename = 'nba_complete_dataset_20152025_final.csv'
    df.to_csv(output_filename, index=False)
    
    print(f"\n📊 SUMMARY:")
    print(f"   Games added: {games_added}")
    print(f"   Games updated: {games_updated}")
    print(f"   Total games in dataset: {len(df):,}")
    print(f"   Output file: {output_filename}")
    
    # Verify the July 30th games are in the dataset
    july_30_check = df[df['date'] == '2020-07-30']
    if len(july_30_check) > 0:
        print(f"\n✅ Verification - July 30th, 2020 games in dataset:")
        print(july_30_check[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'home_money_line']].to_string(index=False))
    else:
        print(f"\n⚠️  No July 30th, 2020 games found in final dataset")
    
    print(f"\n💾 Final dataset saved as: {output_filename}")
    
    return df

def main():
    """Main function to run the July 30th game addition"""
    try:
        updated_df = add_july_30_bubble_games()
        
        if updated_df is not None:
            print(f"\n🎉 SUCCESS! July 30th bubble games processing complete.")
            
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()