import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.optim as optim
import pandas as pd
import argparse
import json
import csv
import requests
from datetime import datetime, timedelta
import stats_getter
import importlib
import os
from zoneinfo import ZoneInfo
import platform

importlib.reload(stats_getter)
from stats_getter import getPointMargin, getPlayersInGame, getGameRotation

def load_games2(api_key, csv_filename='draft_kings_nba_odds.csv', bookmakers=None):
    """
    Loads historical NBA game odds from the API and appends game data to an existing CSV file.
    Prioritizes DraftKings odds, falling back to other bookmakers if DraftKings odds aren't available.
    
    - The game's commence time (from the API, originally in UTC) is converted to PST.
    - The CSV's 'date' column reflects the game date in PST (formatted as YYYY-MM-DD).
    - The 'api_call_date' column is a full PST timestamp.
    
    If the CSV does not exist, it is created with a header row; otherwise, new rows are appended.
    """
    SPORT = 'basketball_nba'
    REGIONS = 'us'
    MARKETS = 'spreads'
    ODDS_FORMAT = 'american'
    DATE_FORMAT = 'iso'

    # Define the start and end dates for the 2020-2021 NBA season (in UTC)
    start_date = datetime.strptime("2020-12-22T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ")  # Season start
    end_date   = datetime.strptime("2021-05-16T23:59:59Z", "%Y-%m-%dT%H:%M:%SZ")  # Regular season end
    
    # CSV headers
    headers = [
        'date',            # Game date in PST (YYYY-MM-DD)
        'game_id',
        'home_team',
        'away_team',
        'home_money_line',
        'home_spread',
        'away_money_line',
        'away_spread',
        'api_call_date',   # API call timestamp in PST
        'bookmaker'        # Added to track which bookmaker provided the odds
    ]
    
    # Ensure the output folder exists
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    
    file_exists = os.path.exists(csv_filename)
    with open(csv_filename, mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)

        current_date = start_date
        while current_date <= end_date:
            # Format the date of the API call in UTC for the request
            date_str = current_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            print(f"Processing date: {date_str}")

            # First try with DraftKings
            params = {
                'api_key': api_key,
                'regions': REGIONS,
                'markets': MARKETS,
                'oddsFormat': ODDS_FORMAT,
                'dateFormat': DATE_FORMAT,
                'date': date_str,
                'bookmakers': 'draftkings'
            }

            response = requests.get(
                f'https://api.the-odds-api.com/v4/historical/sports/{SPORT}/odds',
                params=params
            )
            
            if response.status_code != 200:
                print(f"Failed to get odds for {date_str}: {response.status_code}, {response.text}")
                current_date += timedelta(days=1)
                continue

            odds_json = response.json()
            if 'data' not in odds_json:
                current_date += timedelta(days=1)
                continue

            # Process each game
            for game in odds_json['data']:
                game_id = game.get('id')
                commence_time_utc_str = game.get('commence_time')

                try:
                    # Parse the commence time as UTC and make it timezone-aware
                    game_commence_dt_utc = datetime.strptime(commence_time_utc_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=ZoneInfo("UTC"))
                    # Parse the API call time as UTC
                    api_dt_utc = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=ZoneInfo("UTC"))
                    # Convert both to PST
                    game_commence_dt_pst = game_commence_dt_utc.astimezone(ZoneInfo("America/Los_Angeles"))
                    api_dt_pst = api_dt_utc.astimezone(ZoneInfo("America/Los_Angeles"))
                except Exception as e:
                    print(f"Error parsing date for game {game_id}: {e}")
                    continue
                
                # Skip if the API call (in PST) happens after the game has started
                if api_dt_pst > game_commence_dt_pst:
                    continue
                
                home_team = game.get('home_team')
                away_team = game.get('away_team')
                if not away_team and 'teams' in game:
                    teams = game['teams']
                    if len(teams) > 1:
                        away_team = teams[1] if teams[0] == home_team else teams[0]

                home_money_line = ''
                away_money_line = ''
                home_spread = ''
                away_spread = ''
                used_bookmaker = 'draftkings'

                # Try to get odds from DraftKings first
                draftkings_odds_found = False
                for bookmaker in game.get('bookmakers', []):
                    if bookmaker.get('key') == 'draftkings':
                        for market in bookmaker.get('markets', []):
                            outcomes = market.get('outcomes', [])
                            for outcome in outcomes:
                                if outcome.get('name') == home_team:
                                    home_money_line = outcome.get('price')
                                    home_spread = outcome.get('point')
                                elif outcome.get('name') == away_team:
                                    away_money_line = outcome.get('price')
                                    away_spread = outcome.get('point')
                            if home_money_line and away_money_line and home_spread and away_spread:
                                draftkings_odds_found = True
                                break
                        if draftkings_odds_found:
                            break

                # If DraftKings odds not found, try other bookmakers
                if not draftkings_odds_found:
                    # Make another API call without bookmaker restriction
                    params['bookmakers'] = None
                    alt_response = requests.get(
                        f'https://api.the-odds-api.com/v4/historical/sports/{SPORT}/odds',
                        params=params
                    )
                    
                    if alt_response.status_code == 200:
                        alt_json = alt_response.json()
                        for alt_game in alt_json.get('data', []):
                            if alt_game.get('id') == game_id:
                                for bookmaker in alt_game.get('bookmakers', []):
                                    for market in bookmaker.get('markets', []):
                                        outcomes = market.get('outcomes', [])
                                        for outcome in outcomes:
                                            if outcome.get('name') == home_team:
                                                home_money_line = outcome.get('price')
                                                home_spread = outcome.get('point')
                                            elif outcome.get('name') == away_team:
                                                away_money_line = outcome.get('price')
                                                away_spread = outcome.get('point')
                                        if home_money_line and away_money_line and home_spread and away_spread:
                                            used_bookmaker = bookmaker.get('key', 'unknown')
                                            break
                                    if home_money_line and away_money_line and home_spread and away_spread:
                                        break
                                break

                # Only write to CSV if we found odds from any bookmaker
                if home_money_line and away_money_line and home_spread and away_spread:
                    # Format PST times for output
                    pst_game_date_str = game_commence_dt_pst.strftime("%Y-%m-%d")
                    pst_api_call_str   = api_dt_pst.strftime("%Y-%m-%dT%H:%M:%S %Z")
                    
                    writer.writerow([
                        pst_game_date_str,
                        game_id,
                        home_team,
                        away_team,
                        home_money_line,
                        home_spread,
                        away_money_line,
                        away_spread,
                        pst_api_call_str,
                        used_bookmaker
                    ])
            
            current_date += timedelta(days=1)
    
    # Sort the CSV by date
    df = pd.read_csv(csv_filename)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    df.to_csv(csv_filename, index=False)
    print(f"CSV sorted by date and saved as: {csv_filename}")

def getCSV(csv_filename='draft_kings_nba_odds.csv'):
    """
    Loads the CSV file and returns a Pandas DataFrame sorted by the 'date' column.
    If the file doesn't exist, an error message is printed and None is returned.
    
    Parameters:
        csv_filename (str): The path to the CSV file.
    
    Returns:
        pd.DataFrame or None: The DataFrame containing the CSV data sorted by date,
                              or None if the CSV file does not exist.
    """
    if not os.path.exists(csv_filename):
        print(f"CSV file {csv_filename} does not exist.")
        return None

    try:
        df = pd.read_csv(csv_filename)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

    if 'date' in df.columns:
        try:
            # Convert the 'date' column to datetime objects
            df['date'] = pd.to_datetime(df['date'])
            # Sort the DataFrame by the 'date' column
            df.sort_values(by='date', inplace=True)
        except Exception as e:
            print(f"Error processing the 'date' column: {e}")
    else:
        print("The 'date' column was not found in the CSV.")

    return df


def dateToISO(date):
    """
    Returns the ISO 8601 format of a given date.

    params:
    date (str): The desired date (formatted like "AUG 17, 2020")
    
    Returns:
    str: The ISO 8601 formatted date ("YYYY-MM-DD")
    """
    # Convert the month part from uppercase to title case (e.g., "AUG" -> "Aug")
    formatted_date = date.title()
    # Parse the date string using the format for abbreviated month, day, and full year.
    dt = datetime.strptime(formatted_date, "%b %d, %Y")
    # Return the date in ISO 8601 format
    return dt.date().isoformat()

def convert_date(input_date_str):
    # Ensure the month is in title-case (e.g. "AUG" -> "Aug")
    formatted_input = input_date_str.title()
    
    # Parse the input date string.
    # %b for abbreviated month name, %d for day, and %Y for four-digit year.
    dt = datetime.strptime(formatted_input, "%b %d, %Y")
    
    # Choose the proper strftime format depending on your OS
    # On UNIX/Linux/Mac, use %-m and %-d to avoid leading zeros.
    # On Windows, you may need to use %#m and %#d.
    if platform.system() == 'Windows':
        output_format = "%#m/%#d/%y"  # e.g., "8/17/20"
    else:
        output_format = "%-m/%-d/%y"  # e.g., "8/17/20"
    
    return dt.strftime(output_format)
    
def getOutcome(home_team, season, date):
    """
    Returns the outcome of a bet for a specefic game from the perspective of the home_team

    params:
    home_team: string of home team's name
    season: string of desired nba season (formatted like 2019-20)
    date (str): The desired date (formatted like "AUG 17, 2020")
    """
    new_date = convert_date(date)
    
    df = pd.read_csv('draft_kings_nba_odds.csv')
    margin = getPointMargin(home_team, season, date) # home_team - away_team
    margin_as_float = float(margin)

    
    matching_rows = df[df['date'] == new_date]
    for home_team_name in matching_rows['home_team']:
        if home_team_name == home_team:
            game_info = matching_rows[matching_rows['home_team'] == home_team_name]
            home_spread = game_info['home_spread'].iloc[0]
            return (margin_as_float+home_spread) > 0


def clear_csv_with_header(csv_filename='draft_kings_nba_odds.csv'):
    """
    Clears all contents of the CSV file and writes only the header row.
    
    Parameters:
        csv_filename (str): The name/path of the CSV file to clear.
    """
    headers = ['date', 'game_id', 'home_team', 'away_team', 'home_odds', 'away_odds', 'api_call_date']
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
    print(f"{csv_filename} has been reset with headers.")

def clearNA(df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes rows in the DataFrame where every entry is NaN.
        
        Parameters:
            df (pd.DataFrame): The DataFrame to clean.
            
        Returns:
            pd.DataFrame: The DataFrame after removing rows with all NaN values.
        """
        return df.dropna(how='all')

    
# Parse the API key from command line arguments (or use the default)
parser = argparse.ArgumentParser(description='Sample V4')
parser.add_argument('--api-key', type=str, default='165a18dc5ba981c08ff1ab127432d994')
args = parser.parse_args(args=[])
API_KEY = args.api_key

# Uncomment to run
#load_games2(API_KEY)

#clear_csv_with_header()
# Example usage after the CSV is created:
#df = pd.read_csv('PickPocketProject/draft_kings_nba_odds.csv')
#print(df['date'])

#clearNA(getCSV())

#getOutcome("Dallas Mavericks", "2019-20", "AUG 08, 2020")
#print(getPlayersInGame("Dallas Mavericks", "2019-20", "AUG 08, 2020"))




