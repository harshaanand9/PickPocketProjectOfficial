import stats_getter
from nba_api.stats.endpoints import teamyearbyyearstats, leaguedashteamshotlocations, leaguedashteamstats, teamdashptshots, leaguedashteamstats
from nba_api.stats.endpoints import teamplayerdashboard, playergamelogs, leaguedashplayerstats, commonteamroster, boxscoretraditionalv2, teamgamelog, leaguedashteamptshot, TeamDashPtPass, LeagueDashOppPtShot, LeagueDashTeamStats, LeagueHustleStatsTeam, LeagueDashPtDefend, LeagueDashTeamClutch
from nba_api.stats.static import teams
import importlib
import odds_loader
import stats_getter
importlib.reload(odds_loader)
importlib.reload(stats_getter)
from odds_loader import getOutcome, getCSV

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import teamplayerdashboard
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import norm


if __name__ == "__main__":
    # Example: first 30 days of 2015-16 (fast sanity check)
    df0 = get_season_games("2015-16").copy()
    df0["date"] = pd.to_datetime(df0["date"], format="%m/%d/%Y", errors="coerce")
    start_str = df0["date"].min().strftime("%m/%d/%Y")
    end_str   = (df0["date"].min() + pd.Timedelta(days=30)).strftime("%m/%d/%Y")

    benchmark_games("2015-16", start=start_str, end=end_str, quiet=False, limit=None)


def get_team_stats_cache(team_name, season):
    """Get all team stats once and cache them"""
    try:
        team_id = stats_getter.get_team_id(team_name)
        
        # Get base stats (GP, MIN)
        base_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            team_id_nullable=team_id,
            season=season,
            season_type_all_star='Regular Season',
            measure_type_detailed_defense="Base", 
            per_mode_detailed="PerGame"
        ).get_data_frames()[0]
        
        # Get advanced stats (PIE)  
        advanced_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            team_id_nullable=team_id,
            season=season,
            season_type_all_star='Regular Season',
            measure_type_detailed_defense="Advanced", 
            per_mode_detailed="PerGame"
        ).get_data_frames()[0]
        
        return base_stats, advanced_stats
        
    except Exception as e:
        print(f"Error getting team stats for {team_name} {season}: {e}")
        return None, None

def get_player_stat_from_cache(player_name, stat, base_df, advanced_df):
    """Get player stat from cached dataframes"""
    try:
        if stat in ["GP", "MIN"]:
            df = base_df
        else:  # PIE, USG_PCT
            df = advanced_df
            
        player_data = df[df['PLAYER_NAME'] == player_name]
        
        if player_data.empty:
            return 0
            
        return player_data[stat].iloc[0]
        
    except Exception as e:
        print(f"Error getting {stat} for {player_name}: {e}")
        return 0

def analyze_roster_changes(team_name, curr_season, prev_season, curr_date):
    print(f"Starting analysis for {team_name}...")
    
    # Get rosters
    curr_roster = stats_getter.getRoster(team_name, curr_season)
    prev_roster = stats_getter.getRoster(team_name, prev_season)
    
    print(f"Current roster: {len(curr_roster)} players")
    print(f"Previous roster: {len(prev_roster)} players")
    
    # Get TEAM stats for lost players (they were on this team)
    print("Getting Lakers stats for lost players...")
    team_base, team_advanced = get_team_stats_cache(team_name, prev_season)
    
    # Get LEAGUE stats for added players (they were on other teams)
    print("Getting league-wide stats for added players...")
    league_base, league_advanced = get_league_stats_cache(prev_season)
    
    if team_base is None or league_base is None:
        print("Failed to get stats")
        return None
    
    # Compare rosters
    lost_players = set(prev_roster) - set(curr_roster)
    added_players = set(curr_roster) - set(prev_roster)
    
    result = {}
    
    # Check lost players using TEAM stats (they were on the Lakers)
    print(f"Checking {len(lost_players)} lost players...")
    for player in lost_players:
        gp = get_player_stat_from_cache(player, "GP", team_base, team_advanced)
        mins = get_player_stat_from_cache(player, "MIN", team_base, team_advanced)
        
        if gp >= 30 and mins >= 14:
            result[player] = 0
            print(f"Lost player {player}: {gp} GP, {mins} MIN - QUALIFIED")
        else:
            print(f"Lost player {player}: {gp} GP, {mins} MIN - not qualified")
    
    # Check added players using LEAGUE stats (they were on other teams)
    print(f"Checking {len(added_players)} added players...")
    for player in added_players:
        gp = get_player_stat_from_cache(player, "GP", league_base, league_advanced)
        mins = get_player_stat_from_cache(player, "MIN", league_base, league_advanced)
        
        if gp >= 30 and mins >= 14:
            result[player] = 1
            print(f"Added player {player}: {gp} GP, {mins} MIN - QUALIFIED")
        else:
            print(f"Added player {player}: {gp} GP, {mins} MIN - not qualified")
    
    # Calculate PIE
    lost_pie = 0.0
    added_pie = 0.0
    
    print("\nCalculating PIE...")
    
    for player, status in result.items():
        if status == 0:  # Lost player - use TEAM stats
            player_pie = get_player_stat_from_cache(player, "PIE", team_base, team_advanced)
            lost_pie += player_pie
            print(f"Lost player {player}: PIE = {player_pie}")
            
        elif status == 1:  # Added player - use LEAGUE stats
            player_pie = get_player_stat_from_cache(player, "PIE", league_base, league_advanced)
            added_pie += player_pie
            print(f"Added player {player}: PIE = {player_pie}")
    
    print(f"\nTotal lost PIE: {lost_pie}")
    print(f"Total added PIE: {added_pie}")
    print(f"Net PIE change: {added_pie - lost_pie}")
    
    return {
        'team': team_name,
        'roster_changes': result,
        'lost_players': [p for p, s in result.items() if s == 0],
        'added_players': [p for p, s in result.items() if s == 1], 
        'lost_pie': lost_pie,
        'added_pie': added_pie,
        'net_pie_change': added_pie - lost_pie
    }

def get_league_stats_cache(season):
    """Get league-wide stats for all players"""
    try:
        print("Getting league-wide base stats...")
        base_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star='Regular Season',
            measure_type_detailed_defense="Base", 
            per_mode_detailed="PerGame"
        ).get_data_frames()[0]
        
        print("Getting league-wide advanced stats...")
        advanced_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star='Regular Season',
            measure_type_detailed_defense="Advanced", 
            per_mode_detailed="PerGame"
        ).get_data_frames()[0]
        
        return base_stats, advanced_stats
        
    except Exception as e:
        print(f"Error getting league stats for {season}: {e}")
        return None, None



team_name = "Los Angeles Lakers"
player_name = "Anthony Davis"
curr_season = "2022-23"
prev_season = "2021-22"
curr_date = "02/11/2023"
team_id = stats_getter.get_team_id(team_name)

#print(get_desired_game_roster(team_name, curr_date, curr_season))
print('\n')
#print(getRoster(team_name, prev_season))
print('\n')
print(analyze_roster_changes(team_name, curr_season, prev_season, curr_date))

# 1) Query opponent stats for the season you care about

#resp = LeagueDashTeamStats(season=curr_season,team_id_nullable=team_id,per_mode_detailed="PerGame",measure_type_detailed_defense="Base")
#df_list = resp.get_data_frames()
#print(df_list)



