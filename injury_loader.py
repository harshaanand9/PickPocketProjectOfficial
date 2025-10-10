import requests
import pandas as pd
import pdfplumber
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import time
import json
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NBAInjuryReportScraper:
    def __init__(self, cache_dir: str = "pdf_cache"):
        self.base_url = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date}_{time}.pdf"
        self.cache_dir = cache_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        os.makedirs(cache_dir, exist_ok=True)
        self.all_data = []
    
    def format_player_name(self, name: str) -> str:
        """Convert 'Last,First' to 'First Last' format"""
        if ',' in name:
            parts = name.split(',')
            if len(parts) == 2:
                last_name = parts[0].strip()
                first_name = parts[1].strip()
                return f"{first_name} {last_name}"
        return name  # Return as-is if not in expected format
    
    def debug_game_info(self, pdf_content: bytes):
        """Helper function to debug game information extraction"""
        try:
            import io
            pdf_file = io.BytesIO(pdf_content)
            with pdfplumber.open(pdf_file) as pdf:
                print("\n=== DEBUGGING GAME INFO EXTRACTION ===\n")
                
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        lines = text.split('\n')
                        print(f"\n--- PAGE {page_num + 1} ---")
                        
                        for line_num, line in enumerate(lines):
                            # Check if line contains date pattern
                            has_date = re.search(r'\d{1,2}/\d{1,2}/\d{4}', line)
                            # Check if line contains time pattern
                            has_time = re.search(r'\d{1,2}:\d{2}', line)
                            # Check if line contains matchup pattern
                            has_matchup = re.search(r'[A-Z]{3}@[A-Z]{3}', line)
                            # Check if line contains "ET"
                            has_et = 'ET' in line
                            
                            # Print lines that have any game-related info
                            if has_date or has_time or has_matchup or (has_et and has_time):
                                print(f"Line {line_num}: {line.strip()}")
                                if has_date:
                                    print(f"  → Date found: {has_date.group()}")
                                if has_time:
                                    print(f"  → Time found: {has_time.group()}")
                                if has_matchup:
                                    print(f"  → Matchup found: {has_matchup.group()}")
                                print()
                
                print("=== END DEBUG ===\n")
                
        except Exception as e:
            print(f"Error in debug: {e}")
            import traceback
            traceback.print_exc()
    
    def download_pdf(self, url: str, date: str, time_slot: str) -> Optional[bytes]:
        cache_filename = f"{self.cache_dir}/injury_report_{date}_{time_slot}.pdf"
        
        if os.path.exists(cache_filename):
            logger.info(f"Loading from cache: {cache_filename}")
            with open(cache_filename, 'rb') as f:
                return f.read()
        
        try:
            logger.info(f"Downloading: {url}")
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                with open(cache_filename, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Successfully downloaded and cached: {cache_filename}")
                return response.content
            else:
                logger.warning(f"HTTP {response.status_code} for {url}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading {url}: {e}")
            return None
    
    def parse_2023_format(self, pdf_content: bytes, report_date: str, time_slot: str) -> List[Dict]:
        try:
            import io
            pdf_file = io.BytesIO(pdf_content)
            with pdfplumber.open(pdf_file) as pdf:
                # Get all text and process line by line to find patterns
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                
                if not full_text:
                    return []
                
                # Debug: Search for Rice in the entire text
                if "Rice" in full_text:
                    logger.info("Found 'Rice' in PDF text!")
                    # Find all occurrences
                    for idx, line in enumerate(full_text.split('\n')):
                        if "Rice" in line:
                            logger.info(f"Rice found at line {idx}: {repr(line)}")
                else:
                    logger.warning("'Rice' NOT found anywhere in the PDF text!")
                
                # Find all injury patterns in the text using a comprehensive approach
                all_injuries = []
                
                # Split into lines for context
                lines = full_text.split('\n')
                
                # Team mapping
                team_map = {
                    'BostonCeltics': 'Boston Celtics',
                    'IndianaPacers': 'Indiana Pacers',
                    'WashingtonWizards': 'Washington Wizards',
                    'OrlandoMagic': 'Orlando Magic',
                    'MemphisGrizzlies': 'Memphis Grizzlies',
                    'DallasMavericks': 'Dallas Mavericks',
                    'NewYorkKnicks': 'New York Knicks',
                    'TorontoRaptors': 'Toronto Raptors',
                    'Philadelphia76ers': 'Philadelphia 76ers',
                    'SanAntonioSpurs': 'San Antonio Spurs',
                    'NewOrleansPelicans': 'New Orleans Pelicans',
                    'DenverNuggets': 'Denver Nuggets',
                    'PhoenixSuns': 'Phoenix Suns',
                    'GoldenStateWarriors': 'Golden State Warriors',
                    'LAClippers': 'LA Clippers',
                    'MinnesotaTimberwolves': 'Minnesota Timberwolves',
                    'CharlotteHornets': 'Charlotte Hornets',
                    'ClevelandCavaliers': 'Cleveland Cavaliers',
                    'DetroitPistons': 'Detroit Pistons',
                    'BrooklynNets': 'Brooklyn Nets',
                    'AtlantaHawks': 'Atlanta Hawks',
                    'MilwaukeeBucks': 'Milwaukee Bucks',
                    'MiamiHeat': 'Miami Heat',
                    'ChicagoBulls': 'Chicago Bulls',
                    'OklahomaCityThunder': 'Oklahoma City Thunder',
                    'PortlandTrailBlazers': 'Portland Trail Blazers',
                    'UtahJazz': 'Utah Jazz',
                    'SacramentoKings': 'Sacramento Kings',
                    'HoustonRockets': 'Houston Rockets',
                    'LosAngelesLakers': 'Los Angeles Lakers'
                }
                
                # FIXED: Track game information throughout the document
                current_game_date = None
                current_game_time = None
                current_matchup = None
                
                # FIXED: Track the current team throughout the entire document
                current_team = "Unknown Team"
                
                # Now find all player patterns in the entire text
                for line_idx, line in enumerate(lines):
                    # First check if this line contains game information
                    # Pattern 1: Full pattern with date
                    full_game_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})\s+(\d{1,2}:\d{2}\s*\(ET\))\s+([A-Z]{3}@[A-Z]{3})', line)
                    if full_game_match:
                        current_game_date = full_game_match.group(1)
                        current_game_time = full_game_match.group(2)
                        current_matchup = full_game_match.group(3)
                        logger.info(f"Line {line_idx}: Found full game info: {current_matchup} on {current_game_date} at {current_game_time}")
                    else:
                        # Pattern 2: Time and matchup without date (common for same-day games)
                        time_matchup_match = re.search(r'(\d{1,2}:\d{2}\s*\(ET\))\s+([A-Z]{3}@[A-Z]{3})', line)
                        if time_matchup_match:
                            current_game_time = time_matchup_match.group(1)
                            current_matchup = time_matchup_match.group(2)
                            # Always use report date for games without explicit dates
                            # Convert report date format (2023-12-01) to game date format (12/01/2023)
                            date_parts = report_date.split('-')
                            current_game_date = f"{date_parts[1]}/{date_parts[2]}/{date_parts[0]}"
                            logger.info(f"Line {line_idx}: Found time+matchup: {current_matchup} at {current_game_time} (date: {current_game_date})")
                        else:
                            # Pattern 3: Just matchup (like NYK@TOR)
                            just_matchup_match = re.search(r'([A-Z]{3}@[A-Z]{3})', line)
                            if just_matchup_match:
                                # Only update if this appears to be a new game line
                                # Check if the matchup is at/near the start of the line and followed by a team
                                line_start = line.strip()
                                if line_start.startswith(just_matchup_match.group(1)) or line_start.split()[0] == just_matchup_match.group(1):
                                    for team_key in team_map.keys():
                                        if team_key in line:
                                            current_matchup = just_matchup_match.group(1)
                                            # Keep existing game date/time or use report date if none
                                            if not current_game_date:
                                                date_parts = report_date.split('-')
                                                current_game_date = f"{date_parts[1]}/{date_parts[2]}/{date_parts[0]}"
                                            logger.info(f"Line {line_idx}: Found matchup only: {current_matchup}")
                                            break
                    
                    # Check if this line contains "NOT YET SUBMITTED"
                    if "NOTYETSUBMITTED" in line:
                        # Extract team name from the line
                        for team_key, team_name in team_map.items():
                            if team_key in line:
                                logger.info(f"Found NOT YET SUBMITTED for team: {team_name}")
                                all_injuries.append({
                                    'player_name': "NOT YET SUBMITTED",
                                    'team': team_name,
                                    'status': "NOT YET SUBMITTED",
                                    'reason': "NOT YET SUBMITTED",
                                    'game_date': current_game_date,
                                    'game_time': current_game_time,
                                    'matchup': current_matchup,
                                    'report_date': report_date,
                                    'report_time': time_slot
                                })
                                break
                        continue  # Skip to next line
                    
                    # First check if this line contains a team name
                    for team_key, team_name in team_map.items():
                        if team_key in line:
                            current_team = team_name
                            logger.debug(f"Found team: {current_team} at line {line_idx}")
                            break
                    
                    # Then find players in this line
                    player_matches = re.finditer(r"([A-Za-z\-\.']+,[A-Za-z\-\.']+)\s+(Out|Available|Questionable|Probable|Doubtful)", line)
                    
                    for match in player_matches:
                        player_name_raw = match.group(1)
                        status = match.group(2)
                        
                        # Format the player name from "Last,First" to "First Last"
                        player_name = self.format_player_name(player_name_raw)
                        
                        # Use the current team instead of looking backwards
                        assigned_team = current_team
                        
                        # Get reason from the current line after the status
                        reason_start = match.end()
                        reason = line[reason_start:].strip()
                        
                        # Look for additional injury context in nearby lines
                        context_lines = []
                        for i in range(max(0, line_idx - 3), min(len(lines), line_idx + 4)):
                            if ('Injury/Illness' in lines[i] or 
                                any(term in lines[i].lower() for term in ['contusion', 'sprain', 'strain', 'fracture', 'infection', 'bruise', 'respiratory', 'bone', 'meniscus'])):
                                context_lines.append(lines[i].strip())
                        
                        # Combine context with reason
                        if context_lines and not reason:
                            reason = ' '.join(context_lines)
                        elif context_lines:
                            reason = ' '.join(context_lines) + ' ' + reason
                        
                        all_injuries.append({
                            'player_name': player_name,
                            'team': assigned_team,
                            'status': status,
                            'reason': reason.strip(),
                            'game_date': current_game_date,
                            'game_time': current_game_time,
                            'matchup': current_matchup,
                            'report_date': report_date,
                            'report_time': time_slot
                        })
                        
                        logger.info(f"Found: {player_name} ({assigned_team}) - {status}: {reason}")
                
                # Return all injuries as a single list wrapped in a game structure
                # This maintains compatibility with the existing export format
                if all_injuries:
                    # We'll use a dummy game structure since injuries already contain game info
                    game = {
                        'game_date': 'Multiple',
                        'game_time': 'Multiple',
                        'matchup': 'Multiple',
                        'report_date': report_date,
                        'report_time': time_slot,
                        'injuries': all_injuries
                    }
                    return [game]
                
                return []
                
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def scrape_date(self, date: str) -> List[Dict]:
        """Scrape a single date"""
        for time_slot in ["05PM", "08PM"]:
            url = self.base_url.format(date=date, time=time_slot)
            pdf_content = self.download_pdf(url, date, time_slot)
            
            if pdf_content:
                year = int(date.split('-')[0])
                if year >= 2023:
                    games = self.parse_2023_format(pdf_content, date, time_slot)
                    if games:
                        return games
        
        return []
    
    def scrape_date_range(self, start_date: str, end_date: str):
        """Scrape a date range"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            games = self.scrape_date(date_str)
            self.all_data.extend(games)
            current += timedelta(days=1)
    
    def export_to_csv(self, filename: str):
        if not self.all_data:
            logger.warning("No data to export")
            return
        
        # Flatten the data for CSV export
        flattened_data = []
        
        for game in self.all_data:
            for injury in game.get('injuries', []):
                row = {
                    'report_date': game.get('report_date'),
                    'report_time': game.get('report_time'),
                    'game_date': injury.get('game_date'),  # Use injury's game info
                    'game_time': injury.get('game_time'),  # Use injury's game info
                    'matchup': injury.get('matchup'),      # Use injury's game info
                    'player_name': injury.get('player_name'),
                    'team': injury.get('team'),
                    'status': injury.get('status')
                    # Removed 'reason' column as requested
                }
                flattened_data.append(row)
        
        df = pd.DataFrame(flattened_data)
        df.to_csv(filename, index=False)
        logger.info(f"Exported {len(flattened_data)} injury records to {filename}")

# Usage
if __name__ == "__main__":
    scraper = NBAInjuryReportScraper()
    
    # Test the specific URL you provided
    print("=== Testing 2023-12-01 08PM ===")
    pdf_content = scraper.download_pdf(
        "https://ak-static.cms.nba.com/referee/injury/Injury-Report_2023-12-01_08PM.pdf",
        "2023-12-01", "08PM"
    )
    
    if pdf_content:
        # First, debug to see game info structure
        scraper.debug_game_info(pdf_content)
        
        # Then run normal parsing
        games = scraper.parse_2023_format(pdf_content, "2023-12-01", "08PM")
        
        if games:
            for game in games:
                print(f"\nGame: {game['game_date']} {game['matchup']}")
                print(f"Injuries found: {len(game['injuries'])}")
                
                # Group by team
                by_team = {}
                for injury in game['injuries']:
                    team = injury['team']
                    if team not in by_team:
                        by_team[team] = []
                    by_team[team].append(injury)
                
                for team, injuries in by_team.items():
                    print(f"\n  {team}:")
                    for injury in injuries:
                        print(f"    - {injury['player_name']} - {injury['status']}: {injury['reason'][:100]}...")
        else:
            print("No games found")
    else:
        print("Failed to download PDF")
    
    # Also test the original date for comparison
    print("\n" + "="*50)
    print("=== Testing 2023-12-03 05PM (original) ===")
    games_original = scraper.scrape_date("2023-12-03")
    
    if games_original:
        for game in games_original:
            print(f"\nGame: {game['game_date']} {game['matchup']}")
            print(f"Injuries found: {len(game['injuries'])}")
            
            # Just list player names for quick comparison
            players = [injury['player_name'] for injury in game['injuries']]
            print(f"Players: {', '.join(players)}")
    
    # Export both if we have data
    all_games = []
    if games:
        all_games.extend(games)
    if games_original:
        all_games.extend(games_original)
    
    if all_games:
        scraper.all_data = all_games
        scraper.export_to_csv("nba_injuries_test.csv")
        print(f"\nExported {sum(len(g['injuries']) for g in all_games)} total injuries to CSV")