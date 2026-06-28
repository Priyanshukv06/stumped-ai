import json
import os
import glob
import pandas as pd
import numpy as np
import requests
import zipfile
import pandas_gbq
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()

# --- BIGQUERY CONFIGURATION ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "adk-mini-project")
DATASET_ID = os.getenv("GCP_DATASET_ID", "ipl_stats")
BQ_LOCATION = os.getenv("BQ_LOCATION", "US")

# --- COLUMN DESCRIPTIONS DICTIONARY ---
# Updated to use BigQuery-compliant names (no dots, cannot start with numbers)
COLUMN_DESCRIPTIONS = {
    'match_id': 'Unique file identifier from Cricsheet.',
    'Key': 'Custom Match identifier combining Year and Match Code (e.g., 2025F, 2024Q1).',
    'data_version': 'Cricsheet data version.',
    'created': 'Date the JSON record was created.',
    'revision': 'Revision number of the JSON data.',
    'team1': 'First participating team.',
    'team2': 'Second participating team.',
    'balls_per_over': 'Standard legal deliveries per over (usually 6).',
    'city': 'City where the match was played.',
    'dates': 'Date(s) the match took place.',
    'event_name': 'Name of the tournament (e.g., Indian Premier League).',
    'event_stage': 'Stage of the tournament (e.g., Group, Final).',
    'gender': 'Gender of the participating players.',
    'match_type': 'Format of the match (e.g., T20).',
    'season': 'The year or edition of the tournament.',
    'team_type': 'Type of team (e.g., club).',
    'overs': 'Number of scheduled overs per innings.',
    'toss_winner': 'Team that won the toss.',
    'toss_decision': 'Decision made by the toss winner (bat or field).',
    'venue': 'Stadium where the match was played.',
    'winner': 'Team that won the match.',
    'win_by_runs': 'Margin of victory in runs.',
    'win_by_wickets': 'Margin of victory in wickets.',
    'player_name': 'Name of the player.',
    'team': 'Name of the team.',
    'registry_id': 'Unique alphanumeric identifier from Cricsheet (maps to external profiles).',
    'inning': 'Innings number (1 or 2).',
    'batting_team': 'Team currently batting in this innings.',
    'bowling_team': 'Team currently bowling in this innings.',
    'runs': 'Total runs scored off the bat.',
    'wickets': 'Total wickets lost.',
    'total_extras': 'Total extra runs conceded.',
    'wides': 'Total wide runs conceded.',
    'noballs': 'Total no-ball runs conceded.',
    'byes': 'Total bye runs conceded.',
    'legbyes': 'Total leg-bye runs conceded.',
    'penalty': 'Total penalty runs awarded.',
    'is_powerplay': 'Flag (1/0) indicating if the delivery occurred during a powerplay.',
    'over': 'The current over number (0-indexed).',
    'ball_number': 'The delivery number within the over.',
    'batter_ball': 'Flag (1/0) indicating a legitimate ball faced by the batter (0 for wides).',
    'batter': 'Name of the striker.',
    'bowler': 'Name of the bowler.',
    'non_striker': 'Name of the non-striker.',
    'runs_batter': 'Runs scored off the bat.',
    'is_four': 'Flag (1/0) if the batter hit a four.',
    'is_six': 'Flag (1/0) if the batter hit a six.',
    'runs_extras': 'Extra runs conceded on this delivery.',
    'runs_total': 'Total runs scored on this delivery (batter + extras).',
    'bowler_runs_conceded': 'Runs charged to the bowler (total - byes - legbyes).',
    'is_wicket': 'Flag (1/0) if a wicket fell on this delivery.',
    'bowler_wicket': 'Flag (1/0) if a bowler was credited with the wicket.',
    'player_out': 'Name of the dismissed batter.',
    'kind': 'Type of dismissal (e.g., caught, bowled, run out).',
    'fielders_involved': 'Comma-separated list of fielders involved in the dismissal.',
    'team_score': 'Total team runs at the time of dismissal.',
    'team_wickets': 'Total team wickets down at the time of dismissal.',
    'by_team': 'Team that initiated the review.',
    'umpire': 'On-field umpire being reviewed.',
    'decision': 'Result of the review (e.g., upheld, struck down).',
    'type': 'Type of review (e.g., wicket).',
    'umpires_call': 'Flag indicating if the decision stayed with the umpire\'s call.',
    'player_in': 'Name of the player subbing in.',
    'player_out': 'Name of the player subbing out.',
    'reason': 'Reason for replacement (e.g., impact_player, concussion).',
    'runs_in_over': 'Total runs scored during this specific over.',
    'wickets_in_over': 'Total wickets lost during this specific over.',
    'team_runs_after_over': 'Cumulative team score at the end of this over.',
    'team_wickets_after_over': 'Cumulative wickets lost at the end of this over.',
    'balls': 'Total legal deliveries faced.',
    'fours': 'Count of boundaries (4 runs).',
    'sixes': 'Count of sixes (6 runs).',
    'strike_rate': 'Runs scored per 100 balls.',
    'not_out': 'Flag (1/0) indicating if the batter remained undefeated.',
    'dismissal_kind': 'Type of dismissal, if out.',
    'dismissal_bowler': 'Bowler credited with the dismissal.',
    'dismissal_fielder': 'Fielder(s) involved in the dismissal.',
    'overs_balls': 'Total overs bowled formatted as O.B (e.g., 4.0 or 3.2).',
    'maidens': 'Number of overs bowled conceding 0 bowler runs.',
    'runs_conceded': 'Total runs charged to the bowler\'s economy.',
    'economy': 'Runs conceded per over bowled.',
    'wicket_no': 'The wicket partnership sequence number.',
    'player_1': 'Name of the first batter in the partnership.',
    'p1_runs': 'Runs contributed by player 1 during this partnership.',
    'p1_balls': 'Balls faced by player 1 during this partnership.',
    'player_2': 'Name of the second batter in the partnership.',
    'p2_runs': 'Runs contributed by player 2 during this partnership.',
    'p2_balls': 'Balls faced by player 2 during this partnership.',
    'total_runs': 'Total runs scored by the team during this partnership (including extras).'
}

def apply_bq_descriptions(project_id, dataset_id, table_name):
    """Updates the BigQuery schema to attach descriptions to columns after replacement."""
    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset_id}.{table_name}"
    
    table = client.get_table(table_id) 
    new_schema = []
    
    for field in table.schema:
        new_field = bigquery.SchemaField(
            name=field.name,
            field_type=field.field_type,
            mode=field.mode,
            description=COLUMN_DESCRIPTIONS.get(field.name, field.description)
        )
        new_schema.append(new_field)
        
    table.schema = new_schema
    client.update_table(table, ["schema"])  

def download_and_extract_cricsheet_data():
    zip_url = "https://cricsheet.org/downloads/ipl_male_json.zip"
    zip_filename = "ipl_male_json.zip"
    extract_folder = "ipl_json_files"

    print(f"Downloading IPL JSON data from: {zip_url}")
    try:
        response = requests.get(zip_url, stream=True)
        response.raise_for_status() 
        
        with open(zip_filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Successfully downloaded to '{zip_filename}'.")
        
    except requests.exceptions.RequestException as e:
        print(f"\nDownload failed: {e}")
        print("In case the link has changed or is broken, please check the updated link at:")
        print("https://cricsheet.org/downloads")
        return None

    print(f"Extracting files into '{extract_folder}' folder...")
    os.makedirs(extract_folder, exist_ok=True)
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
        
    print("Extraction complete!\n")
    return extract_folder

def process_cricsheet_data():
    folder_path = download_and_extract_cricsheet_data()
    
    if not folder_path:
        print("Aborting process due to download failure.")
        return

    master_data = {
        'matches': [], 'mom': [], 'players': [], 'totals': [], 
        'deliveries': [], 'wickets': [], 'reviews': [], 'replacements': [],
        'overs': [], 'batting_scorecard': [], 'bowling_scorecard': [],
        'partnerships': []
    }
    
    processing_log = []
    json_files = glob.glob(os.path.join(folder_path, '*.json'))
    
    if not json_files:
        print(f"No JSON files found in '{folder_path}'.")
        return

    print(f"Found {len(json_files)} files. Starting extraction...")

    stage_map = {
        'final': 'F', 'eliminator': 'E', 'qualifier 1': 'Q1', 'qualifier 2': 'Q2',
        'semi final 1': 'S1', 'semi-final 1': 'S1', 'semi final 2': 'S2', 
        'semi-final 2': 'S2', 'third place play-off': 'T', 'third place match': 'T'
    }

    non_bowler_dismissals = ['run out', 'retired hurt', 'obstructing the field', 'retired out', 'retired']

    deliveries_col_order = [
        'match_id', 'Key', 'inning', 'batting_team', 'bowling_team', 'is_powerplay', 
        'over', 'ball_number', 'batter_ball', 
        'batter', 'bowler', 'non_striker', 
        'runs_batter', 'is_four', 'is_six', 
        'runs_extras', 'wides', 'noballs', 'byes', 'legbyes', 'penalty', 
        'runs_total', 'bowler_runs_conceded', 
        'is_wicket', 'bowler_wicket'
    ]

    for file_path in json_files:
        filename = os.path.basename(file_path)
        match_id = filename.split('.')[0]
        
        log_entry = {
            'match_id': match_id, 'file_name': filename, 'error_status': 'None',
            'matches': 'Not Required', 'mom': 'Not Required', 'players': 'Not Required', 
            'totals': 'Not Required', 'deliveries': 'Not Required', 'wickets': 'Not Required', 
            'reviews': 'Not Required', 'replacements': 'Not Required', 'overs': 'Not Required', 
            'batting_scorecard': 'Not Required', 'bowling_scorecard': 'Not Required',
            'partnerships': 'Not Required'
        }

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            info = data.get('info', {})
            event = info.get('event', {})
            
            dates = info.get('dates', [])
            year = dates[0].split('-')[0] if dates else str(info.get('season', ''))[:4]
            
            stage_str = event.get('stage', '').lower()
            match_num = event.get('match_number', '')
            
            if stage_str in stage_map:
                match_code = stage_map[stage_str]
            elif match_num:
                match_code = str(match_num)
            else:
                match_code = 'U' 
                
            match_key = f"{year}{match_code}"

            meta = data.get('meta', {})
            outcome = info.get('outcome', {})
            by = outcome.get('by', {})
            toss = info.get('toss', {})
            
            teams_list = info.get('teams', [])
            team1 = teams_list[0] if len(teams_list) > 0 else None
            team2 = teams_list[1] if len(teams_list) > 1 else None

            master_data['matches'].append({
                'match_id': match_id, 'Key': match_key, 'data_version': meta.get('data_version'),
                'created': meta.get('created'), 'revision': meta.get('revision'),
                'team1': team1, 'team2': team2,
                'balls_per_over': info.get('balls_per_over'), 'city': info.get('city'),
                'dates': ", ".join(info.get('dates', [])), 'event_name': event.get('name'),
                'event_stage': event.get('stage'), 'gender': info.get('gender'),
                'match_type': info.get('match_type'), 'season': info.get('season'),
                'team_type': info.get('team_type'), 'overs': info.get('overs'),
                'toss_winner': toss.get('winner'), 'toss_decision': toss.get('decision'),
                'venue': info.get('venue'), 'winner': outcome.get('winner'),
                'win_by_runs': by.get('runs'), 'win_by_wickets': by.get('wickets')
            })
            log_entry['matches'] = 'Updated'

            if 'player_of_match' in info:
                for p in info['player_of_match']:
                    master_data['mom'].append({'match_id': match_id, 'Key': match_key, 'player_name': p})
                log_entry['mom'] = 'Updated'

            if 'players' in info:
                registry = info.get('registry', {}).get('people', {})
                for team, players in info['players'].items():
                    for player in players:
                        master_data['players'].append({
                            'match_id': match_id, 'Key': match_key, 'team': team, 
                            'player_name': player, 'registry_id': registry.get(player)
                        })
                log_entry['players'] = 'Updated'

            if 'innings' in data:
                log_entry['deliveries'] = 'Updated'

                for inning_idx, inning in enumerate(data['innings'], start=1):
                    batting_team = inning.get('team')
                    bowling_team = team2 if batting_team == team1 else team1
                    
                    inning_runs = 0
                    inning_wickets = 0
                    
                    inning_extras = 0
                    inning_wides = 0
                    inning_noballs = 0
                    inning_byes = 0
                    inning_legbyes = 0
                    inning_penalty = 0
                    
                    legal_balls = 0
                    pp_list = inning.get('powerplays', [])

                    for over in inning.get('overs', []):
                        over_num = over.get('over')
                        
                        for ball_idx, ball in enumerate(over.get('deliveries', []), start=1):
                            bowler = ball.get('bowler')
                            runs = ball.get('runs', {})
                            extras = ball.get('extras', {})
                            
                            runs_total = runs.get('total', 0)
                            inning_runs += runs_total
                            
                            wides = extras.get('wides', 0)
                            byes = extras.get('byes', 0)
                            legbyes = extras.get('legbyes', 0)
                            noballs = extras.get('noballs', 0)
                            penalty = extras.get('penalty', 0)
                            
                            inning_extras += runs.get('extras', 0)
                            inning_wides += wides
                            inning_noballs += noballs
                            inning_byes += byes
                            inning_legbyes += legbyes
                            inning_penalty += penalty
                            
                            bowler_runs_conceded = runs_total - byes - legbyes
                            batter_ball = 0 if wides > 0 else 1

                            is_wicket = 1 if 'wickets' in ball else 0
                            bowler_wicket = 0

                            if is_wicket:
                                wicket_list = ball.get('wickets', [])
                                inning_wickets += len(wicket_list)
                                for w in wicket_list:
                                    if w.get('kind') not in non_bowler_dismissals:
                                        bowler_wicket += 1
                            
                            if wides == 0 and noballs == 0:
                                legal_balls += 1

                            is_powerplay = 0
                            for pp in pp_list:
                                start_over = int(pp.get('from', 99))
                                end_over = int(pp.get('to', -1))
                                if start_over <= over_num <= end_over:
                                    is_powerplay = 1
                                    break
                            
                            batter_runs = runs.get('batter', 0)
                            is_four = 1 if batter_runs in [4, 5] else 0
                            is_six = 1 if batter_runs == 6 else 0

                            master_data['deliveries'].append({
                                'match_id': match_id, 'Key': match_key, 'inning': inning_idx, 
                                'batting_team': batting_team, 'bowling_team': bowling_team, 
                                'over': over_num, 'ball_number': ball_idx,
                                'batter': ball.get('batter'), 'bowler': bowler, 'non_striker': ball.get('non_striker'),
                                'runs_batter': batter_runs, 'runs_extras': runs.get('extras', 0), 'runs_total': runs_total,
                                'bowler_runs_conceded': bowler_runs_conceded, 
                                'is_four': is_four, 'is_six': is_six, 'is_powerplay': is_powerplay,
                                'batter_ball': batter_ball,
                                'wides': wides, 'legbyes': legbyes,
                                'noballs': noballs, 'byes': byes, 'penalty': penalty,
                                'is_wicket': is_wicket, 'bowler_wicket': bowler_wicket
                            })

                            if is_wicket:
                                for wicket in ball.get('wickets', []):
                                    player_out = wicket.get('player_out')
                                    kind = wicket.get('kind')
                                    
                                    fielders_list = wicket.get('fielders', [])
                                    fielder_names = [f.get('name') for f in fielders_list if 'name' in f]
                                    fielders_str = ", ".join(fielder_names) if fielder_names else None

                                    master_data['wickets'].append({
                                        'match_id': match_id, 'Key': match_key, 'inning': inning_idx, 
                                        'over': over_num, 'ball_number': ball_idx,
                                        'bowler': bowler, 'player_out': player_out, 'kind': kind,
                                        'fielders_involved': fielders_str,
                                        'team_score': inning_runs, 'team_wickets': inning_wickets
                                    })
                                    log_entry['wickets'] = 'Updated'

                            if 'review' in ball:
                                rev = ball['review']
                                master_data['reviews'].append({
                                    'match_id': match_id, 'Key': match_key, 'inning': inning_idx, 
                                    'over': over_num, 'ball_number': ball_idx,
                                    'by_team': rev.get('by'), 'umpire': rev.get('umpire'), 'batter': rev.get('batter'),
                                    'decision': rev.get('decision'), 'type': rev.get('type'), 'umpires_call': rev.get('umpires_call', False)
                                })
                                log_entry['reviews'] = 'Updated'

                            if 'replacements' in ball and 'match' in ball['replacements']:
                                for rep in ball['replacements']['match']:
                                    master_data['replacements'].append({
                                        'match_id': match_id, 'Key': match_key, 'inning': inning_idx, 
                                        'over': over_num, 'ball_number': ball_idx,
                                        'player_in': rep.get('in'), 'player_out': rep.get('out'),
                                        'team': rep.get('team'), 'reason': rep.get('reason')
                                    })
                                    log_entry['replacements'] = 'Updated'

                    overs_formatted = f"{legal_balls // 6}.{legal_balls % 6}"
                    master_data['totals'].append({
                        'match_id': match_id, 'Key': match_key, 'inning': inning_idx,
                        'batting_team': batting_team, 'runs': inning_runs, 
                        'wickets': inning_wickets, 'overs': overs_formatted,
                        'total_extras': inning_extras,
                        'wides': inning_wides,
                        'noballs': inning_noballs,
                        'byes': inning_byes,
                        'legbyes': inning_legbyes,
                        'penalty': inning_penalty
                    })
                    log_entry['totals'] = 'Updated'

        except Exception as e:
            log_entry['error_status'] = f"Error: {str(e)}"
            print(f"Failed to process {filename}: {str(e)}")

        finally:
            log_entry['Key_Generated'] = match_key if 'match_key' in locals() else 'Failed'
            processing_log.append(log_entry)

    # --- DATAFRAME AGGREGATIONS ---
    if master_data['deliveries']:
        print("Aggregating Scorecards, Overs, and Partnerships...")
        del_df = pd.DataFrame(master_data['deliveries'])
        del_df['is_legal_ball'] = ((del_df['wides'] == 0) & (del_df['noballs'] == 0)).astype(int)

        # 1. OVERS TABLE
        overs_group = del_df.groupby(['match_id', 'Key', 'inning', 'batting_team', 'bowling_team', 'over'], as_index=False).agg(
            runs_in_over=('runs_total', 'sum'),
            wickets_in_over=('is_wicket', 'sum')
        ).sort_values(['match_id', 'inning', 'over'])
        overs_group['team_runs_after_over'] = overs_group.groupby(['match_id', 'inning'])['runs_in_over'].cumsum()
        overs_group['team_wickets_after_over'] = overs_group.groupby(['match_id', 'inning'])['wickets_in_over'].cumsum()
        master_data['overs'] = overs_group.to_dict('records')

        # 2. PARTNERSHIPS TABLE
        del_df['pair_tuple'] = del_df.apply(lambda r: tuple(sorted([r['batter'], r['non_striker']])), axis=1)
        del_df['pair_changed'] = (del_df['pair_tuple'] != del_df.groupby(['match_id', 'inning'])['pair_tuple'].shift()).astype(int)
        del_df['partnership_id'] = del_df.groupby(['match_id', 'inning'])['pair_changed'].cumsum()

        partnership_data = []
        for (m_id, key, inn, b_team, p_id), group in del_df.groupby(['match_id', 'Key', 'inning', 'batting_team', 'partnership_id']):
            pair = group['pair_tuple'].iloc[0]
            p1, p2 = pair[0], pair[1]
            
            p1_mask = group['batter'] == p1
            p1_runs = group.loc[p1_mask, 'runs_batter'].sum()
            p1_balls = group.loc[p1_mask, 'batter_ball'].sum()
            
            p2_mask = group['batter'] == p2
            p2_runs = group.loc[p2_mask, 'runs_batter'].sum()
            p2_balls = group.loc[p2_mask, 'batter_ball'].sum()
            
            total_runs = group['runs_total'].sum()
            
            partnership_data.append({
                'match_id': m_id, 'Key': key, 'inning': inn, 'batting_team': b_team,
                'wicket_no': p_id, 'player_1': p1, 'p1_runs': p1_runs, 'p1_balls': p1_balls,
                'player_2': p2, 'p2_runs': p2_runs, 'p2_balls': p2_balls, 'total_runs': total_runs
            })
        master_data['partnerships'] = partnership_data

        # 3. BATTING SCORECARD
        batters_s = del_df[['match_id', 'Key', 'inning', 'batting_team', 'batter']].rename(columns={'batter': 'player'})
        batters_ns = del_df[['match_id', 'Key', 'inning', 'batting_team', 'non_striker']].rename(columns={'non_striker': 'player'})
        all_batters = pd.concat([batters_s, batters_ns]).drop_duplicates()

        batting_stats = del_df.groupby(['match_id', 'inning', 'batter'], as_index=False).agg(
            runs=('runs_batter', 'sum'), balls=('batter_ball', 'sum'),
            fours=('is_four', 'sum'), sixes=('is_six', 'sum')
        )
        # We removed the renaming to 4s and 6s to comply with BigQuery

        scorecard = pd.merge(all_batters, batting_stats, left_on=['match_id', 'inning', 'player'], right_on=['match_id', 'inning', 'batter'], how='left')
        scorecard.fillna({'runs': 0, 'balls': 0, 'fours': 0, 'sixes': 0}, inplace=True)
        scorecard['strike_rate'] = (scorecard['runs'] / scorecard['balls'] * 100).replace([np.inf, -np.inf], 0.0).fillna(0.0).round(2)

        if master_data['wickets']:
            wickets_df = pd.DataFrame(master_data['wickets'])
            w_sub = wickets_df[['match_id', 'inning', 'player_out', 'kind', 'bowler', 'fielders_involved']]
            
            scorecard = pd.merge(scorecard, w_sub, left_on=['match_id', 'inning', 'player'], right_on=['match_id', 'inning', 'player_out'], how='left')
            scorecard['not_out'] = scorecard['player_out'].isna().astype(int)
            
            scorecard.rename(columns={
                'player': 'batter', 'kind': 'dismissal_kind', 
                'bowler': 'dismissal_bowler', 'fielders_involved': 'dismissal_fielder'
            }, inplace=True)
            scorecard.drop(columns=['player_out', 'batter_y'], errors='ignore', inplace=True)
            scorecard.rename(columns={'batter_x': 'batter'}, errors='ignore', inplace=True)
        else:
            scorecard['not_out'] = 1
            scorecard['dismissal_kind'] = None
            scorecard['dismissal_bowler'] = None
            scorecard['dismissal_fielder'] = None
            scorecard.rename(columns={'player': 'batter'}, inplace=True)

        master_data['batting_scorecard'] = scorecard.to_dict('records')

        # 4. BOWLING SCORECARD
        maidens_df = del_df.groupby(['match_id', 'inning', 'bowler', 'over'], as_index=False).agg(
            over_runs_conceded=('bowler_runs_conceded', 'sum'),
            over_legal_balls=('is_legal_ball', 'sum')
        )
        maidens_df['is_maiden'] = ((maidens_df['over_runs_conceded'] == 0) & (maidens_df['over_legal_balls'] >= 6)).astype(int)
        maidens_agg = maidens_df.groupby(['match_id', 'inning', 'bowler'], as_index=False).agg(maidens=('is_maiden', 'sum'))

        bowling_group = del_df.groupby(['match_id', 'Key', 'inning', 'bowling_team', 'bowler'], as_index=False).agg(
            runs_conceded=('bowler_runs_conceded', 'sum'),
            wickets=('bowler_wicket', 'sum'),
            legal_balls=('is_legal_ball', 'sum')
        )
        
        bowling_group = pd.merge(bowling_group, maidens_agg, on=['match_id', 'inning', 'bowler'], how='left')
        
        # Changed overs.balls to overs_balls to comply with BigQuery
        bowling_group['overs_balls'] = (bowling_group['legal_balls'] // 6).astype(str) + "." + (bowling_group['legal_balls'] % 6).astype(str)
        bowling_group['economy'] = (bowling_group['runs_conceded'] / (bowling_group['legal_balls'] / 6)).replace([np.inf, -np.inf], 0.0).fillna(0.0).round(2)
        
        bowling_group = bowling_group[['match_id', 'Key', 'inning', 'bowling_team', 'bowler', 'overs_balls', 'maidens', 'wickets', 'runs_conceded', 'economy']]
        master_data['bowling_scorecard'] = bowling_group.to_dict('records')

        for d in master_data['deliveries']:
            d.pop('is_legal_ball', None)
            d.pop('pair_tuple', None)
            d.pop('pair_changed', None)
            d.pop('partnership_id', None)

    # --- PUSH DIRECTLY TO BIGQUERY ---
    print(f"\nProcessing complete. Pushing to BigQuery ({PROJECT_ID}.{DATASET_ID})...")
    
    for table_name, data_list in master_data.items():
        if data_list:
            df = pd.DataFrame(data_list)
            
            if table_name == 'deliveries':
                df = df[deliveries_col_order]
            else:
                cols = ['match_id', 'Key'] + [c for c in df.columns if c not in ['match_id', 'Key']]
                df = df[cols]
                
            # LOCAL SAVE COMMENTED OUT
            # file_path = os.path.join(output_dir, f"{table_name}.csv")
            # df.to_csv(file_path, index=False)
            # print(f"\n - Created {table_name}.csv ({len(df)} rows)")

            # Upload to BigQuery
            table_id = f"{DATASET_ID}.{table_name}"
            print(f"   -> Uploading to BigQuery table: {table_id}...")
            try:
                pandas_gbq.to_gbq(
                    df,
                    destination_table=table_id,
                    project_id=PROJECT_ID,
                    if_exists='replace',
                    location=BQ_LOCATION
                )
                
                # Apply column descriptions to BigQuery schema
                apply_bq_descriptions(PROJECT_ID, DATASET_ID, table_name)
                print(f"   -> Successfully updated table and applied column descriptions.")
                
            except Exception as e:
                print(f"   -> BigQuery upload failed for {table_name}: {str(e)}")

    # Handle Log Table
    log_df = pd.DataFrame(processing_log)
    cols = ['match_id', 'Key_Generated', 'file_name', 'error_status'] + [c for c in log_df.columns if c not in ['match_id', 'Key_Generated', 'file_name', 'error_status']]
    log_df = log_df[cols]
    
    # LOCAL SAVE COMMENTED OUT
    # log_path = os.path.join(output_dir, 'processing_log.csv')
    # log_df.to_csv(log_path, index=False)
    # print(f"\n - Created processing_log.csv")
    
    print("   -> Uploading processing_log to BigQuery...")
    try:
        pandas_gbq.to_gbq(
            log_df,
            destination_table=f"{DATASET_ID}.processing_log",
            project_id=PROJECT_ID,
            if_exists='replace',
            location=BQ_LOCATION
        )
        print("   -> Log table pushed successfully.")
    except Exception as e:
        print(f"   -> Log table push failed: {str(e)}")
    
    print("\nData pipeline execution fully complete!")

if __name__ == "__main__":
    process_cricsheet_data()