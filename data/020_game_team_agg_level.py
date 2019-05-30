import os
import sys
import datetime as dt
import pandas as pd
import numpy as np
from pre_process_library import batting as bats
import utilities as util
CONFIG = util.load_config()


def game_level_batting_metrics(data):
    """
    """

    # Summary Stats Table
    game_metrics = data.groupby(
        by=['gameId', 'team'],
        as_index=False
    ).agg({'batterH': {'game_hits_sum': 'sum',
                       'game_batter_hits_mean': 'mean',
                       'game_batter_hits_std': 'std',
                       'game_batter_hits_var': 'var'},

           'batterR': {'game_runs_sum': 'sum',
                       'game_batter_runs_std': 'std',
                       'game_batter_runs_var': 'var'},

           'batterOBP': {'game_obp_mean': 'mean',
                         'game_batter_obp_var': 'var',
                         'game_batter_obp_std': 'std'},

           'batterOPS': {'game_ops_mean': 'mean',
                         'game_batter_ops_var': 'var',
                         'game_batter_ops_std': 'std'},

           'batterHR': {'game_hr_sum': 'sum',
                        'game_batter_hr_mean': 'mean',
                        'game_batter_hr_std': 'std',
                        'game_batter_hr_var': 'var'},
           
           'batterBB': {'game_bb_sum': 'sum',
                        'game_batter_bb_mean': 'mean',
                        'game_batter_bb_std': 'std',
                        'game_batter_bb_var': 'var'}})

    # Format Summary Stats Table
    game_metrics.reset_index(inplace=True)
    game_metrics.columns = [x[0] if x[1] == '' else x[1] for x in
                            game_metrics.columns]

    return game_metrics


def game_level_pitching_metrics(data):
    """
    """

    game_metrics = data.groupby(
        by=['gameId', 'team'],
        as_index=False
    ).agg({'pitcherId': {'pitchersUsed': 'count'},
           'pitcherBBAllowed': {'game_bb_sum': 'sum',
                                'game_pitcher_bb_mean': 'mean',
                                'game_pitcher_bb_std': 'std',
                                'game_pitcher_bb_var': 'var'}})
    game_metrics.reset_index(inplace=True)
    game_metrics.columns = [x[0] if x[1] == '' else x[1] for x in
                            game_metrics.columns]
    return game_metrics


def game_level_inning_metrics(data):
    """
    """

    game_metrics = data.loc[:, [
        'gameId', 'team', 'atbat_pitcher', 'starting_pitcher_flag'
    ]].drop_duplicates(inplace=False)
    return game_metrics


def assemble_game_team_metrics(metrics_tables):
    """
    """

    # Merge Cols
    merge_cols = ['gameId', 'team']

    # Master Game/Team Table to Merge Left on to
    game_team_metrics = [
        table[['gameId', 'team']] for table in metrics_tables
    ]
    game_team_metrics_table = pd.concat(
        game_team_metrics,
        axis=0
    )
    game_team_metrics_table.drop_duplicates(inplace=True)

    # Iterate over metrics tables to merge
    for metrics_tbl in metrics_tables:
        game_team_metrics_table = pd.merge(
            game_team_metrics_table,
            metrics_tbl,
            how='left',
            on=['gameId', 'team'],
            validate='1:1'
        )
    return game_team_metrics_table
    

def process_date_games(path):
    """
    """

    # Read in 4 standardized tables
    df_batting = pd.read_parquet(path+"batting.parquet")
    df_pitching = pd.read_parquet(path+"pitching.parquet")
    df_boxscore = pd.read_parquet(path+"boxscore.parquet")
    df_innings = pd.read_parquet(path+"innings.parquet")

    # Create desination dir
    if not os.path.exists(
        CONFIG.get('paths').get('game_team_stats') + \
        path.split("/")[-2]+"/"
    ):
        os.makedirs(
            CONFIG.get('paths').get('game_team_stats') + \
            path.split("/")[-2]+"/"
        )

    # Metrics results
    metrics = []
    
    # Game Team Level Batting Metrics
    batting_game_metrics = game_level_batting_metrics(df_batting)
    metrics.append(batting_game_metrics)

    # Game Team Level Pitching Metrics
    pitching_game_metrics = game_level_pitching_metrics(df_pitching)
    metrics.append(pitching_game_metrics)
    
    # Game Team Level Inning Details
    inning_game_metrics = game_level_inning_metrics(df_innings)
    metrics.append(inning_game_metrics)
    
    # Assemble Game Team Metrics
    game_metrics = assemble_game_team_metrics(metrics)

    # Save out
    game_metrics.to_parquet(
        CONFIG.get('paths').get('game_team_stats') + \
        path.split("/")[-2]+"/" + \
        "game_team_stats.parquet"
    )
    if 'day_01' in path:
        game_metrics.to_csv(
            CONFIG.get('paths').get('game_team_stats') + \
            path.split("/")[-2] + "/" + \
            "game_team_stats.csv",
            index=False
        )


if __name__ == "__main__":

    # Run Log
    min_date = dt.datetime(year=2018, month=8, day=1)
    max_date = dt.datetime(year=2018, month=9, day=1)

    # Iterate over years
    years = [y for y in np.arange(min_date.year, max_date.year+1, 1)]
    dates = [min_date+dt.timedelta(days=i)
             for i in range((max_date-min_date).days+1)]

    # Establish path
    for dd in dates:
        path = CONFIG.get('paths').get('normalized')
        dd = dd.strftime('year_%Ymonth_%mday_%d/')
        path += dd         
        print(path)
        process_date_games(path)

    
