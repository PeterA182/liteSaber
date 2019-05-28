import os
import sys
import datetime as dt
import pandas as pd
import numpy as np
from pre_process_library import batting as bats
import utilities as util
CONFIG = util.load_config()


def add_batting_metrics(metrics, batting):
    """

    """

    # Decide from config

    # Metrics
    metrics = []
    
    # Team Hits in Last 5 Games
    m = bats.hits_in_last_5_games(batting)
    metrics.append(m)
    
    # Team Runs in Last 5 Games
    m = bats.runs_in_last_5_games(batting)
    metrics.append(m)

    # Team Runs Per Game Last 10
    m = bats.runs_per_game_last_10(batting)
    metrics.append(m)

    # Construct metrics
    metrics = pd.concat(objs=metrics, axis=1)
    return metrics
    

def process_date_games(path):
    """
    """

    # Read in 4 standardized tabes
    df_game_team_mx = pd.read_parquet(path+"game_metrics.parquet")
    
    # Iterate over metrics
    # ------------------------------
    
    # Add Game-Team Batting
    df_game_team_mx = bats.hits_in_last_5_games(df_game_team_mx)
    df_game_team_mx = bats.hits_in_last_10_games(df_game_team_mx)
    df_game_team_mx = bats.runs_in_last_5_games(df_game_team_mx)
    df_game_team_mx = bats.runs_in_last_10_games(df_game_team_mx)
    df_game_team_mx = bats.runs_per_game_last_10(df_game_team_mx)
    df_game_team_mx = bats.runs_per_hit_last_5(df_game_team_rx)
    df_game_team_mx = bats.runs_per_hit_last_10(df_game_team_rx)
    

    return 0


if __name__ == "__main__":

    # Run Log
    min_date = dt.datetime(year=2018, month=8, day=1)
    max_date = dt.datetime(year=2018, month=8, day=2)

    # Iterate over years
    years = [y for y in np.arange(min_date.year, max_date.year+1, 1)]
    dates = [min_date+dt.timedelta(days=i)
             for i in range((max_date-min_date).days+1)]

    # Establish path
    for dd in dates:
        path = CONFIG.get('paths').get('game_metrics')
        dd = dd.strftime('year_%Ymonth_%mday_%d/')
        path += dd         
        print(path)
        process_date_games(path)
