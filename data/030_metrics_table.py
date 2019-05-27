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
    df_batting = pd.read_parquet(path+"batting.parquet")
    df_pitching = pd.read_parquet(path+"pitching.parquet")
    df_boxscore = pd.read_parquet(path+"boxscore.parquet")
    df_innings = pd.read_parquet(path+"innings.parquet")

    # Establish metrics table for game date team
    metrics_table = \
        df_batting[['gameId', 'team']].drop_duplicates(inplace=False)

    
    # Iterate over metrics
    # ------------------------------
    
    # Add Batting
    metrics_table = add_batting_metrics(metrics_table, df_batting)
                       

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
        path = CONFIG.get('paths').get('normalized')
        dd = dd.strftime('year_%Ymonth_%mday_%d/')
        path += dd         
        print(path)
        process_date_games(path)
