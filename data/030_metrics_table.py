import os
import sys
import datetime as dt
import pandas as pd
import numpy as np
from pre_process_library import batting as bats
from pre_process_library import pitching as pitch
import utilities as util
CONFIG = util.load_config()
    

def process_date_games(data):
    """
    """

    # Iterate over metrics
    # ------------------------------
    
    # Add Game-Team Batting
    df_game_team_mx = bats.hits_in_last_5_games(data)
    df_game_team_mx = bats.hits_in_last_10_games(df_game_team_mx)
    df_game_team_mx = bats.runs_in_last_5_games(df_game_team_mx)
    df_game_team_mx = bats.runs_in_last_10_games(df_game_team_mx)
    df_game_team_mx = bats.runs_per_game_last_10(df_game_team_mx)
    df_game_team_mx = bats.runs_per_hit_last_5(df_game_team_mx)
    df_game_team_mx = bats.runs_per_hit_last_10(df_game_team_mx)

    # Add Game-Team Pitching
    df_game_team_mx = pitch.pitcher_id_indicator(df_game_team_mx)
    
    # Return
    return df_game_team_mx




if __name__ == "__main__":

    # Run Log
    min_date = dt.datetime(year=2018, month=8, day=1)
    max_date = dt.datetime(year=2019, month=9, day=1)

    # Iterate over years
    years = [y for y in np.arange(min_date.year, max_date.year+1, 1)]
    dates = [min_date+dt.timedelta(days=i)
             for i in range((max_date-min_date).days+1)]

    # Establish path
    df_all = []
    for dd in dates:

        # Get path for individual file and add date to path
        path = CONFIG.get('paths').get('game_team_stats')
        dd = dd.strftime('year_%Ymonth_%mday_%d/')
        path += dd         
        print(path)

        # Read in from path and append to list
        try:
            df_game_team_mx = pd.read_parquet(path+"game_team_stats.parquet")
            df_all.append(df_game_team_mx)
        except:
            continue

        # Handle path once split back out to days
        if not os.path.exists(
            CONFIG.get('paths').get('game_team_metrics') + \
            path.split("/")[-2]+"/"
        ):
            os.makedirs(
                CONFIG.get('paths').get('game_team_metrics') + \
                path.split("/")[-2]+"/"
            )
    # Concatenate stats tables vertically and process
    df_all = pd.concat(objs=df_all, axis=0)
    df_metrics = process_date_games(df_all)

    # Wite back out
    for dd in dates:
        print(dd)
        df_metrics_cut = df_metrics.loc[
            df_metrics['gameId'].str.contains(
                dd.strftime('%Y_%m_%d')
            ),
        :]
        print(df_metrics_cut.shape)
        path = CONFIG.get('paths').get('game_team_stats')
        dd = dd.strftime('year_%Ymonth_%mday_%d/')
        path += dd
        if df_metrics_cut.shape[0] > 0:
            df_metrics_cut.to_parquet(
                CONFIG.get('paths').get('game_team_metrics') + \
                path.split("/")[-2]+"/" + \
                'game_team_metrics.parquet'
            )
            df_metrics_cut.to_csv(
                CONFIG.get('paths').get('game_team_metrics') + \
                path.split("/")[-2]+"/" + \
                'game_team_metrics.csv',
                index=False
            )
