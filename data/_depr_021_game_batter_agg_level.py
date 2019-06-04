import os
import sys
import datetime as dt
import pandas as pd
import numpy as np
from pre_process_library import batting as bats
import utilities as util
CONFIG = util.load_config()


def add_batting_stats(data):
    """
    """

    game_batter_stats = pd.groupby(
        by=['gameId', 'batterId'],
        as_index=False
    ).agg(({
        


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

    # Base Table
    df_master = df_boxscore.loc[:, ['gameId', 'home_team_code', 'away_team_code']]
    
    # Get Batting Stats from boxscore
    boxscore_batting = batting_from_boxscore(df_boxscore)
    
