import os
import sys
import pandas as pd
import numpy as np
import datetime as dt
import inspect
import utilities as util
from pre_process_library import rename as names
from pre_process_library import additions as add
CONFIG = util.load_config()


def hits_in_last_5_games(data):
    """
    https://stackoverflow.com/questions/50413786/pandas-groupby-multiple-columns-with-rolling-date-offset-how

    https://github.com/pandas-dev/pandas/issues/13966
    """

    # Rolling
    data.sort_values(by=['team', 'gameId'], ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    data['team_hits_last_5_games'] = data.groupby('team')\
        ['game_hits_sum'].rolling(5).sum().reset_index(drop=True)
    return data


def hits_in_last_10_games(data):
    """
    """

    # Rolling
    data.sort_values(by=['team', 'gameId'], ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    data['team_hits_last_10_games'] = data.groupby('team')\
        ['game_hits_sum'].rolling(10).sum().reset_index(drop=True)
    
    return data


def runs_in_last_5_games(data):
    """
    """

    # Rolling
    data.sort_values(by=['team', 'gameId'], ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    data['team_runs_last_5_games'] = data.groupby('team')\
        ['game_runs_sum'].rolling(5).sum().reset_index(drop=True)
    return data


def runs_in_last_10_games(data):
    """
    """

    # Rolling
    data.sort_values(by=['team', 'gameId'], ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    data['team_runs_last_10_games'] = data.groupby('team')\
        ['game_runs_sum'].rolling(10).sum().reset_index(drop=True)
    return data


def runs_per_game_last_10(data):
    """
    """

    # Initial agg
    data.sort_values(by=['team', 'gameId'], ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    data['team_runs_per_game_last_10'] = data.groupby('team')\
        ['game_runs_sum'].rolling(10).mean().reset_index(drop=True)
    return data
    

def runs_per_hit_last_5(data):
    """
    """

    # Add necessary other metrics
    if 'team_runs_last_5_games' not in data.columns:
        data = runs_in_last_5_games(data)
    if 'team_hits_last_5_games' not in data.columns:
        data = hits_in_last_5_games(data)

    # Divide
    data['runs_per_hit_last_5'] = (
        data['team_runs_last_5_games'] /
        data['team_hits_last_5_games']
    )
    return data


def runs_per_hit_last_10(data):
    """
    """

    # Add necessary other metrics
    if 'team_runs_last_10_games' not in data.columns:
        data = runs_in_last_10_games(data)
    if 'team_hits_last_10_games' not in data.columns:
        data = hits_in_last_5_games(data)

    # Divide
    data['runs_per_hit_last_10'] = (
        data['team_runs_last_10_games'] /
        data['team_hits_last_10_games']
    )
    return data



    
