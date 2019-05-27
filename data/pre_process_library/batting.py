import os
import sys
import datetime as dt
import pandas as pd
import numpy as np


def sum_last_x_games(data, metric, level, stat, lookback):
    """
    """

    # Sort by team and gameDate
    data.sort_values(
        by=[level, 'gameDate'],
        ascending=True,
        inplace=True
    )

    # Rolling
    data[metric] = \
        data.groupby(level)[stat].apply(
            pd.rolling_sum,
            lookback,
            min_periods=lookback
        )

    return data


def hits_in_last_5_games(data):
    """
    """

    # Hits at team level
    data = data.groupby(
        by=['gameId'],
        as_index=False
    ).agg({'batterH': 'sum'})
    data['team_hits_in_last_5_games'] = data.groupby(
        by=['gameId'],
        as_index=False
    )['batterH'].rolling(5).sum().reset_index(0, drop=True)

    return data


def runs_in_last_5_games(data):
    """
    """

    # Runs at team level
    data = data.groupby(
        by=['gameId'],
        as_index=False
    ).agg({'batterR': 'sum'})
    data['team_runs_in_last_5_games'] = data.groupby(
        by=['gameId'],
        as_index=False
    )['batterR'].rolling(5).sum().reset_index(0, drop=True)

    return data


def runs_per_game_last_10(data):
    """
    """

    # Runs at team level
    data = data.groupby(
        by=['gameId'],
        as_index=False
    ).agg({'batterR': 'sum'})
    data['runs_last_10'] = data.groupby(
        by=['gameId'],
        as_index=False
    )['batterR'].rolling(10).sum().reset_index(0, drop=True)
    data['runs_per_game_last_10'] = (
        data['runs_last_10'] / 10
    )

    return data
    
         
