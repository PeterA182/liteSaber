import os
import sys
import pandas as pd
import numpy as np
import datetime as dt
import utilities as util
CONFIG = util.load_config()


def add_team(data, path, data_table_key):
    """
    Parse team information from path and add to table
    
    """

    assert data_table_key in ['batter', 'pitcher']

    # Reset Index
    data.reset_index(inplace=True)
    data.drop(labels=['index'], axis=1, inplace=True)
    
    # Add first and second team listed
    data.loc[:, 'firstTeamListed'] = data['gameId'].str.split("_").apply(
        lambda x: x[-3]
    )
    data.loc[:, 'secondTeamListed'] = data['gameId'].str.split("_").apply(
        lambda x: x[-2]
    )

    # Use team flag to determine team
    msk = (data['{}TeamFlag'.format(data_table_key)] == 'home')
    data.loc[msk, 'team'] = data['secondTeamListed']
    data.loc[~msk, 'team'] = data['firstTeamListed']
    data.drop(labels=['firstTeamListed', 'secondTeamListed'],
              axis=1,
              inplace=True)
    return data


def add_game_date(data, path):
    """
    """

    # Add Game Date
    data.loc[:, 'gameDate'] = data['gameId'].str.split("_").apply(
        lambda x: x[-6:-3]
    )
    data['gameDate'] = data['gameDate'].apply(
        lambda x: dt.datetime(
            year=int(x[0]),
            month=int(x[1]),
            day=int(x[2]))
    )
    return data
