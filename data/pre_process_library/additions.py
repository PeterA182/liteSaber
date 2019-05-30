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
    if 'gameId' not in data.columns:
        data['gameId'] = data['game_id'].copy(deep=True)
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


def add_starting_pitcher_flag(data):
    """
    """

    # Inning Start Pitcher
    df_inning_1 = data.loc[
        ((data['inning_num'] == 1) &
         (data['atbat_o'] == 0)),
    :]

    # Min atbat_start per pitcher
    min_pitcher_tfs = df_inning_start.groupby(
        by=['atbat_pitcher'],
        as_index=False
    ).agg({'atbat_start_tfs_zulu': 'min'})
    min_pitcher_tfs.rename(
        columns={'atbat_start_tfs_zulu': 'pitcher_min_tfs'},
        inplace=True
    )
    
    # Merge back to first inning
    df_inning_1 = pd.merge(
        df_inning_1, min_pitcher_tfs,
        how='left',
        on=['atbat_pitcher']
    )
    df_inning_1['starting_pitcher_flag'] = (
        df_inning['atbat_start_tfs_zulu'] ==
        df_inning['pitcher_min_tfs']
    ).astype(int)
    df_inning_1 = df_inning_1[['atbat_pitcher', 'starting_pitcher_flag']]\
        .drop_duplicates(inplace=False)
    
    # Merge back to entire game
    data = pd.merge(
        data, df_inning,
        how='left',
        on=['atbat_pitcher'],
        validate='m:1'
    )
    return data


def add_inning_half(data):
    """
    """

    # Get min tfs zulu per inning
    min_tfs = data.groupby(
        by=['inning_num'],
        as_index=False
    ).agg({'atbat_start_tfs_zulu': 'min'})
    min_inning_tfs_dict = min_tfs.set_index('inning_num')\
        ['atbat_start_tfs_zulu'].to_dict()

    # Flag top of inning
    data.loc[:, 'inning_side'] = np.NaN
    data.loc[(
        data['atbat_start_tfs_zulu'] ==
        data['inning_num'].map(min_inning_tfs_dict)
    ), 'inning_side'] = 'top'

    # Flag where outs above == 3 and inning top not flagged
    data.loc[(
        (data['inning_side'].isnull())
        &
        (data['atbat_o'].shift(-1) == 3)
    ), 'inning_side'] == 'bottom'
    data.loc[:, 'inning_side'] = data['inning_side'].ffill()

def add_atbat_team():
    return data
