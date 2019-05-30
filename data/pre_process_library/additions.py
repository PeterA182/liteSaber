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

    assert data_table_key in ['batter', 'pitcher', 'inning']

    # Add inning half if not already in data and tabke key is inning
    if (('inning_half' not in data.columns) & (data_table_key == 'inning')):
        data = add_inning_half(data)

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
    if data_table_key in ['batter', 'pitcher']:
        msk = (data['{}TeamFlag'.format(data_table_key)] == 'home')
    if data_table_key == 'inning':
        msk = (data['inning_half'] == 'bottom')
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

    # Astype Float
    data.loc[:, 'inning_num'] = data['inning_num'].astype(float)
    data.loc[:, 'atbat_o'] = data['atbat_o'].astype(float)
    
    # Inning Start Pitcher
    df_inning_1 = data.loc[
        ((data['inning_num'] == 1) &
         (data['atbat_o'] == 0)),
    :]

    # Min atbat_start per pitcher
    min_pitcher_tfs = df_inning_1.groupby(
        by=['atbat_pitcher'],
        as_index=False
    ).agg({'atbat_start_tfs': 'min'})
    min_pitcher_tfs.rename(
        columns={'atbat_start_tfs': 'pitcher_min_tfs'},
        inplace=True
    )
    
    # Merge back to first inning
    df_inning_1 = pd.merge(
        df_inning_1, min_pitcher_tfs,
        how='left',
        on=['atbat_pitcher']
    )
    df_inning_1['starting_pitcher_flag'] = (
        df_inning_1['atbat_start_tfs'] ==
        df_inning_1['pitcher_min_tfs']
    ).astype(int)
    df_inning_1 = df_inning_1.loc[df_inning_1['starting_pitcher_flag'] == 1, :]
    df_inning_1 = df_inning_1[['atbat_pitcher', 'starting_pitcher_flag']]\
        .drop_duplicates(inplace=False)
    
    # Merge back to entire game
    data = pd.merge(
        data, df_inning_1,
        how='left',
        on=['atbat_pitcher'],
        validate='m:1'
    )
    data['starting_pitcher_flag'].fillna(0, inplace=True)
    return data


def add_inning_half(data):
    """
    """
    data.loc[:, 'atbat_start_tfs'] = data['atbat_start_tfs'].astype(float)
    data.loc[:, 'inning_num'] = data['inning_num'].astype(float)
    data.loc[:, 'atbat_o'] = data['atbat_o'].astype(float)

    # Get min tfs zulu per inning
    min_tfs = data.groupby(
        by=['game_id', 'inning_num'],
        as_index=False
    ).agg({'atbat_start_tfs': 'min'})
    min_tfs['key_term'] = list(zip(min_tfs.game_id, min_tfs.inning_num))
    min_inning_tfs_dict = min_tfs.set_index('key_term')\
        ['atbat_start_tfs'].to_dict()

    # Flag top of inning
    #print(sum(data['game_id'].isnull()))
    #print(sum(data['inning_num'].isnull()))
    data['key_term'] = list(zip(data.game_id, data.inning_num))
    data.loc[:, 'inning_half'] = np.NaN
    data.loc[(
        data['atbat_start_tfs'] ==
        data['key_term'].map(min_inning_tfs_dict)
    ), 'inning_half'] = 'top'

    # Flag where outs above == 3 and inning top not flagged
    data.sort_values(
        by=['game_id', 'inning_num', 'atbat_o'],
        ascending=True,
        inplace=True
    )
    data.loc[(
        (data['inning_half'].isnull())
        &
        (data['atbat_o'].shift(1)==3)
        &
        (data['atbat_o'] == 0)
    ), 'inning_half'] = 'bottom'
    data.loc[:, 'inning_half'] = data['inning_half'].ffill()
    data.drop(labels=['key_term'], axis=1, inplace=True)
    return data


def add_atbat_team():
    return data
