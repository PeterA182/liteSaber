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

    # Assure inning half
    if 'inning_half' not in data.columns:
        data = add_inning_half(data)

    # Min atbat_num
    data.loc[:, 'atbat_num'] = data['atbat_num'].astype(float)
    df = data.loc[data['inning_num'].astype(float) == 1, :].groupby(
        by=['game_id', 'inning_half'],
        as_index=False
    ).agg({'atbat_num': 'min'})
    df.rename(columns={'atbat_num': 'atbat_num_min'}, inplace=True)

    # Merge back
    df = pd.merge(
        data, df,
        how='left',
        on=['game_id', 'inning_half'],
        validate='m:1'
    )
    df = df.loc[:, [
        'game_id', 'inning_half', 'atbat_num', 'atbat_num_min',
        'atbat_pitcher'
    ]].drop_duplicates(inplace=False)
    df.loc[:, 'home_starting_pitcher'] = np.NaN
    df.loc[(
        (df['atbat_num'] == df['atbat_num_min'])
        &
        (df['inning_half'] == 'top')
    ), 'away_starting_pitcher'] = df['atbat_pitcher']
    df.loc[(
        (df['atbat_num'] == df['atbat_num_min'])
         &
        (df['inning_half'] == 'bottom')
    ), 'home_starting_pitcher'] = df['atbat_pitcher'] 
    home_sp = df.loc[df['home_starting_pitcher'].notnull(), :]
    home_sp = home_sp[['game_id', 'home_starting_pitcher']].\
        drop_duplicates(inplace=False)
    away_sp = df.loc[df['away_starting_pitcher'].notnull(), :]
    away_sp = away_sp[['game_id', 'away_starting_pitcher']].\
        drop_duplicates(inplace=False)

    # Merge back
    data = pd.merge(
        data,
        home_sp,
        how='left',
        on=['game_id'],
        validate='m:1'
    )
    data = pd.merge(
        data,
        away_sp,
        how='left',
        on=['game_id'],
        validate='m:1'
    )
    
    t = """
    df = data.loc[data['inning_num'].astype(float) == 1, :]
    df.loc[:, 'atbat_num'] = df['atbat_num'].astype(float)
    df = df.groupby(
        by=['game_id', 'inning_half'],
        as_index=False
    ).agg({'atbat_num': 'min'})
    df.rename(columns={'atbat_num': 'atbat_num_min'}, inplace=True)    
    starters = pd.merge(
        data,
        df,
        how='left',
        on=['game_id', 'inning_half'],
        validate='m:1'
    )
    starters = starters.loc[starters['inning_num'].astype(float) == 1, :]
    starters = starters.loc[:, [
        'game_id', 'atbat_num', 'inning_half',
        'atbat_num_min', 'atbat_pitcher'
    ]].drop_duplicates(inplace=False)
    starters = starters.loc[(
        starters['atbat_num'].astype(float) == starters['atbat_num_min']
    ), :]
    starters = starters.loc[:, ['game_id', 'inning_half', 'atbat_pitcher']].\
        drop_duplicates(inplace=False)
    starters.loc[starters['inning_half'] == 'top',
                 'away_starting_pitcher'] = starters['atbat_pitcher']
    starters.loc[starters['inning_half'] == 'bottom',
                 'home_starting_pitcher'] = starters['atbat_pitcher']
    starters = starters.drop_duplicates(
        subset=['game_id', 'away_starting_pitcher', 'home_starting_pitcher'],
        inplace=False
    )
    assert sum((starters['home_starting_pitcher'].notnull()) &
               (starters['away_starting_pitcher'].notnull())) == 0
    home_starter = starters.loc[starters['home_starting_pitcher'].notnull(), :]
    home_starter = home_starter[['game_id', 'home_starting_pitcher']].drop_duplicates()
    data = pd.merge(
        data,
        home_starter,
        how='left',
        on=['game_id'],
        validate='m:1'
    )
    
    away_starter = starters.loc[starters['away_starting_pitcher'].notnull(), :]
    away_starter = away_starter[['game_id', 'away_starting_pitcher']].drop_duplicates()
    data = pd.merge(
        data,
        away_starter,
        how='left',
        on=['game_id'],
        validate='m:1'
    )"""
    return data



def add_inning_half(data):
    """
    """

    # Check
    if 'inning_half' in data.columns:
        return data

    # Add np.NaN
    data.loc[:, 'inning_half'] = np.NaN

    # DTypes
    data.loc[:, 'atbat_start_tfs'] = data['atbat_start_tfs'].astype(float)
    data.loc[:, 'inning_num'] = data['inning_num'].astype(float)
    data.loc[:, 'atbat_o'] = data['atbat_o'].astype(float)
    data.loc[:, 'atbat_num'] = data['atbat_num'].astype(float)

    # min atbat_num per inning
    min_atbat_num_per_inning = data.groupby(
        by=['game_id', 'inning_num'],
        as_index=False
    ).agg({'atbat_num': 'min'})
    min_atbat_num_per_inning.rename(
        columns={'atbat_num': 'atbat_num_min'}, inplace=True)
    data = pd.merge(
        data,
        min_atbat_num_per_inning[['game_id', 'inning_num', 'atbat_num_min']],
        how='left',
        on=['game_id', 'inning_num'],
        validate='m:1'
    )
    data.loc[
        data['atbat_num'] == data['atbat_num_min'], 'inning_half'] = 'top'
    data.drop(labels=['atbat_num_min'], axis=1, inplace=True)
    data.sort_values(
        by=['game_id', 'inning_num', 'atbat_num', 'atbat_o'],
        ascending=True,
        inplace=True
    )
    
    # Flag start of bottom
    data.loc[(
        (data['inning_half'].isnull())
        &
        (data['atbat_o'].shift(1) == 3)
        &
        (data['atbat_o'].isin([0, 1]))
    ), 'inning_half'] = 'bottom'
    data.loc[:, 'inning_half'] = data['inning_half'].ffill()
    return data


def add_atbat_team():
    return data
