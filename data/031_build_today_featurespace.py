import os
import gc
import sys
import pandas as pd
import datetime as dt
import numpy as np
import utilities as util
CONFIG = util.load_config()

"""
GAMES ARE WIDE 

|| gameID || home_starting_pitcher || away_starting_pitcher || ... ||

"""


def get_matchup_base_table(date):
    """
    """

    # Read in current year Summaries (will be cut down later)
    df_base = pd.read_parquet(
        CONFIG.get('paths').get('raw') + \
        date.strftime("year_%Ymonth_%mday_%d") +
        "/" + "probableStarters.parquet"
    )

    return df_base


def add_starter_details(data):
    """
    """

    last_registries = [
        fname for fname in sorted(os.listdir(ref_dest))[-100:]
    ]
    registry = pd.concat(
        objs=[
            pd.read_parquet(ref_dest + fname) for fname in last_registries
        ],
        axis=0
    )
    for col in ['first_name', 'last_name']:
        registry.loc[:, col] = registry[col].astype(str)
        registry.loc[:, col] = \
            registry[col].apply(lambda x: x.lower().strip())
    registry.reset_index(drop=True, inplace=True)
    registry.drop_duplicates(
        subset=['id'],
        inplace=True
    )
    registry = registry.loc[:, [
        'id', 'height', 'throws', 'weight', 'dob'
    ]]

    # Merge for Home
    data = pd.merge(
        data,
        registry,
        how='left',
        left_on=['home_starting_pitcher'],
        right_on=['id'],
        validate='1:1',
    )
    data.drop(labels=['id'], axis=1, inplace=True)
    data.rename(
        columns={'height': 'home_starting_pitcher_height',
                 'throws': 'home_starting_pitcher_throws',
                 'weight': 'home_starting_pitcher_weight',
                 'dob': 'home_starting_pitcher_dob'},
        inplace=True
    )

    # Merge for Away
    data = pd.merge(
        data,
        registry,
        how='left',
        left_on=['away_starting_pitcher'],
        right_on=['id'],
        validate='1:1',
    )
    data.drop(labels=['id'], axis=1, inplace=True)
    data.rename(
        columns={'height': 'away_starting_pitcher_height',
                 'throws': 'away_starting_pitcher_throws',
                 'weight': 'away_starting_pitcher_weight',
                 'dob': 'away_starting_pitcher_dob'},
        inplace=True
    )
    return data


if __name__ == "__main__":


    # ----------  ----------  ----------
    # Parameters
    date = dt.datetime(year=2019, month=6, day=30)
    bullpen_top_pitcher_count = 6
    pitcher_metrics = [
        'BF', 'ER', 'ERA', 'HitsAllowed', 'Holds',
        'SeasonLosses', 'SeasonWins', 'numberPitches',
        'Outs', 'RunsAllowed', 'Strikes', 'SO'
    ]
    top_batter_count = 12
    batter_metrics = [
        'Assists', 'AB', 'BB', 'FO', 'Avg', 'H',
        'HBP', 'HR', 'Doubles' 'GroundOuts', 'batterLob',
        'OBP', 'OPS', 'R', 'RBI', 'SluggingPct',
        'StrikeOuts', 'Triples'
    ]

    # Paths
    ref_dest = "/Volumes/Transcend/99_reference/"

    # ---------- ---------- ----------
    # Get Basic Matchup Table
    df_matchup_base = get_matchup_base_table(date=date)

    # Narrow to immediate dimensions
    df_matchup_base = df_matchup_base.loc[:, [
        'gameId', 'startingPitcherId', 'probableStarterSide'
    ]]

    # Only keep starting pitcher id that are filled
    df_matchup_base = df_matchup_base.loc[
        df_matchup_base['startingPitcherId'].notnull(), :]

    # Only keep gameIds with both home and away
    vc = df_matchup_base['gameId'].value_counts()
    vc = pd.DataFrame(vc).reset_index(inplace=False)
    vc.columns = ['gameId', 'freq']
    gids = list(vc.loc[vc['freq'] == 2, :]['gameId'])
    df_matchup_base = df_matchup_base.loc[
        df_matchup_base['gameId'].isin(gids), :]

    # Pivot out to flatten
    df_matchup_base = df_matchup_base.pivot_table(
        index=['gameId'],
        columns=['probableStarterSide'],
        values=['startingPitcherId'],
        aggfunc='first'
    )
    df_matchup_base.reset_index(inplace=True)
    df_matchup_base.columns = [
        x[0] if x[1] == '' else x[0]+"_"+x[1] for
        x in df_matchup_base.columns
    ]

    # Rename
    df_matchup_base.rename(
        columns={
            'startingPitcherId_away': 'away_starting_pitcher',
            'startingPitcherId_home': 'home_starting_pitcher'
        },
        inplace=True
    )
    df_matchup_base = df_matchup_base.loc[:, [
        'gameId', 'away_starting_pitcher', 'home_starting_pitcher'
    ]]
    df_matchup_base.to_csv('/Users/peteraltamura/Desktop/df_matchup_base_tomorrow.csv', index=False)
    df_matchup_base = add_starter_details(df_matchup_base)
    df_matchup_base.to_csv('/Users/peteraltamura/Desktop/df_matchup_base_tomorrow_details.csv', index=False)
